#include "Commandlets/CardCapBakeCommandlet.h"
#include "CardCapAnimBaker.h"
#include "CardCapJsonParser.h"
#include "CardCapSequenceBuilder.h"
#include "Animation/AnimSequence.h"
#include "Dom/JsonObject.h"
#include "Engine/SkeletalMesh.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/StrongObjectPtr.h"

DEFINE_LOG_CATEGORY_STATIC(LogCardCapBake, Log, All);

UCardCapBakeCommandlet::UCardCapBakeCommandlet()
{
    // RendererScene::AllocateScene requires GIsClient for a real render scene.
    // -NullRHI still keeps ordinary bake/reload-only invocations headless.
    IsClient = true; IsServer = false; IsEditor = true;
    LogToConsole = true; ShowErrorCount = true;
    HelpUsage = TEXT("-run=CardCapBake -Capture=<cardcap.json> -Mesh=<asset> -Animation=<new package> -Evidence=<json> [-Mapping=<json>] [-ValidateOnly] [-SequenceDirectory=<content path> -RenderDirectory=<new folder> -SequenceName=<name> -DisplayConfig=<display_space.json>]");
}

int32 UCardCapBakeCommandlet::Main(const FString& Params)
{
    FString CaptureFile, MeshPath, AnimationPath, Evidence, Mapping, Error;
    if (!FParse::Value(*Params, TEXT("Capture="), CaptureFile) || !FParse::Value(*Params, TEXT("Mesh="), MeshPath)
        || !FParse::Value(*Params, TEXT("Animation="), AnimationPath) || !FParse::Value(*Params, TEXT("Evidence="), Evidence))
    { UE_LOG(LogCardCapBake, Error, TEXT("%s"), *HelpUsage); return 1; }
    FParse::Value(*Params, TEXT("Mapping="), Mapping);
    FCardCapCapture Capture;
    if (!FCardCapJsonParser::ParseFile(CaptureFile, Capture, Error, Mapping))
    { UE_LOG(LogCardCapBake, Error, TEXT("%s"), *Error); return 2; }
    const bool bPerHandLocal = Capture.FormatVersion == TEXT("1.3") && !Capture.bInterHandTransformKnown;
    if ((Capture.FormatVersion == TEXT("1.2") || Capture.FormatVersion == TEXT("1.3")) && !bPerHandLocal && !Capture.Camera.bHasIntrinsics && Params.Contains(TEXT("SequenceDirectory=")))
    { UE_LOG(LogCardCapBake, Error, TEXT("camera.intrinsics is null: animation-only bake is supported, but a visible camera preview needs an explicitly sourced conditional or calibrated projection; UE will not invent one.")); return 2; }
    TStrongObjectPtr<USkeletalMesh> Mesh(LoadObject<USkeletalMesh>(nullptr, *MeshPath));
    const bool bValidateOnly = FParse::Param(*Params, TEXT("ValidateOnly"));
    TStrongObjectPtr<UAnimSequence> Animation(bValidateOnly
        ? LoadObject<UAnimSequence>(nullptr, *AnimationPath)
        : FCardCapAnimBaker::Bake(Capture, Mesh.Get(), AnimationPath, Error));
    if (!Animation.IsValid())
    { UE_LOG(LogCardCapBake, Error, TEXT("Animation unavailable: %s"), *Error); return 3; }
    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("capture_file"), FPaths::ConvertRelativePathToFull(CaptureFile));
    Report->SetBoolField(TEXT("reloaded_in_fresh_process"), bValidateOnly);
    const bool bPassed = FCardCapAnimBaker::Validate(Capture, Mesh.Get(), Animation.Get(), Report, Error);
    Report->SetStringField(TEXT("error"), Error);
    FString Json;
    if (!FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&Json))
        || !IFileManager::Get().MakeDirectory(*FPaths::GetPath(Evidence), true)
        || !FFileHelper::SaveStringToFile(Json, *Evidence, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    { UE_LOG(LogCardCapBake, Error, TEXT("Could not write evidence.")); return 4; }
    if (!bPassed) { UE_LOG(LogCardCapBake, Error, TEXT("%s"), *Error); return 5; }
    FString SequenceDirectory, RenderDirectory, SequenceName;
    if (FParse::Value(*Params, TEXT("SequenceDirectory="), SequenceDirectory))
    {
        if (!FParse::Value(*Params, TEXT("RenderDirectory="), RenderDirectory)
            || !FParse::Value(*Params, TEXT("SequenceName="), SequenceName))
        { UE_LOG(LogCardCapBake, Error, TEXT("SequenceDirectory also requires RenderDirectory and SequenceName.")); return 6; }
        FCardCapSequenceBuildSettings Settings;
        Settings.PackageDirectory = SequenceDirectory;
        Settings.AssetName = SequenceName;
        Settings.DisplayRate = Animation->GetSamplingFrameRate();
        Settings.FrameCount = Capture.Meta.FrameCount;
        Settings.Resolution = Capture.Meta.Resolution;
        Settings.Fx = Capture.Camera.Fx; Settings.Fy = Capture.Camera.Fy;
        Settings.Cx = Capture.Camera.Cx; Settings.Cy = Capture.Camera.Cy;
        Settings.bCalibrated = Capture.Camera.bCalibrated;
        Settings.bPerHandLocalPreview = bPerHandLocal;
        if (bPerHandLocal)
        {
            Settings.LeftWristBone = Capture.BoneMapping.Hands.FindChecked(TEXT("left")).BoneNames[0];
            Settings.RightWristBone = Capture.BoneMapping.Hands.FindChecked(TEXT("right")).BoneNames[0];
        }
        if (Capture.FormatVersion == TEXT("1.2") || Capture.FormatVersion == TEXT("1.3"))
        {
            switch (Capture.Provenance.CameraIntrinsics)
            {
                case ECardCapProvenance::Observed: Settings.CameraSourceLabel = TEXT("observed intrinsics"); break;
                case ECardCapProvenance::Calibrated: Settings.CameraSourceLabel = TEXT("calibrated"); break;
                case ECardCapProvenance::UserMeasured: Settings.CameraSourceLabel = TEXT("user-measured intrinsics"); break;
                case ECardCapProvenance::Inferred: Settings.CameraSourceLabel = TEXT("inferred, conditional projection"); break;
                default: Settings.CameraSourceLabel = TEXT("unobservable"); break;
            }
            Settings.CoordinateUnits = Capture.Provenance.CoordinateUnits;
            if (bPerHandLocal) Settings.CameraSourceLabel = TEXT("unobservable source camera; separate local display_only viewers");
            Settings.DistortionPolicy = Capture.Camera.bHasDistortion
                ? TEXT("Ideal pinhole preview; supplied capture pixel-space distortion remains in the capture metadata")
                : TEXT("Physical lens distortion is unknown; ideal pinhole preview is a declared rendering condition, not a measured zero-distortion lens");
        }
        FString DisplayConfig;
        if (FParse::Value(*Params, TEXT("DisplayConfig="), DisplayConfig))
        {
            if (!bPerHandLocal || Capture.Camera.bHasIntrinsics
                || !FCardCapDisplaySpaceData::Load(DisplayConfig, CaptureFile, Settings.FrameCount,
                    Capture.Meta.Fps, Settings.Resolution, Settings.DisplaySpace, Error))
            { UE_LOG(LogCardCapBake, Error, TEXT("Display-only common space requires unchanged local capture with null source intrinsics: %s"), *Error); return 6; }
            Settings.bPerHandLocalPreview = false;
            Settings.bAssumedCommonDisplay = true;
            Settings.Fx = Settings.Fy = Settings.DisplaySpace.AssumedFocalPx;
            Settings.Cx = Settings.DisplaySpace.PrincipalPoint.X;
            Settings.Cy = Settings.DisplaySpace.PrincipalPoint.Y;
            Settings.bCalibrated = false;
            Settings.CameraSourceLabel = TEXT("display-only assumed focal length; uncalibrated; source intrinsics remain null");
            Settings.CoordinateUnits = TEXT("conditional_ue_display_units");
        }
        FCardCapSequenceBuildResult Built;
        bool bRendered = FCardCapSequenceBuilder::Build(Mesh.Get(), Animation.Get(), Settings, Built, Error);
        if (bRendered)
            bRendered = FCardCapSequenceBuilder::CaptureFrames(Built, Settings,
                FPaths::ConvertRelativePathToFull(RenderDirectory), MakeShared<FJsonObject>(), Error);
        FCardCapSequenceBuilder::ReleaseWorld(Built);
        if (!bRendered) { UE_LOG(LogCardCapBake, Error, TEXT("Sequence render failed: %s"), *Error); return 7; }
    }
    UE_LOG(LogCardCapBake, Display, TEXT("Animation baked/reloaded and evaluated successfully: %s"), *Animation->GetPathName());
    return 0;
}
