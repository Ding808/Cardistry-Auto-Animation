#include "CardCapAnimBaker.h"

#include "CardCapData.h"
#include "CardCapRetargeter.h"
#include "Animation/AnimSequence.h"
#include "Animation/AnimData/IAnimationDataController.h"
#include "Animation/AnimData/IAnimationDataModel.h"
#include "Animation/Skeleton.h"
#include "AssetCompilingManager.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "BoneIndices.h"
#include "Dom/JsonObject.h"
#include "Engine/SkeletalMesh.h"
#include "HAL/FileManager.h"
#include "HAL/IConsoleManager.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "UObject/Package.h"
#include "UObject/SavePackage.h"
#include "UObject/StrongObjectPtr.h"

namespace
{
    FFrameRate RationalFrameRate(double Fps)
    {
        // Preserve common broadcast rates; otherwise represent to 1e-6 fps.
        for (int32 Numerator : {24000, 30000, 60000, 120000, 240000})
        {
            if (FMath::Abs(Fps - double(Numerator) / 1001.0) < 0.00001)
                return FFrameRate(Numerator, 1001);
        }
        int32 Numerator = FMath::RoundToInt(Fps * 1000000.0);
        int32 Denominator = 1000000;
        int32 A = Numerator, B = Denominator;
        while (B != 0) { const int32 R = A % B; A = B; B = R; }
        return FFrameRate(Numerator / A, Denominator / A);
    }

    TArray<TSharedPtr<FJsonValue>> VJson(const FVector& V)
    {
        return {MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)};
    }
}

UAnimSequence* FCardCapAnimBaker::Bake(const FCardCapCapture& Capture, USkeletalMesh* Mesh,
    const FString& PackageName, FString& OutError)
{
    if (!Mesh || !Mesh->GetSkeleton() || !FPackageName::IsValidLongPackageName(PackageName)
        || !FMath::IsFinite(Capture.Meta.Fps) || Capture.Meta.Fps < 0.001 || Capture.Meta.Fps > 1000)
    {
        OutError = TEXT("A valid skeletal mesh, package path and frame rate in [0.001,1000] are required.");
        return nullptr;
    }
    const FString Filename = FPackageName::LongPackageNameToFilename(PackageName, FPackageName::GetAssetPackageExtension());
    if (FPaths::FileExists(Filename) || FindPackage(nullptr, *PackageName))
    {
        OutError = TEXT("Animation destination already exists; use a new asset path.");
        return nullptr;
    }
    FCardCapRetargetResult Poses;
    if (!FCardCapRetargeter::BuildLocalPoseFrames(Capture, Mesh->GetRefSkeleton(), Poses, OutError)) return nullptr;

    UPackage* Package = CreatePackage(*PackageName);
    TStrongObjectPtr<UAnimSequence> Animation(NewObject<UAnimSequence>(Package,
        FName(*FPackageName::GetLongPackageAssetName(PackageName)), RF_Public | RF_Standalone));
    Animation->SetSkeleton(Mesh->GetSkeleton());
    Animation->SetPreviewMesh(Mesh);
    IAnimationDataController& Controller = Animation->GetController();
    Controller.InitializeModel();
    Controller.OpenBracket(FText::FromString(TEXT("Import Cardistry animation")), false);
    Controller.SetFrameRate(RationalFrameRate(Poses.Fps), false);
    Controller.SetNumberOfFrames(FFrameNumber(Poses.FrameCount), false);
    for (int32 Bone = 0; Bone < Poses.BoneNames.Num(); ++Bone)
    {
        TArray<FVector3f> Positions, Scales;
        TArray<FQuat4f> Rotations;
        Positions.Reserve(Poses.FrameCount + 1);
        Scales.Reserve(Poses.FrameCount + 1);
        Rotations.Reserve(Poses.FrameCount + 1);
        for (int32 Key = 0; Key <= Poses.FrameCount; ++Key)
        {
            const FTransform& Local = Poses.Frames[FMath::Min(Key, Poses.FrameCount - 1)].LocalTransforms[Bone];
            FQuat4f Rotation(Local.GetRotation());
            Rotation.Normalize();
            if (!Rotations.IsEmpty() && (Rotation | Rotations.Last()) < 0)
                Rotation = FQuat4f(-Rotation.X, -Rotation.Y, -Rotation.Z, -Rotation.W);
            Positions.Add(FVector3f(Local.GetTranslation()));
            Rotations.Add(Rotation);
            Scales.Add(FVector3f(Local.GetScale3D()));
        }
        if (!Controller.AddBoneCurve(Poses.BoneNames[Bone], false)
            || !Controller.SetBoneTrackKeys(Poses.BoneNames[Bone], Positions, Rotations, Scales, false))
        {
            Controller.CloseBracket(false);
            OutError = FString::Printf(TEXT("Animation controller rejected bone %s."), *Poses.BoneNames[Bone].ToString());
            return nullptr;
        }
    }
    Controller.NotifyPopulated();
    Controller.CloseBracket(false);
    Animation->PostEditChange();
    Animation->WaitOnExistingCompression();
    FAssetCompilingManager::Get().FinishAllCompilation();
    if (!Animation->IsCompressedDataValid())
    {
        OutError = TEXT("Animation compression did not produce valid playback data.");
        return nullptr;
    }
    FAssetRegistryModule::AssetCreated(Animation.Get());
    Package->MarkPackageDirty();
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    SaveArgs.SaveFlags = SAVE_NoError;
    SaveArgs.bSlowTask = false;
    if (!IFileManager::Get().MakeDirectory(*FPaths::GetPath(Filename), true)
        || !UPackage::SavePackage(Package, Animation.Get(), *Filename, SaveArgs))
    {
        OutError = TEXT("Could not save the animation package.");
        return nullptr;
    }
    OutError.Reset();
    return Animation.Get();
}

bool FCardCapAnimBaker::Validate(const FCardCapCapture& Capture, USkeletalMesh* Mesh,
    UAnimSequence* Animation, TSharedRef<FJsonObject> Report, FString& OutError)
{
    if (!Mesh || !Animation || Animation->GetSkeleton() != Mesh->GetSkeleton())
    { OutError = TEXT("Animation and mesh must share a skeleton."); return false; }
    FCardCapRetargetResult Expected;
    if (!FCardCapRetargeter::BuildLocalPoseFrames(Capture, Mesh->GetRefSkeleton(), Expected, OutError)) return false;
    Animation->WaitOnExistingCompression();
    const FReferenceSkeleton& Reference = Mesh->GetRefSkeleton();
    const FReferenceSkeleton& SkeletonReference = Mesh->GetSkeleton()->GetReferenceSkeleton();
    const IAnimationDataModel* Model = Animation->GetDataModel();
    if (!Model || Model->GetNumberOfFrames() != Capture.Meta.FrameCount
        || Model->GetNumberOfKeys() != Capture.Meta.FrameCount + 1 || !Animation->IsCompressedDataValid())
    { OutError = TEXT("Unexpected animation timing, missing data model, or invalid compression."); return false; }
    if (FMath::Abs(Model->GetFrameRate().AsDecimal() - Capture.Meta.Fps) > 0.00001
        || FMath::Abs(Animation->GetPlayLength() - Capture.Meta.FrameCount / Capture.Meta.Fps) > 0.00001)
    { OutError = TEXT("Animation fps or play length differs from the capture timeline."); return false; }
    const IConsoleVariable* ForceRaw = IConsoleManager::Get().FindConsoleVariable(TEXT("a.ForceEvalRawData"));
    if (!ForceRaw || ForceRaw->GetInt() != 0 || !Animation->CompressedData.BoneCompressionCodec
        || !Animation->CompressedData.CompressedDataStructure || Animation->GetCompressedTrackToSkeletonMapTable().IsEmpty())
    { OutError = TEXT("Cannot verify the compressed playback path: forced raw evaluation or missing codec/structure/tracks."); return false; }
    TArray<FName> TrackNames;
    Model->GetBoneTrackNames(TrackNames);
    if (TrackNames.Num() != Reference.GetNum())
    { OutError = TEXT("Animation must retain every target bone track."); return false; }
    TSet<FName> UniqueTracks;
    for (const FName Track : TrackNames)
    {
        if (UniqueTracks.Contains(Track) || Reference.FindBoneIndex(Track) == INDEX_NONE)
        { OutError = TEXT("Animation tracks are duplicated or differ from target bones."); return false; }
        UniqueTracks.Add(Track);
    }

    TSet<int32> WristBones;
    for (const FCardCapHand& Hand : Capture.Hands)
        WristBones.Add(Reference.FindBoneIndex(Capture.BoneMapping.Hands[Hand.Side].BoneNames[0]));
    TArray<TSharedPtr<FJsonValue>> ModeReports;
    bool bPassed = true;
    const double Fps = Model->GetFrameRate().AsDecimal();
    for (bool bRaw : {true, false})
    {
        double MaxLocalTranslationErrorCm = 0, MaxLocalRotationErrorRad = 0;
        double MaxBoneLengthDeviationMm = 0, MaxQuatUnitError = 0, MaxSourceJointDistanceCm = 0, MaxScaleError = 0;
        TArray<TSharedPtr<FJsonValue>> FrameReports;
        // Sample integer keys and midpoints, plus the explicit terminal hold key.
        for (int32 HalfFrame = 0; HalfFrame <= Capture.Meta.FrameCount * 2; ++HalfFrame)
        {
            const double Time = double(HalfFrame) * 0.5 / Fps;
            TArray<FTransform> Components;
            Components.SetNum(Reference.GetNum());
            for (int32 Bone = 0; Bone < Reference.GetNum(); ++Bone)
            {
                const int32 SkeletonBone = SkeletonReference.FindBoneIndex(Reference.GetBoneName(Bone));
                if (SkeletonBone == INDEX_NONE) { OutError = TEXT("Mesh bone absent from animation skeleton."); return false; }
                FTransform Local = Reference.GetRefBonePose()[Bone];
                Animation->GetBoneTransform(Local, FSkeletonPoseBoneIndex(SkeletonBone), Time, bRaw);
                if (Local.ContainsNaN()) { OutError = TEXT("Animation evaluation produced a non-finite transform."); return false; }
                MaxQuatUnitError = FMath::Max(MaxQuatUnitError, FMath::Abs(Local.GetRotation().SizeSquared() - 1.0));
                const FTransform& First = Expected.Frames[FMath::Min(HalfFrame / 2, Capture.Meta.FrameCount - 1)].LocalTransforms[Bone];
                const FTransform& Second = Expected.Frames[FMath::Min(HalfFrame / 2 + 1, Capture.Meta.FrameCount - 1)].LocalTransforms[Bone];
                FTransform Want;
                Want.Blend(First, Second, HalfFrame % 2 == 0 ? 0.0f : 0.5f);
                MaxLocalTranslationErrorCm = FMath::Max(MaxLocalTranslationErrorCm, FVector::Distance(Local.GetTranslation(), Want.GetTranslation()));
                MaxLocalRotationErrorRad = FMath::Max(MaxLocalRotationErrorRad, Local.GetRotation().AngularDistance(Want.GetRotation()));
                MaxScaleError = FMath::Max(MaxScaleError, FVector::Distance(Local.GetScale3D(), Want.GetScale3D()));
                const int32 Parent = Reference.GetParentIndex(Bone);
                Components[Bone] = Parent == INDEX_NONE ? Local : Local * Components[Parent];
                if (Parent != INDEX_NONE && !WristBones.Contains(Bone))
                {
                    const double LengthCm = FVector::Distance(Components[Bone].GetTranslation(), Components[Parent].GetTranslation());
                    MaxBoneLengthDeviationMm = FMath::Max(MaxBoneLengthDeviationMm,
                        FMath::Abs(LengthCm - Reference.GetRefBonePose()[Bone].GetTranslation().Size()) * 10.0);
                }
            }
            if (HalfFrame % 2 == 0 && HalfFrame / 2 < Capture.Meta.FrameCount)
            {
                TSharedRef<FJsonObject> Frame = MakeShared<FJsonObject>();
                Frame->SetNumberField(TEXT("frame"), HalfFrame / 2);
                TArray<TSharedPtr<FJsonValue>> Hands;
                for (const FCardCapHand& Hand : Capture.Hands)
                {
                    const FCardCapHandBoneMapping& Mapping = Capture.BoneMapping.Hands[Hand.Side];
                    TSharedRef<FJsonObject> HandReport = MakeShared<FJsonObject>();
                    HandReport->SetStringField(TEXT("side"), Hand.Side);
                    TArray<TSharedPtr<FJsonValue>> Joints;
                    for (int32 Joint = 0; Joint < Mapping.BoneNames.Num(); ++Joint)
                    {
                        const FVector Point = Components[Reference.FindBoneIndex(Mapping.BoneNames[Joint])].GetTranslation();
                        Joints.Add(MakeShared<FJsonValueArray>(VJson(Point)));
                        MaxSourceJointDistanceCm = FMath::Max(MaxSourceJointDistanceCm, FVector::Distance(Point,
                            Hand.Frames[HalfFrame / 2].JointPositionsCm[Capture.BoneMapping.LandmarkIndices[Joint]]));
                    }
                    HandReport->SetArrayField(TEXT("joint_positions_cm_mano16"), Joints);
                    Hands.Add(MakeShared<FJsonValueObject>(HandReport));
                }
                Frame->SetArrayField(TEXT("hands"), Hands);
                FrameReports.Add(MakeShared<FJsonValueObject>(Frame));
            }
        }
        // These are serialization/playback tolerances, never pose-accuracy targets.
        const bool bModePass = MaxLocalTranslationErrorCm < (bRaw ? 0.001 : 0.05)
            && MaxLocalRotationErrorRad < (bRaw ? 0.002 : 0.02)
            && MaxBoneLengthDeviationMm < 0.01 && MaxQuatUnitError < 0.0001 && MaxScaleError < 0.00001;
        bPassed &= bModePass;
        TSharedRef<FJsonObject> Mode = MakeShared<FJsonObject>();
        Mode->SetStringField(TEXT("mode"), bRaw ? TEXT("raw") : TEXT("compressed"));
        Mode->SetBoolField(TEXT("passed"), bModePass);
        Mode->SetNumberField(TEXT("evaluation_times"), Capture.Meta.FrameCount * 2 + 1);
        Mode->SetNumberField(TEXT("max_local_translation_error_cm"), MaxLocalTranslationErrorCm);
        Mode->SetNumberField(TEXT("max_local_rotation_error_rad"), MaxLocalRotationErrorRad);
        Mode->SetNumberField(TEXT("max_finger_bone_length_deviation_mm"), MaxBoneLengthDeviationMm);
        Mode->SetNumberField(TEXT("max_quaternion_unit_error"), MaxQuatUnitError);
        Mode->SetNumberField(TEXT("max_local_scale_error"), MaxScaleError);
        Mode->SetNumberField(TEXT("max_source_joint_distance_cm"), MaxSourceJointDistanceCm);
        Mode->SetStringField(TEXT("source_distance_note"), TEXT("Agreement with input MANO16 joints; not video ground truth. A custom target may have different rest lengths."));
        Mode->SetArrayField(TEXT("frames"), FrameReports);
        ModeReports.Add(MakeShared<FJsonValueObject>(Mode));
    }
    Report->SetStringField(TEXT("status"), bPassed ? TEXT("passed") : TEXT("failed"));
    const bool bConditionalUnits = (Capture.FormatVersion == TEXT("1.2") || Capture.FormatVersion == TEXT("1.3")) && !Capture.Scale.bHasMetersPerUnit;
    Report->SetStringField(TEXT("coordinate_units"), bConditionalUnits ? TEXT("conditional_ue_units") : TEXT("ue_centimeters"));
    Report->SetBoolField(TEXT("metric_scale_known"), !bConditionalUnits);
    Report->SetBoolField(TEXT("physical_accuracy_validated"), false);
    if (Capture.FormatVersion == TEXT("1.3"))
    {
        Report->SetStringField(TEXT("coordinate_frame"), Capture.Provenance.CoordinateFrame);
        Report->SetBoolField(TEXT("inter_hand_transform_known"), Capture.bInterHandTransformKnown);
        Report->SetStringField(TEXT("local_origin_policy"), Capture.bInterHandTransformKnown
            ? TEXT("Source camera-frame wrist translations are consumed unchanged")
            : TEXT("Each hand is independently wrist-relative. Animation origins are display_only, not measured zero positions and not a shared two-hand space. The source global translations remain null."));
    }
    Report->SetStringField(TEXT("diagnostic_length_units"), bConditionalUnits
        ? TEXT("All legacy *_cm and joint_positions_cm_mano16 fields are conditional UE display units; *_mm fields are those same display units multiplied by 10. These are serialization/playback discrepancies, not physical centimeters, millimeters or measurement accuracy.")
        : TEXT("Legacy *_cm fields use declared centimeters; *_mm fields multiply those values by 10. These are serialization/playback discrepancies, not independently measured motion accuracy."));
    Report->SetStringField(TEXT("animation_path"), Animation->GetPathName());
    Report->SetStringField(TEXT("mesh_path"), Mesh->GetPathName());
    Report->SetNumberField(TEXT("source_frames"), Capture.Meta.FrameCount);
    Report->SetNumberField(TEXT("animation_intervals"), Model->GetNumberOfFrames());
    Report->SetNumberField(TEXT("animation_keys"), Model->GetNumberOfKeys());
    Report->SetNumberField(TEXT("bone_tracks"), TrackNames.Num());
    Report->SetNumberField(TEXT("fps"), Fps);
    Report->SetBoolField(TEXT("compressed_codec_and_structure_present"), true);
    Report->SetNumberField(TEXT("force_raw_console_value"), ForceRaw->GetInt());
    Report->SetNumberField(TEXT("compressed_tracks"), Animation->GetCompressedTrackToSkeletonMapTable().Num());
    Report->SetNumberField(TEXT("play_length_seconds"), Animation->GetPlayLength());
    Report->SetStringField(TEXT("endpoint_policy"), TEXT("Last source pose held one frame; final key is not an additional observation."));
    Report->SetArrayField(TEXT("evaluations"), ModeReports);
    OutError = bPassed ? FString() : TEXT("Actual animation evaluation exceeded serialization or constant-length tolerances.");
    return bPassed;
}
