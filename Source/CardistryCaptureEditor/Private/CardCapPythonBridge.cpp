#include "CardCapPythonBridge.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Dom/JsonObject.h"
#include "Editor.h"
#include "EngineUtils.h"
#include "FileHelpers.h"
#include "HAL/FileManager.h"
#include "Interfaces/IPluginManager.h"
#include "ILevelSequenceEditorToolkit.h"
#include "ISequencer.h"
#include "LevelSequence.h"
#include "LevelEditorViewport.h"
#include "Misc/DateTime.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "Subsystems/AssetEditorSubsystem.h"
#include "UnrealClient.h"

namespace
{
    constexpr int64 MaxStatusBytes = 1024 * 1024;
    bool WriteJson(const FString& File, const TSharedRef<FJsonObject>& Object)
    {
        FString Text;
        if (!FJsonSerializer::Serialize(Object, TJsonWriterFactory<>::Create(&Text))) return false;
        return FFileHelper::SaveStringToFile(Text, *File, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
    }
    FString Absolute(const FString& File)
    {
        return FPaths::ConvertRelativePathToFull(File);
    }
    FString ObjectPath(const FString& Value)
    {
        const FString Package = FPackageName::ObjectPathToPackageName(Value);
        return Package + TEXT(".") + FPackageName::GetLongPackageAssetName(Package);
    }
    bool FiniteNumber(const TSharedPtr<FJsonObject>& Object, const TCHAR* Field, double& Out)
    {
        return Object->TryGetNumberField(Field, Out) && FMath::IsFinite(Out);
    }
}

FCardCapPythonBridge::FCardCapPythonBridge() { RestoreLastJob(); }
FCardCapPythonBridge::~FCardCapPythonBridge() { Shutdown(); }

FString FCardCapPythonBridge::JobsRoot() const
{
    return Absolute(FPaths::ProjectSavedDir() / TEXT("CardistryCapture/Runs"));
}

FString FCardCapPythonBridge::QuoteArgument(const FString& Value)
{
    // Windows argv escaping, not shell interpolation. No shell is launched.
    FString Result = TEXT("\"");
    int32 Slashes = 0;
    for (TCHAR Character : Value)
    {
        if (Character == TEXT('\\')) { ++Slashes; continue; }
        Result += FString::ChrN(Character == TEXT('"') ? Slashes * 2 + 1 : Slashes, TEXT('\\'));
        Slashes = 0;
        Result.AppendChar(Character);
    }
    Result += FString::ChrN(Slashes * 2, TEXT('\\'));
    Result += TEXT("\"");
    return Result;
}

bool FCardCapPythonBridge::Start(const FCardCapJobOptions& Options, FString& OutError)
{
    if (!CanStart()) { OutError = TEXT("The current job is still stopping. Please wait."); return false; }
    const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("CardistryCapture"));
    if (!Plugin) { OutError = TEXT("The Cardistry Capture plugin folder could not be found."); return false; }
    const FString PluginDir = Absolute(Plugin->GetBaseDir());
    const FString PythonDir = PluginDir / TEXT("PythonPipeline");
    const FString Python = PythonDir / TEXT(".venv/Scripts/python.exe");
    const FString Script = PythonDir / TEXT("editor_job.py");
    const FString Project = Absolute(FPaths::GetProjectFilePath());
    const FString Editor = Absolute(FPaths::EngineDir() / TEXT("Binaries/Win64/UnrealEditor-Cmd.exe"));
    const FString Video = Options.VideoPath.TrimStartAndEnd();
    const FString Mapping = Options.BoneMappingPath.IsEmpty()
        ? PluginDir / TEXT("Config/BoneMapping_UE5Mannequin.json") : Absolute(Options.BoneMappingPath);
    if (Video.IsEmpty() || !FPaths::FileExists(Video))
    { OutError = TEXT("Choose a video file that exists on this computer."); return false; }
    if (!FPaths::FileExists(Python) || !FPaths::FileExists(Script))
    { OutError = TEXT("The processing environment is not ready. Run Scripts/Setup.cmd in the plugin folder first."); return false; }
    if (!FPaths::FileExists(Project) || !FPaths::FileExists(Editor))
    { OutError = TEXT("The current project or Unreal background executable could not be found. Open this panel from a saved project."); return false; }
    if (!FPaths::FileExists(Mapping))
    { OutError = TEXT("The bone mapping file does not exist. Choose it again in Advanced Settings."); return false; }

    Snapshot = FCardCapJobSnapshot();
    Snapshot.JobId = TEXT("Take_") + FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S_")) + FGuid::NewGuid().ToString(EGuidFormats::Digits).Left(8);
    Snapshot.OutputDirectory = JobsRoot() / Snapshot.JobId;
    Snapshot.LogPath = Snapshot.OutputDirectory / TEXT("job.log");
    Snapshot.Status = TEXT("running");
    Snapshot.Stage = TEXT("preflight");
    Snapshot.Message = TEXT("Checking the video and processing environment...");
    if (!IFileManager::Get().MakeDirectory(*Snapshot.OutputDirectory, true))
    { OutError = TEXT("The result folder could not be created. Check that the project folder is writable."); Fail(OutError); return false; }
    TSharedRef<FJsonObject> Request = MakeShared<FJsonObject>();
    Request->SetNumberField(TEXT("schema_version"), 1);
    Request->SetStringField(TEXT("job_id"), Snapshot.JobId);
    Request->SetStringField(TEXT("video_path"), Absolute(Video));
    Request->SetStringField(TEXT("plugin_dir"), PluginDir);
    Request->SetStringField(TEXT("project_file"), Project);
    Request->SetStringField(TEXT("editor_executable"), Editor);
    Request->SetStringField(TEXT("output_dir"), Snapshot.OutputDirectory);
    Request->SetStringField(TEXT("bone_mapping_path"), Mapping);
    Request->SetBoolField(TEXT("allow_blurry"), Options.bAllowBlurry);
    Request->SetBoolField(TEXT("render_preview"), Options.bRenderPreview);
    Request->SetStringField(TEXT("display_view_mode"), Options.bUseSeparateLocalPreview ? TEXT("per_hand_local") : TEXT("common_space"));
    const FString RequestFile = Snapshot.OutputDirectory / TEXT("request.json");
    if (!WriteJson(RequestFile, Request))
    { OutError = TEXT("The processing settings could not be saved."); Fail(OutError); return false; }
    bCancelRequested = false;
    StartedAt = FPlatformTime::Seconds();
    LastPoll = 0;
    const FString Arguments = QuoteArgument(Script) + TEXT(" --request ") + QuoteArgument(RequestFile);
    Process = FPlatformProcess::CreateProc(*Python, *Arguments, false, true, true,
        &ProcessId, 0, *PythonDir, nullptr);
    if (!Process.IsValid())
    { OutError = TEXT("The background processor could not be started. Check the setup status."); Fail(OutError); SaveTerminalStatus(); return false; }
    TSharedRef<FJsonObject> Last = MakeShared<FJsonObject>();
    Last->SetStringField(TEXT("job_id"), Snapshot.JobId);
    Last->SetStringField(TEXT("output_dir"), Snapshot.OutputDirectory);
    WriteJson(Absolute(FPaths::ProjectSavedDir() / TEXT("CardistryCapture/last_job.json")), Last);
    return true;
}

bool FCardCapPythonBridge::ParseStatus(const FString& Json, const FString& ExpectedJob,
    FCardCapJobSnapshot& InOut, FString& OutError)
{
    TSharedPtr<FJsonObject> Root;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root)
    { OutError = TEXT("Invalid status JSON"); return false; }
    FString Job, Status;
    double Schema = 0, Progress = 0, Elapsed = 0;
    if (!FiniteNumber(Root, TEXT("schema_version"), Schema) || Schema != 1
        || !Root->TryGetStringField(TEXT("job_id"), Job) || Job != ExpectedJob
        || !Root->TryGetStringField(TEXT("status"), Status)
        || !(Status == TEXT("running") || Status == TEXT("succeeded") || Status == TEXT("failed") || Status == TEXT("cancelled"))
        || !FiniteNumber(Root, TEXT("progress"), Progress) || Progress < 0 || Progress > 1
        || !FiniteNumber(Root, TEXT("elapsed_seconds"), Elapsed) || Elapsed < 0)
    { OutError = TEXT("Status schema, job identity or progress is invalid"); return false; }
    FCardCapJobSnapshot Next = InOut;
    Next.JobId = Job; Next.Status = Status; Next.Progress = Progress; Next.ElapsedSeconds = Elapsed;
    Root->TryGetStringField(TEXT("stage"), Next.Stage);
    Root->TryGetStringField(TEXT("message"), Next.Message);
    FString MessageLanguage;
    Root->TryGetStringField(TEXT("message_language"), MessageLanguage);
    Next.bMessagesAreEnglish = MessageLanguage == TEXT("en");
    Next.HistoricalError.Empty();
    double Frames = 0, Total = 0;
    if (FiniteNumber(Root, TEXT("frames_completed"), Frames) && Frames >= 0 && Frames <= MAX_int32)
        Next.FramesCompleted = static_cast<int32>(Frames);
    if (FiniteNumber(Root, TEXT("frames_total"), Total) && Total >= 0 && Total <= MAX_int32)
        Next.FramesTotal = static_cast<int32>(Total);
    const TSharedPtr<FJsonObject>* Error;
    if (Root->TryGetObjectField(TEXT("error"), Error) && Error && Error->IsValid())
        (*Error)->TryGetStringField(TEXT("message"), Next.Error);
    const TSharedPtr<FJsonObject>* Result;
    if (Root->TryGetObjectField(TEXT("result"), Result) && Result && Result->IsValid())
    {
        (*Result)->TryGetStringField(TEXT("map_asset"), Next.MapAsset);
        (*Result)->TryGetStringField(TEXT("sequence_asset"), Next.SequenceAsset);
        (*Result)->TryGetStringField(TEXT("animation_asset"), Next.AnimationAsset);
        (*Result)->TryGetStringField(TEXT("capture_file"), Next.CaptureFile);
        (*Result)->TryGetStringField(TEXT("preview_video"), Next.PreviewVideo);
        (*Result)->TryGetStringField(TEXT("scale_confidence"), Next.ScaleConfidence);
        (*Result)->TryGetStringField(TEXT("intrinsics_source"), Next.IntrinsicsSource);
        Next.DisplayViewMode.Empty();
        Next.DisplayAssumedFocalPx = 0;
        (*Result)->TryGetStringField(TEXT("display_view_mode"), Next.DisplayViewMode);
        FiniteNumber(*Result, TEXT("display_assumed_focal_px"), Next.DisplayAssumedFocalPx);
        Next.CoordinateFrame.Empty(); // Absent on legacy jobs; never inherit a prior local result.
        if ((*Result)->HasField(TEXT("coordinate_frame")) &&
            (!(*Result)->TryGetStringField(TEXT("coordinate_frame"), Next.CoordinateFrame) ||
             (Next.CoordinateFrame != TEXT("per_hand_wrist_local") && Next.CoordinateFrame != TEXT("shared_camera"))))
        { OutError = TEXT("Invalid result coordinate_frame"); return false; }
        FiniteNumber(*Result, TEXT("fps"), Next.Fps);
        double FrameCount = 0;
        if (FiniteNumber(*Result, TEXT("frame_count"), FrameCount) && FrameCount > 0 && FrameCount <= MAX_int32)
            Next.FramesTotal = static_cast<int32>(FrameCount);
        const TArray<TSharedPtr<FJsonValue>>* Ranges;
        if ((*Result)->TryGetArrayField(TEXT("low_confidence_ranges"), Ranges))
        {
            Next.LowConfidenceRanges.Reset();
            for (const auto& Item : *Ranges)
            {
                const TArray<TSharedPtr<FJsonValue>>* Pair;
                double Start = 0, End = 0;
                if (!Item->TryGetArray(Pair) || Pair->Num() != 2 || !(*Pair)[0]->TryGetNumber(Start)
                    || !(*Pair)[1]->TryGetNumber(End) || !FMath::IsFinite(Start) || !FMath::IsFinite(End)
                    || Start < 0 || End < Start || End >= Next.FramesTotal || Start != FMath::FloorToDouble(Start) || End != FMath::FloorToDouble(End))
                { OutError = TEXT("Invalid low-confidence range"); return false; }
                Next.LowConfidenceRanges.Emplace(static_cast<int32>(Start), static_cast<int32>(End));
            }
        }
    }
    if (Status == TEXT("succeeded") && (Next.MapAsset.IsEmpty() || Next.SequenceAsset.IsEmpty()
        || Next.AnimationAsset.IsEmpty() || Next.CaptureFile.IsEmpty() || Next.FramesTotal < 1 || Next.Fps <= 0))
    { OutError = TEXT("Success status lacks complete outputs"); return false; }
    InOut = MoveTemp(Next);
    return true;
}

bool FCardCapPythonBridge::ReadStatus()
{
    const FString Path = Snapshot.OutputDirectory / TEXT("status.json");
    const int64 Bytes = IFileManager::Get().FileSize(*Path);
    if (Bytes < 1 || Bytes > MaxStatusBytes) return false;
    FString Json, Error;
    return FFileHelper::LoadFileToString(Json, *Path) && ParseStatus(Json, Snapshot.JobId, Snapshot, Error);
}

bool FCardCapPythonBridge::ValidateResults(FString& OutError) const
{
    const FString ProjectPrefix = TEXT("/Game/CardistryCapture/Generated/") + Snapshot.JobId + TEXT("/");
    const FString LegacyPrefix = TEXT("/CardistryCapture/GeneratedResearch/") + Snapshot.JobId + TEXT("/");
    // Restore old results in place; every new job writes into the active project.
    const FString Prefix = Snapshot.MapAsset.StartsWith(ProjectPrefix) ? ProjectPrefix : LegacyPrefix;
    for (const FString* Asset : {&Snapshot.MapAsset, &Snapshot.SequenceAsset, &Snapshot.AnimationAsset})
    {
        const FString Package = FPackageName::ObjectPathToPackageName(*Asset);
        if (!Package.StartsWith(Prefix) || !FPackageName::IsValidLongPackageName(Package))
        { OutError = TEXT("The output asset does not belong to this job and cannot be opened."); return false; }
        const FString Extension = Asset == &Snapshot.MapAsset ? FPackageName::GetMapPackageExtension() : FPackageName::GetAssetPackageExtension();
        if (!FPaths::FileExists(FPackageName::LongPackageNameToFilename(Package, Extension)))
        { OutError = TEXT("The result files are incomplete. Check the job log."); return false; }
    }
    if (!FPaths::IsUnderDirectory(Snapshot.CaptureFile, Snapshot.OutputDirectory) || !FPaths::FileExists(Snapshot.CaptureFile)
        || (!Snapshot.PreviewVideo.IsEmpty() && (!FPaths::IsUnderDirectory(Snapshot.PreviewVideo, Snapshot.OutputDirectory) || !FPaths::FileExists(Snapshot.PreviewVideo))))
    { OutError = TEXT("The result path is invalid or a required file is missing."); return false; }
    return true;
}

void FCardCapPythonBridge::Tick(float DeltaTime)
{
    if (!PendingSequenceAsset.IsEmpty()) SeekOpenedSequence();
    if (!Process.IsValid()) return;
    const double Now = FPlatformTime::Seconds();
    if (Now - LastPoll < 0.25) return;
    LastPoll = Now;
    ReadStatus();
    Snapshot.ElapsedSeconds = Now - StartedAt;
    if (bCancelRequested)
    {
        Snapshot.Status = TEXT("cancelling"); Snapshot.Message = TEXT("Stopping this job...");
        if (Now - CancelledAt > 5.0 && FPlatformProcess::IsProcRunning(Process))
            FPlatformProcess::TerminateProc(Process, true);
    }
    if (FPlatformProcess::IsProcRunning(Process))
    {
        // A status file alone cannot finish the task before its process exits.
        if (!bCancelRequested && Snapshot.Status != TEXT("running")) Snapshot.Status = TEXT("running");
        return;
    }
    int32 Code = -1;
    FPlatformProcess::GetProcReturnCode(Process, &Code);
    const bool bValidStatus = ReadStatus();
    FPlatformProcess::CloseProc(Process); ProcessId = 0;
    if (bCancelRequested)
    {
        // Windows TerminateProcess uses exit code 0. Local cancel intent wins.
        Snapshot.Status = TEXT("cancelled"); Snapshot.Message = TEXT("Cancelled. Existing files remain in the result folder.");
        SaveTerminalStatus();
    }
    else if (!bValidStatus || Code != 0 || Snapshot.Status != TEXT("succeeded"))
    {
        if (Snapshot.Status != TEXT("cancelled"))
            Fail(Snapshot.Error.IsEmpty() ? TEXT("Processing did not complete. Check the log before trying again.") : Snapshot.Error);
        SaveTerminalStatus();
    }
    else
    {
        FString Error;
        if (!ValidateResults(Error)) { Fail(Error); SaveTerminalStatus(); }
    }
}

bool FCardCapPythonBridge::IsTickable() const
{
    return !bShuttingDown && (Process.IsValid() || !PendingSequenceAsset.IsEmpty());
}
TStatId FCardCapPythonBridge::GetStatId() const
{
    RETURN_QUICK_DECLARE_CYCLE_STAT(FCardCapPythonBridge, STATGROUP_Tickables);
}

void FCardCapPythonBridge::Cancel()
{
    if (!Process.IsValid() || bCancelRequested) return;
    bCancelRequested = true; CancelledAt = FPlatformTime::Seconds();
    FFileHelper::SaveStringToFile(TEXT("cancel\n"), *(Snapshot.OutputDirectory / TEXT("cancel.request")));
    Snapshot.Status = TEXT("cancelling"); Snapshot.Message = TEXT("Stopping this job...");
}

void FCardCapPythonBridge::Fail(const FString& Message)
{
    Snapshot.Status = TEXT("failed"); Snapshot.Error = Message; Snapshot.Message = Message;
}

void FCardCapPythonBridge::SaveTerminalStatus()
{
    if (Snapshot.OutputDirectory.IsEmpty()) return;
    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetNumberField(TEXT("schema_version"), 1);
    Root->SetStringField(TEXT("job_id"), Snapshot.JobId);
    Root->SetStringField(TEXT("status"), Snapshot.Status);
    Root->SetStringField(TEXT("stage"), Snapshot.Stage);
    Root->SetStringField(TEXT("message"), Snapshot.Message);
    Root->SetStringField(TEXT("message_language"), TEXT("en"));
    Root->SetNumberField(TEXT("progress"), Snapshot.Progress);
    Root->SetNumberField(TEXT("elapsed_seconds"), Snapshot.ElapsedSeconds);
    TSharedRef<FJsonObject> Error = MakeShared<FJsonObject>();
    Error->SetStringField(TEXT("message"), Snapshot.Error);
    Error->SetStringField(TEXT("log_path"), Snapshot.LogPath);
    Root->SetObjectField(TEXT("error"), Error);
    // Only used after this bridge owns a stopped process, never races the writer.
    WriteJson(Snapshot.OutputDirectory / TEXT("status.json"), Root);
}

void FCardCapPythonBridge::Shutdown()
{
    if (bShuttingDown) return;
    if (Process.IsValid())
    {
        Cancel();
        FPlatformProcess::TerminateProc(Process, true);
        FPlatformProcess::CloseProc(Process); ProcessId = 0;
        Snapshot.Status = TEXT("cancelled"); Snapshot.Message = TEXT("The editor closed and this job was stopped.");
        SaveTerminalStatus();
    }
    PendingSequenceAsset.Empty(); bShuttingDown = true;
}

void FCardCapPythonBridge::RestoreLastJob()
{
    FString Json;
    if (!FFileHelper::LoadFileToString(Json, *Absolute(FPaths::ProjectSavedDir() / TEXT("CardistryCapture/last_job.json")))) return;
    TSharedPtr<FJsonObject> Root;
    if (!FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root) return;
    FString Job, Output;
    if (!Root->TryGetStringField(TEXT("job_id"), Job) || !Root->TryGetStringField(TEXT("output_dir"), Output)
        || !FPaths::IsUnderDirectory(Output, JobsRoot()) || FPaths::GetCleanFilename(Output) != Job) return;
    Snapshot.JobId = Job; Snapshot.OutputDirectory = Output; Snapshot.LogPath = Output / TEXT("job.log");
    if (!ReadStatus()) { Snapshot = FCardCapJobSnapshot(); return; }
    if (!Snapshot.bMessagesAreEnglish) { Snapshot.HistoricalError = Snapshot.Error; }
    if (Snapshot.IsRunning()) Fail(TEXT("The previous job ended unexpectedly. Check the log or start a new job."));
    else if (Snapshot.Status == TEXT("succeeded"))
    {
        FString Error;
        if (!ValidateResults(Error)) Fail(Error);
    }
}

void FCardCapPythonBridge::RefreshResultAssets()
{
    FAssetRegistryModule& Module = FModuleManager::LoadModuleChecked<FAssetRegistryModule>(TEXT("AssetRegistry"));
    const FString Root = Snapshot.MapAsset.StartsWith(TEXT("/Game/CardistryCapture/Generated/"))
        ? TEXT("/Game/CardistryCapture/Generated/") : TEXT("/CardistryCapture/GeneratedResearch/");
    Module.Get().ScanPathsSynchronous({Root + Snapshot.JobId}, true);
}

void FCardCapPythonBridge::OpenResultScene()
{
    if (Snapshot.Status != TEXT("succeeded") || !GEditor) return;
    FString Error;
    if (!ValidateResults(Error)) { Snapshot.Error = Error; return; }
    // Explicit result-open action only. Never discard the user's unsaved map.
    if (!FEditorFileUtils::SaveDirtyPackages(true, true, true, false, false, false)) return;
    TArray<UPackage*> UnsavedMaps;
    FEditorFileUtils::GetDirtyWorldPackages(UnsavedMaps);
    FEditorFileUtils::GetDirtyContentPackages(UnsavedMaps);
    if (!UnsavedMaps.IsEmpty()) { Snapshot.Error = TEXT("The project has unsaved changes. The current scene has been kept open."); return; }
    RefreshResultAssets();
    const FString Package = FPackageName::ObjectPathToPackageName(Snapshot.MapAsset);
    if (!FEditorFileUtils::LoadMap(FPackageName::LongPackageNameToFilename(Package, FPackageName::GetMapPackageExtension())))
    { Snapshot.Error = TEXT("The result scene could not be opened. Check the editor log."); return; }
    UObject* Sequence = LoadObject<ULevelSequence>(nullptr, *ObjectPath(Snapshot.SequenceAsset));
    if (Sequence && GEditor->GetEditorSubsystem<UAssetEditorSubsystem>()->OpenEditorForAsset(Sequence))
    { PendingSequenceAsset = ObjectPath(Snapshot.SequenceAsset); PendingSequenceTicks = 0; }
}

void FCardCapPythonBridge::SeekOpenedSequence()
{
    if (!GEditor || ++PendingSequenceTicks > 120)
    {
        Snapshot.Error = TEXT("The result scene loaded, but its viewport could not be prepared. Open the scene again.");
        PendingSequenceAsset.Empty(); return;
    }
    ULevelSequence* Sequence = FindObject<ULevelSequence>(nullptr, *PendingSequenceAsset);
    if (!Sequence) return;
    IAssetEditorInstance* Instance = GEditor->GetEditorSubsystem<UAssetEditorSubsystem>()->FindEditorForAsset(Sequence, false);
    if (!Instance || Instance->GetEditorName() != FName(TEXT("LevelSequenceEditor"))) return;
    const TSharedPtr<ISequencer> Sequencer = static_cast<ILevelSequenceEditorToolkit*>(Instance)->GetSequencer();
    if (!Sequencer || Sequencer->GetRootMovieSceneSequence() != Sequence) return;
    UWorld* World = GEditor->GetEditorWorldContext().World();
    if (!World) return;
    ACameraActor* DisplayCamera = nullptr;
    const FString CameraPrefix = Snapshot.HasCommonDisplay() ? TEXT("Common display camera")
        : (Snapshot.IsPerHandLocal() ? TEXT("Left local viewer") : TEXT("Capture camera"));
    for (TActorIterator<ACameraActor> It(World); It; ++It)
    {
        if (It->GetActorLabel().StartsWith(CameraPrefix)) { DisplayCamera = *It; break; }
    }
    if (!DisplayCamera)
    {
        Snapshot.Error = TEXT("The result scene has no preview camera. Generate the result again.");
        PendingSequenceAsset.Empty(); return;
    }
    FLevelEditorViewportClient* ResultViewport = nullptr;
    FViewport* ActiveViewport = GEditor->GetActiveViewport();
    for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
    {
        if (!Client || !Client->Viewport || Client->GetWorld() != World || !Client->IsVisible() || !Client->IsPerspective()) continue;
        if (!ResultViewport || Client->Viewport == ActiveViewport) ResultViewport = Client;
        if (Client->Viewport == ActiveViewport) break;
    }
    if (!ResultViewport) return;
    // Sequencer owns cinematic locks. With camera-cut preview disabled, its
    // next evaluation releases even a manually assigned cinematic actor lock.
    // Enable the saved camera-cut track before evaluating the initial pose.
    ResultViewport->SetAllowCinematicControl(true);
    Sequencer->SetPerspectiveViewportPossessionEnabled(true);
    Sequencer->SetPerspectiveViewportCameraCutEnabled(true);
    ResultViewport->SetViewMode(VMI_Lit);
    ResultViewport->SetGameView(true);
    Sequencer->SetGlobalTime(FFrameTime(0), false);
    // SetGlobalTime intentionally skips evaluation when already at zero.
    Sequencer->ForceEvaluate();
    if (ResultViewport->GetCameraComponentForView() != DisplayCamera->GetCameraComponent()) return;
    // The evaluated camera-cut inherits this display camera's aspect ratio and
    // postprocess without piloting or moving any actor. In local mode it is
    // only the left pose viewer, with no reconstructed inter-hand transform.
    ResultViewport->UpdateViewForLockedActor();
    ResultViewport->SetCurrentViewport();
    ResultViewport->Invalidate();
    PendingSequenceAsset.Empty();
}

void FCardCapPythonBridge::OpenResultAnimation()
{
    if (Snapshot.Status != TEXT("succeeded") || !GEditor) return;
    if (Snapshot.IsPerHandLocal())
    { Snapshot.Error = TEXT("The source animation contains separate local hand poses. Choose Open Scene to review the display, including common-space assumptions when available."); return; }
    RefreshResultAssets();
    GEditor->GetEditorSubsystem<UAssetEditorSubsystem>()->OpenEditorForAsset(Snapshot.AnimationAsset);
}
void FCardCapPythonBridge::OpenPreview()
{
    if (Snapshot.Status == TEXT("succeeded") && FPaths::FileExists(Snapshot.PreviewVideo))
        FPlatformProcess::LaunchFileInDefaultExternalApplication(*Snapshot.PreviewVideo);
}
void FCardCapPythonBridge::OpenOutputFolder()
{
    if (FPaths::DirectoryExists(Snapshot.OutputDirectory)) FPlatformProcess::ExploreFolder(*Snapshot.OutputDirectory);
}
void FCardCapPythonBridge::OpenLog()
{
    if (FPaths::FileExists(Snapshot.LogPath)) FPlatformProcess::LaunchFileInDefaultExternalApplication(*Snapshot.LogPath);
}
