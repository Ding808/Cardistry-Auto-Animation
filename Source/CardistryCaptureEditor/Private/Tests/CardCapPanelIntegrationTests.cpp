#include "CardCapEditorModule.h"
#include "CardCapPythonBridge.h"
#include "SCardCapPanel.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "Async/AsyncResult.h"
#include "Camera/CameraComponent.h"
#include "CardCapJsonParser.h"
#include "CardCapSequenceBuilder.h"
#include "Dom/JsonObject.h"
#include "DriverConfiguration.h"
#include "Editor.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "FileHelpers.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/Docking/TabManager.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "IAutomationDriver.h"
#include "IAutomationDriverModule.h"
#include "IDriverElement.h"
#include "IDriverSequence.h"
#include "ImageUtils.h"
#include "InputCoreTypes.h"
#include "ILevelSequenceEditorToolkit.h"
#include "ISequencer.h"
#include "LevelEditorViewport.h"
#include "LevelSequence.h"
#include "Layout/Children.h"
#include "LocateBy.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/DefaultValueHelper.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "SceneView.h"
#include "Subsystems/AssetEditorSubsystem.h"
#include "UnrealClient.h"
#include "UObject/Package.h"
#include "Widgets/SWindow.h"

// These are opt-in tests of the real editor and real worker, never mock jobs.
namespace CardCapPanelIntegration
{
enum class ECase { RealVideo, Cancel, InvalidPath };

TSharedPtr<SWidget> FindTaggedWidget(const TSharedRef<SWidget>& Widget, FName Tag)
{
    if (Widget->GetTag() == Tag) return Widget;
    FChildren* Children = Widget->GetChildren();
    for (int32 Index = 0; Children && Index < Children->Num(); ++Index)
    {
        TSharedPtr<SWidget> Found = FindTaggedWidget(Children->GetChildAt(Index), Tag);
        if (Found.IsValid()) return Found;
    }
    return nullptr;
}

class FEditorHeartbeat final : public FTickableEditorObject
{
public:
    bool bMeasure = false;
    int32 Ticks = 0;
    double First = 0, Last = 0, MaxGap = 0;
    TSet<int32> OccupiedSeconds;
    virtual void Tick(float) override
    {
        if (!bMeasure) { return; }
        const double Now = FPlatformTime::Seconds();
        if (Ticks == 0) { First = Now; }
        else { MaxGap = FMath::Max(MaxGap, Now - Last); }
        Last = Now;
        ++Ticks;
        OccupiedSeconds.Add(static_cast<int32>(Now - First));
    }
    virtual TStatId GetStatId() const override
    { RETURN_QUICK_DECLARE_CYCLE_STAT(FCardCapIntegrationHeartbeat, STATGROUP_Tickables); }
};

class FPanelCase final : public IAutomationLatentCommand
{
public:
    FPanelCase(FAutomationTestBase* InTest, ECase InCase, FString InVideo, FString InDirectory,
        int32 InExpectedFrames, double InExpectedFps)
        : Test(InTest), Case(InCase), Video(MoveTemp(InVideo)), Directory(MoveTemp(InDirectory)),
          Started(FPlatformTime::Seconds()), Heartbeat(MakeUnique<FEditorHeartbeat>())
    { ExpectedFrames = InExpectedFrames; ExpectedFps = InExpectedFps; }

    virtual ~FPanelCase() override
    {
        for (auto& Entry : ObservedProcesses) { FPlatformProcess::CloseProc(Entry.Value); }
        ActiveSequence.Reset();
        Driver.Reset();
        if (bOwnsDriver) { IAutomationDriverModule::Get().Disable(); }
    }

    virtual bool Update() override
    {
        const double Now = FPlatformTime::Seconds();
        if (!Failure.IsEmpty()) { return FinishFailure(Now); }
        if (Now - Started > 900.) { return Fail(TEXT("Real panel test exceeded 900 seconds.")); }
        if (bWaitingBool)
        {
            if (!BoolResult.GetFuture().IsReady())
            {
                if (Now - ActionStarted > 60.) { return Fail(TEXT("AutomationDriver action timed out: ") + PendingAction); }
                return false;
            }
            bWaitingBool = false;
            ActiveSequence.Reset();
            if (!BoolResult.GetFuture().Get()) { return Fail(TEXT("AutomationDriver action failed: ") + PendingAction); }
            Actions.Add(MakeShared<FJsonValueString>(PendingAction));
            ++Step;
        }
        if (bWaitingText)
        {
            if (!TextResult.GetFuture().IsReady())
            {
                if (Now - ActionStarted > 60.) { return Fail(TEXT("AutomationDriver text read timed out.")); }
                return false;
            }
            bWaitingText = false;
            VisibleText = TextResult.GetFuture().Get().ToString();
            ++Step;
        }

        switch (Step)
        {
        case 0:
        {
            if (IAutomationDriverModule::Get().IsEnabled())
            { return Fail(TEXT("Another AutomationDriver owner is active; this test will not take over.")); }
            FCardCapEditorModule& Module = FModuleManager::LoadModuleChecked<FCardCapEditorModule>(TEXT("CardistryCaptureEditor"));
            Bridge = Module.GetBridge();
            if (!Bridge.IsValid() || !Bridge->CanStart())
            { return Fail(TEXT("Panel service unavailable or another job is active; no existing job was touched.")); }
            PreviousJob = Bridge->GetSnapshot().JobId;
            Module.OpenPanel(); // Opens the real registered tab; does not call any processing callback.
            Panel = Module.GetPanel();
            if (!Panel.IsValid()) { return Fail(TEXT("Registered tab did not create the actual panel.")); }
            IAutomationDriverModule::Get().Enable();
            bOwnsDriver = true;
            const TSharedRef<FDriverConfiguration, ESPMode::ThreadSafe> Config = MakeShared<FDriverConfiguration, ESPMode::ThreadSafe>();
            Config->ImplicitWait = FTimespan::FromSeconds(8.);
            Config->ExecutionSpeedMultiplier = 1.f;
            Driver = IAutomationDriverModule::Get().CreateAsyncDriver(Config);
            Step = 1;
            ActionStarted = Now;
            return false;
        }
        case 1:
            if (!FSlateApplication::Get().FindWidgetWindow(Panel.ToSharedRef()).IsValid()
                || Panel->GetCachedGeometry().GetAbsoluteSize().X < 100.)
            {
                if (Now - ActionStarted > 15.) { return Fail(TEXT("Real panel has no visible native window/geometry.")); }
                return false;
            }
            if (!Screenshot(TEXT("panel_initial"))) { return Fail(TEXT("Initial real Slate screenshot failed.")); }
            BeginClick(TEXT("CardCap.VideoPath"), TEXT("Click CardCap.VideoPath"));
            return false;
        case 2:
            ActiveSequence = Driver->CreateSequence();
            // UE 5.4 InternalPress emits OnKeyChar even after a printable Ctrl
            // shortcut was handled. Select using navigation keys, then delete.
            ActiveSequence->Actions().TypeChord(By::Path(FString(TEXT("CardCap.VideoPath"))), EKeys::LeftControl, EKeys::Home)
                .TypeChord(EKeys::LeftControl, EKeys::LeftShift, EKeys::End).Type(EKeys::BackSpace);
            BeginBool(ActiveSequence->Perform(), TEXT("Ctrl+Home, Ctrl+Shift+End, Backspace CardCap.VideoPath"));
            return false;
        case 3:
            ActiveSequence = Driver->CreateSequence();
            ActiveSequence->Actions().Type(By::Path(FString(TEXT("CardCap.VideoPath"))), Video);
            BeginBool(ActiveSequence->Perform(), TEXT("Type actual video path"));
            return false;
        case 4:
            BeginText(TEXT("CardCap.VideoPath"));
            return false;
        case 5:
            if (VisibleText != Video) { return Fail(TEXT("Actual text box does not contain the typed path.")); }
            if (!Screenshot(TEXT("panel_ready"))) { return Fail(TEXT("Ready screenshot failed.")); }
            bActualInputVerified = true;
            BeginClick(TEXT("CardCap.Start"), TEXT("Click CardCap.Start"));
            return false;
        case 6:
            if (Case == ECase::InvalidPath)
            {
                BeginText(TEXT("CardCap.Message"));
                return false;
            }
            if (Bridge->GetProcessId() == 0 || Bridge->GetSnapshot().JobId == PreviousJob)
            {
                if (Now - ActionStarted > 15.) { return Fail(TEXT("Actual Start click did not launch a new worker. ") + Bridge->GetSnapshot().Error); }
                return false;
            }
            JobId = Bridge->GetSnapshot().JobId;
            WorkerId = Bridge->GetProcessId();
            WorkerStarted = Now;
            ObserveProcess(WorkerId);
            if (!ObservedProcesses.Contains(WorkerId)) { return Fail(TEXT("Could not retain the actual worker process handle.")); }
            Heartbeat->bMeasure = true;
            if (!Screenshot(TEXT("panel_running"))) { return Fail(TEXT("Running screenshot failed.")); }
            Step = 7;
            return false;
        case 7:
            if (Case == ECase::InvalidPath)
            {
                if (Bridge->GetProcessId() != 0 || Bridge->GetSnapshot().JobId != PreviousJob)
                { return Fail(TEXT("Invalid input unexpectedly launched/replaced a job.")); }
                if (!VisibleText.Contains(TEXT("请选择本机存在的视频文件")))
                { return Fail(TEXT("Expected actual visible missing-video error; got: ") + VisibleText); }
                if (!Screenshot(TEXT("panel_invalid_path"))) { return Fail(TEXT("Invalid-path screenshot failed.")); }
                return Complete();
            }
            RecordSnapshot();
            if (Case == ECase::Cancel)
            {
                ObserveChildren(Now);
                if (ObservedProcesses.Num() < 2 || Heartbeat->Ticks < 10 || Now - WorkerStarted < 1.)
                {
                    if (!Bridge->GetSnapshot().IsRunning() || Now - WorkerStarted > 45.)
                    { return Fail(TEXT("No real running child observed before cancellation; cancellation test is not satisfied.")); }
                    return false;
                }
                if (!Screenshot(TEXT("panel_before_cancel"))) { return Fail(TEXT("Before-cancel screenshot failed.")); }
                BeginClick(TEXT("CardCap.Cancel"), TEXT("Click CardCap.Cancel with running child"));
                return false;
            }
            if (Bridge->GetProcessId() != 0 || Bridge->GetSnapshot().IsRunning()) { return false; }
            if (Bridge->GetSnapshot().Status != TEXT("succeeded"))
            { return Fail(TEXT("Real video job did not succeed: ") + Bridge->GetSnapshot().Error); }
            {
                FString ValidationError;
                if (!CheckSuccess(ValidationError)) { return Fail(ValidationError); }
            }
            if (!Screenshot(TEXT("panel_success"))) { return Fail(TEXT("Success screenshot failed.")); }
            return Complete();
        case 8:
            RecordSnapshot();
            ObserveChildren(Now);
            if (Bridge->GetProcessId() != 0 || AnyObservedProcessAlive())
            {
                if (Now - ActionStarted > 30.) { return Fail(TEXT("Cancel did not stop the observed worker and child processes.")); }
                return false;
            }
            if (Bridge->GetSnapshot().Status != TEXT("cancelled"))
            { return Fail(TEXT("Cancel did not produce cancelled status.")); }
            if (!FPaths::FileExists(Bridge->GetSnapshot().OutputDirectory / TEXT("cancel.request")))
            { return Fail(TEXT("Actual cancel request file missing.")); }
            if (Heartbeat->Ticks < 10) { return Fail(TEXT("Insufficient actual editor heartbeat while worker ran.")); }
            if (!Screenshot(TEXT("panel_cancelled"))) { return Fail(TEXT("Cancelled screenshot failed.")); }
            return Complete();
        default:
            return Fail(TEXT("Unexpected integration test step."));
        }
    }

private:
    TSharedRef<IAsyncDriverElement, ESPMode::ThreadSafe> Element(const TCHAR* Tag)
    { return Driver->FindElement(By::Path(FString(Tag))); }

    void BeginClick(const TCHAR* Tag, const FString& Action)
    {
        // UE 5.4's convenience element Click creates a temporary sequence, but
        // StepExecutor schedules only a weak self reference. Retain the real
        // sequence until its future completes so queued input survives this call.
        ActiveSequence = Driver->CreateSequence();
        ActiveSequence->Actions().Click(By::Path(FString(Tag)));
        BeginBool(ActiveSequence->Perform(), Action);
    }

    void BeginBool(TAsyncResult<bool>&& Result, const FString& Action)
    {
        BoolResult = MoveTemp(Result);
        PendingAction = Action;
        ActionStarted = FPlatformTime::Seconds();
        bWaitingBool = true;
    }
    void BeginText(const TCHAR* Tag)
    {
        TextResult = Element(Tag)->GetText();
        ActionStarted = FPlatformTime::Seconds();
        bWaitingText = true;
    }
    bool Screenshot(const FString& Name)
    {
        TArray<FColor> Pixels;
        FIntVector Size(0, 0, 0);
        if (!FSlateApplication::Get().TakeScreenshot(Panel.ToSharedRef(), Pixels, Size)
            || Size.X <= 0 || Size.Y <= 0 || Pixels.Num() != Size.X * Size.Y) { return false; }
        TArray64<uint8> Png;
        FImageUtils::PNGCompressImageArray(Size.X, Size.Y, TArrayView64<const FColor>(Pixels.GetData(), Pixels.Num()), Png);
        const FString Path = Directory / (Name + TEXT(".png"));
        if (Png.IsEmpty() || !FFileHelper::SaveArrayToFile(Png, *Path)) { return false; }
        const TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
        Entry->SetStringField(TEXT("path"), Path);
        Entry->SetNumberField(TEXT("width"), Size.X);
        Entry->SetNumberField(TEXT("height"), Size.Y);
        Entry->SetStringField(TEXT("method"), TEXT("FSlateApplication::TakeScreenshot(actual SCardCapPanel)"));
        Screenshots.Add(MakeShared<FJsonValueObject>(Entry));
        return true;
    }
    void ObserveProcess(uint32 Id)
    {
        if (Id != 0 && !ObservedProcesses.Contains(Id))
        {
            FProcHandle Handle = FPlatformProcess::OpenProcess(Id);
            if (Handle.IsValid()) { ObservedProcesses.Add(Id, Handle); }
        }
    }
    void ObserveChildren(double Now)
    {
        if (Now - LastEnumeration < .25) { return; }
        LastEnumeration = Now;
#if PLATFORM_WINDOWS
        TArray<TPair<uint32, uint32>> Relations;
        FPlatformProcess::FProcEnumerator Processes;
        while (Processes.MoveNext())
        {
            const FPlatformProcess::FProcEnumInfo Info = Processes.GetCurrent();
            Relations.Emplace(Info.GetPID(), Info.GetParentPID());
        }
        // Work only from the worker/descendants whose handles we observed alive.
        // Repeated passes support children listed before their parents in the snapshot.
        for (int32 Pass = 0; Pass < 4; ++Pass)
        {
            for (const auto& Pair : Relations)
            {
                FProcHandle* Parent = ObservedProcesses.Find(Pair.Value);
                if (Parent && FPlatformProcess::IsProcRunning(*Parent)) { ObserveProcess(Pair.Key); }
            }
        }
#endif
    }
    bool AnyObservedProcessAlive()
    {
        for (auto& Entry : ObservedProcesses)
        { if (FPlatformProcess::IsProcRunning(Entry.Value)) { return true; } }
        return false;
    }
    void RecordSnapshot()
    {
        const FCardCapJobSnapshot& State = Bridge->GetSnapshot();
        const FString Key = State.Status + TEXT(":") + State.Stage;
        if (Key == PreviousStage) { return; }
        PreviousStage = Key;
        const TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
        Entry->SetStringField(TEXT("status"), State.Status);
        Entry->SetStringField(TEXT("stage"), State.Stage);
        Entry->SetNumberField(TEXT("elapsed_test_seconds"), FPlatformTime::Seconds() - Started);
        Entry->SetNumberField(TEXT("editor_heartbeat_ticks"), Heartbeat->Ticks);
        Stages.Add(MakeShared<FJsonValueObject>(Entry));
    }
    bool CheckSuccess(FString& OutError)
    {
        const FCardCapJobSnapshot& State = Bridge->GetSnapshot();
        const FString Prefix = TEXT("/Game/CardistryCapture/Generated/") + JobId + TEXT("/");
        if (State.FramesTotal != ExpectedFrames || State.Fps != ExpectedFps)
        {
            OutError = FString::Printf(TEXT("Test fixture mismatch: expected %d frames at %.17g fps, got %d frames at %.17g fps. The backend status is %s; use CardCapUITestExpectedFrames/CardCapUITestExpectedFps for this source."),
                ExpectedFrames, ExpectedFps, State.FramesTotal, State.Fps, *State.Status);
            return false;
        }
        if (State.IsPerHandLocal())
        {
            const TSharedPtr<SWidget> OpenAnimation = FindTaggedWidget(Panel.ToSharedRef(), FName(TEXT("CardCap.OpenAnimation")));
            if (!OpenAnimation.IsValid() || OpenAnimation->IsEnabled())
            {
                OutError = TEXT("The actual local-preview panel must disable the combined raw animation viewer.");
                return false;
            }
        }
        const bool bValid = State.JobId == JobId
            && State.MapAsset.StartsWith(Prefix) && State.SequenceAsset.StartsWith(Prefix)
            && State.AnimationAsset.StartsWith(Prefix)
            && FPaths::FileExists(State.CaptureFile) && FPaths::FileExists(State.PreviewVideo)
            && FPaths::FileExists(State.OutputDirectory / TEXT("UE_Import.json"))
            && FPaths::FileExists(State.OutputDirectory / TEXT("UE_Bake.json"))
            && FPaths::FileExists(State.OutputDirectory / TEXT("UE_Reload.json"))
            && Heartbeat->Ticks >= 10 && Heartbeat->OccupiedSeconds.Num() >= 3
            && !AnyObservedProcessAlive();
        if (!bValid) { OutError = TEXT("Success is missing actual job-bound outputs or editor heartbeat evidence."); }
        return bValid;
    }
    bool Fail(const FString& Message)
    {
        if (Failure.IsEmpty())
        {
            Failure = Message;
            Test->AddError(Message);
            FailureAt = FPlatformTime::Seconds();
            if (Panel.IsValid()) { Screenshot(TEXT("panel_failure")); }
            if (Bridge.IsValid() && !JobId.IsEmpty() && Bridge->GetSnapshot().JobId == JobId
                && Bridge->GetProcessId() != 0)
            {
                // Emergency cleanup only after a failed test, never counted as a UI action.
                bEmergencyCleanup = true;
                Bridge->Cancel();
            }
        }
        return false;
    }
    bool FinishFailure(double Now)
    {
        if (bEmergencyCleanup && Bridge->GetProcessId() != 0 && Now - FailureAt < 30.) { return false; }
        WriteReceipt(false);
        return true;
    }
    bool Complete()
    {
        Heartbeat->bMeasure = false;
        if (!WriteReceipt(true)) { Test->AddError(TEXT("Could not save panel integration receipt.")); }
        return true;
    }
    bool WriteReceipt(bool bPassed)
    {
        const TSharedRef<FJsonObject> Receipt = MakeShared<FJsonObject>();
        Receipt->SetStringField(TEXT("status"), bPassed ? TEXT("passed") : TEXT("failed"));
        Receipt->SetStringField(TEXT("case"), Case == ECase::RealVideo ? TEXT("PanelRealVideo") : Case == ECase::Cancel ? TEXT("PanelCancel") : TEXT("PanelInvalidPath"));
        Receipt->SetBoolField(TEXT("actual_text_input_verified"), bActualInputVerified);
        Receipt->SetBoolField(TEXT("mock_backend"), false);
        Receipt->SetBoolField(TEXT("emergency_direct_cancel_cleanup"), bEmergencyCleanup);
        Receipt->SetStringField(TEXT("input_path"), Video);
        Receipt->SetStringField(TEXT("job_id"), JobId);
        Receipt->SetNumberField(TEXT("worker_pid_observed"), WorkerId);
        Receipt->SetNumberField(TEXT("expected_frames"), ExpectedFrames);
        Receipt->SetNumberField(TEXT("expected_fps"), ExpectedFps);
        Receipt->SetStringField(TEXT("failure"), Failure);
        Receipt->SetStringField(TEXT("visible_message_last_read"), VisibleText);
        Receipt->SetNumberField(TEXT("elapsed_seconds"), FPlatformTime::Seconds() - Started);
        Receipt->SetNumberField(TEXT("editor_tick_count_during_worker"), Heartbeat->Ticks);
        Receipt->SetNumberField(TEXT("editor_tick_max_gap_seconds"), Heartbeat->MaxGap);
        Receipt->SetNumberField(TEXT("editor_tick_occupied_seconds"), Heartbeat->OccupiedSeconds.Num());
        Receipt->SetArrayField(TEXT("actual_driver_actions_completed"), Actions);
        Receipt->SetArrayField(TEXT("actual_slate_screenshots"), Screenshots);
        Receipt->SetArrayField(TEXT("observed_stage_transitions"), Stages);
        TArray<TSharedPtr<FJsonValue>> ProcessRecords;
        for (auto& Entry : ObservedProcesses)
        {
            const TSharedRef<FJsonObject> Process = MakeShared<FJsonObject>();
            Process->SetNumberField(TEXT("pid"), Entry.Key);
            Process->SetBoolField(TEXT("alive_at_receipt"), FPlatformProcess::IsProcRunning(Entry.Value));
            Process->SetStringField(TEXT("observation"), Entry.Key == WorkerId ? TEXT("bridge_owned_worker") : TEXT("enumerated_descendant_with_retained_handle"));
            ProcessRecords.Add(MakeShared<FJsonValueObject>(Process));
        }
        Receipt->SetArrayField(TEXT("observed_processes"), ProcessRecords);
        Receipt->SetStringField(TEXT("process_observation_scope"), TEXT("Retained actual process handles; enumeration is sampled, not a claim to have observed every transient descendant."));
        if (Bridge.IsValid())
        {
            const FCardCapJobSnapshot& State = Bridge->GetSnapshot();
            Receipt->SetStringField(TEXT("final_status"), State.Status);
            Receipt->SetStringField(TEXT("output_directory"), State.OutputDirectory);
            Receipt->SetStringField(TEXT("map_asset"), State.MapAsset);
            Receipt->SetStringField(TEXT("sequence_asset"), State.SequenceAsset);
            Receipt->SetStringField(TEXT("animation_asset"), State.AnimationAsset);
            Receipt->SetStringField(TEXT("capture_file"), State.CaptureFile);
            Receipt->SetStringField(TEXT("preview_video"), State.PreviewVideo);
            Receipt->SetStringField(TEXT("worker_child_process_log"), State.OutputDirectory / TEXT("child_processes.jsonl"));
            Receipt->SetNumberField(TEXT("frames_total"), State.FramesTotal);
            Receipt->SetNumberField(TEXT("actual_fps"), State.Fps);
            Receipt->SetStringField(TEXT("coordinate_frame"), State.CoordinateFrame);
            Receipt->SetBoolField(TEXT("local_raw_animation_button_disabled_checked"), State.IsPerHandLocal());
        }
        FString Text;
        return FJsonSerializer::Serialize(Receipt, TJsonWriterFactory<>::Create(&Text))
            && FFileHelper::SaveStringToFile(Text, *(Directory / TEXT("receipt.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
    }

    FAutomationTestBase* Test;
    ECase Case;
    FString Video, Directory, PreviousJob, JobId, PreviousStage, VisibleText, PendingAction, Failure;
    double Started, ActionStarted = 0, WorkerStarted = 0, LastEnumeration = 0, FailureAt = 0;
    int32 Step = 0;
    int32 ExpectedFrames = 97;
    double ExpectedFps = 30.;
    uint32 WorkerId = 0;
    bool bWaitingBool = false, bWaitingText = false, bOwnsDriver = false;
    bool bActualInputVerified = false, bEmergencyCleanup = false;
    TUniquePtr<FEditorHeartbeat> Heartbeat;
    TSharedPtr<FCardCapPythonBridge> Bridge;
    TSharedPtr<SCardCapPanel> Panel;
    TSharedPtr<IAsyncAutomationDriver, ESPMode::ThreadSafe> Driver;
    TSharedPtr<IAsyncDriverSequence, ESPMode::ThreadSafe> ActiveSequence;
    TAsyncResult<bool> BoolResult;
    TAsyncResult<FText> TextResult;
    TMap<uint32, FProcHandle> ObservedProcesses;
    TArray<TSharedPtr<FJsonValue>> Actions, Screenshots, Stages;
};

bool Enqueue(FAutomationTestBase* Test, ECase Case, const FString& Name)
{
    FString Evidence, Video, ExpectedFramesText, ExpectedFpsText;
    int32 ExpectedFrames = 97;
    double ExpectedFps = 30.;
    if (!FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestEvidence="), Evidence))
    {
        Test->AddInfo(TEXT("NOT EXECUTED: actual Slate test requires -CardCapUITestEvidence=<fresh directory>. No input or worker was simulated."));
        return true;
    }
    if (Case != ECase::InvalidPath && (!FParse::Param(FCommandLine::Get(), TEXT("CardCapUITestAllowRealRun"))
        || !FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestVideo="), Video)))
    {
        Test->AddInfo(TEXT("NOT EXECUTED: real video/cancel case requires -CardCapUITestAllowRealRun and -CardCapUITestVideo=<actual source>. Fixture defaults are 97 frames/30 fps; override with CardCapUITestExpectedFrames/CardCapUITestExpectedFps."));
        return true;
    }
    if (FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestExpectedFrames="), ExpectedFramesText))
    {
        double Value = 0;
        if (!FDefaultValueHelper::ParseDouble(ExpectedFramesText, Value) || !FMath::IsFinite(Value)
            || Value <= 0 || Value > MAX_int32 || Value != static_cast<double>(static_cast<int32>(Value)))
        { Test->AddError(TEXT("CardCapUITestExpectedFrames must be a positive int32 frame count.")); return false; }
        ExpectedFrames = static_cast<int32>(Value);
    }
    if (FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestExpectedFps="), ExpectedFpsText)
        && (!FDefaultValueHelper::ParseDouble(ExpectedFpsText, ExpectedFps) || !FMath::IsFinite(ExpectedFps) || ExpectedFps <= 0))
    { Test->AddError(TEXT("CardCapUITestExpectedFps must be finite and positive.")); return false; }
    if (IsRunningCommandlet() || FParse::Param(FCommandLine::Get(), TEXT("nullrhi")) || !FSlateApplication::IsInitialized())
    { Test->AddError(TEXT("Opted-in panel integration requires a fresh interactive editor with real Slate/RHI; commandlet/NullRHI is not an actual UI test.")); return false; }
    const FString Directory = FPaths::ConvertRelativePathToFull(Evidence) / Name;
    if (IFileManager::Get().DirectoryExists(*Directory) || !IFileManager::Get().MakeDirectory(*Directory, true))
    { Test->AddError(TEXT("Refusing to overwrite existing integration case evidence: ") + Directory); return false; }
    if (Case == ECase::InvalidPath) { Video = Directory / TEXT("nonexistent_source.mp4"); }
    else if (!FPaths::FileExists(Video)) { Test->AddError(TEXT("Explicit real source does not exist.")); return false; }
    Test->AddCommand(new FPanelCase(Test, Case, Video, Directory, ExpectedFrames, ExpectedFps));
    return true;
}

// Separate opt-in case: opens an already successful job, without reconstruction.
class FOpenLastResultCase final : public IAutomationLatentCommand
{
public:
    FOpenLastResultCase(FAutomationTestBase* InTest, FString InJob, FString InDirectory)
        : Test(InTest), ExpectedJob(MoveTemp(InJob)), Directory(MoveTemp(InDirectory)), Started(FPlatformTime::Seconds()) {}
    virtual ~FOpenLastResultCase() override
    {
        Sequence.Reset();
        Driver.Reset();
        if (bOwnsDriver) { IAutomationDriverModule::Get().Disable(); }
    }
    virtual bool Update() override
    {
        if (FPlatformTime::Seconds() - Started > 120.)
        { return Finish(TEXT("Opening the result did not satisfy map, Sequencer frame zero, live pose and actual level-viewport hand visibility within 120 seconds.")); }
        if (Step == 0)
        {
            if (!GEditor || IAutomationDriverModule::Get().IsEnabled())
            { return Finish(TEXT("Editor unavailable or another driver owns input.")); }
            FCardCapEditorModule& Module = FModuleManager::LoadModuleChecked<FCardCapEditorModule>(TEXT("CardistryCaptureEditor"));
            Bridge = Module.GetBridge();
            if (!Bridge || Bridge->GetProcessId() != 0 || Bridge->GetSnapshot().Status != TEXT("succeeded")
                || Bridge->GetSnapshot().JobId != ExpectedJob)
            { return Finish(TEXT("The restored successful result does not match the explicitly expected job; no scene was opened.")); }
            TArray<UPackage*> Dirty;
            FEditorFileUtils::GetDirtyWorldPackages(Dirty);
            FEditorFileUtils::GetDirtyContentPackages(Dirty);
            UWorld* Before = GEditor->GetEditorWorldContext().World();
            InitialMap = Before ? Before->GetOutermost()->GetName() : FString();
            if (!Dirty.IsEmpty() || (Before && Before->GetOutermost()->IsDirty()))
            { return Finish(TEXT("Run this test in a fresh editor opened on a clean saved map. Unsaved work was not changed or dismissed.")); }
            FString Error;
            if (!FCardCapJsonParser::ParseFile(Bridge->GetSnapshot().CaptureFile, Capture, Error)) { return Finish(Error); }
            Module.OpenPanel();
            Panel = Module.GetPanel();
            if (!Panel) { return Finish(TEXT("Actual panel was not created.")); }
            IAutomationDriverModule::Get().Enable();
            bOwnsDriver = true;
            Driver = IAutomationDriverModule::Get().CreateAsyncDriver();
            Step = 1;
            return false;
        }
        if (Step == 1)
        {
            if (Panel->GetCachedGeometry().GetAbsoluteSize().X < 100.) { return false; }
            if (!SaveScreenshot(Panel.ToSharedRef(), TEXT("panel_before_open.png"))) { return Finish(TEXT("Actual before-open screenshot failed.")); }
            Sequence = Driver->CreateSequence();
            Sequence->Actions().Click(By::Path(FString(TEXT("CardCap.OpenResult"))));
            Click = Sequence->Perform();
            Step = 2;
            return false;
        }
        if (Step == 2)
        {
            if (!Click.GetFuture().IsReady()) { return false; }
            if (!Click.GetFuture().Get()) { return Finish(TEXT("Actual OpenResult button click failed.")); }
            bClicked = true;
            Sequence.Reset();
            Step = 3;
            return false;
        }
        const FCardCapJobSnapshot& State = Bridge->GetSnapshot();
        if (Bridge->GetProcessId() != 0 || State.JobId != ExpectedJob)
        { return Finish(TEXT("Opening a result unexpectedly launched or changed the job.")); }
        if (!State.Error.IsEmpty()) { return Finish(TEXT("Open-result UI reported: ") + State.Error); }
        UWorld* World = GEditor->GetEditorWorldContext().World();
        ActualMap = World ? World->GetOutermost()->GetName() : FString();
        if (ActualMap != FPackageName::ObjectPathToPackageName(State.MapAsset)) { return false; }
        ULevelSequence* Asset = FindObject<ULevelSequence>(nullptr, *State.SequenceAsset);
        if (!Asset) { return false; }
        IAssetEditorInstance* Instance = GEditor->GetEditorSubsystem<UAssetEditorSubsystem>()->FindEditorForAsset(Asset, false);
        if (!Instance || Instance->GetEditorName() != FName(TEXT("LevelSequenceEditor"))) { return false; }
        const TSharedPtr<ISequencer> Sequencer = static_cast<ILevelSequenceEditorToolkit*>(Instance)->GetSequencer();
        if (!Sequencer || Sequencer->GetRootMovieSceneSequence() != Asset) { return false; }
        SequencerTime = Sequencer->GetGlobalTime().Time.AsDecimal();
        if (!FMath::IsNearlyZero(SequencerTime, 1.e-8)) { return false; }
        // Read the actual editor-world components. Never call SetGlobalTime,
        // RefreshBoneTransforms or change an actor to make this check pass.
        int32 ActorCount = 0;
        ACardCapResearchHandsActor* Hands = nullptr;
        for (TActorIterator<ACardCapResearchHandsActor> It(World); It; ++It) { Hands = *It; ++ActorCount; }
        if (ActorCount != 1) { return false; }
        USkeletalMeshComponent* Mesh = Hands->GetSkeletalMeshComponent();
        bActorIdentity = Hands->GetActorTransform().Equals(FTransform::Identity);
        bComponentIdentity = Mesh->GetComponentTransform().Equals(FTransform::Identity);
        if (!bActorIdentity || !bComponentIdentity) { return Finish(TEXT("Result hands have a non-identity transform.")); }
        if (Capture.FormatVersion == TEXT("1.3") && !Capture.bInterHandTransformKnown)
        {
            const auto* LocalMesh = Cast<UCardCapResearchSkeletalMeshComponent>(Mesh);
            if (!LocalMesh || !LocalMesh->bSeparateLocalHands || LocalMesh->bShowRightLocalHand
                || LocalMesh->LocalLeftMaterialIds.IsEmpty() || LocalMesh->LocalRightMaterialIds.IsEmpty())
                return Finish(TEXT("Local scene did not restore its one-side display selection."));
            for (int32 LOD = 0; LOD < Mesh->GetNumLODs(); ++LOD)
                for (int32 Material = 0; Material < Mesh->GetNumMaterials(); ++Material)
                    if (Mesh->IsMaterialSectionShown(Material, LOD) != LocalMesh->LocalLeftMaterialIds.Contains(Material))
                        return Finish(TEXT("Reloaded local scene reveals both hand origins instead of one isolated side."));
            for (TActorIterator<ACameraActor> It(World); It; ++It)
                if (It->GetActorLabel().StartsWith(TEXT("Capture camera")))
                    return Finish(TEXT("Unknown source camera was incorrectly recreated as a Capture camera."));
        }
        JointChecks.Empty();
        MaxJointErrorCm = 0;
        for (const FCardCapHand& Hand : Capture.Hands)
        {
            const FCardCapHandBoneMapping* Mapping = Capture.BoneMapping.Hands.Find(Hand.Side);
            if (!Mapping || Hand.Frames.IsEmpty() || Hand.Frames[0].Frame != 0) { return Finish(TEXT("Capture lacks configured first-frame hands.")); }
            for (int32 Index = 0; Index < Mapping->BoneNames.Num(); ++Index)
            {
                const FName Bone = Mapping->BoneNames[Index];
                if (Mesh->GetBoneIndex(Bone) == INDEX_NONE) { return Finish(TEXT("Live result is missing a configured hand bone.")); }
                const FVector Actual = Mesh->GetBoneLocation(Bone, EBoneSpaces::WorldSpace);
                const FVector Expected = Hand.Frames[0].JointPositionsCm[Capture.BoneMapping.LandmarkIndices[Index]];
                const double Error = FVector::Dist(Actual, Expected);
                if (!FMath::IsFinite(Error)) { return Finish(TEXT("Live result contains a non-finite joint location.")); }
                MaxJointErrorCm = FMath::Max(MaxJointErrorCm, Error);
                const TSharedRef<FJsonObject> Joint = MakeShared<FJsonObject>();
                Joint->SetStringField(TEXT("bone"), Bone.ToString());
                Joint->SetNumberField(TEXT("distance_to_capture_frame_zero_cm"), Error);
                JointChecks.Add(MakeShared<FJsonValueObject>(Joint));
            }
        }
        if (JointChecks.Num() != 32 || MaxJointErrorCm > .01) { return false; }
        if (!ReadActualViewport(World)) { return false; }
        const TSharedPtr<SWindow> MainWindow = FGlobalTabmanager::Get()->GetRootWindow();
        const TSharedPtr<SWindow> ActiveWindow = FSlateApplication::Get().GetActiveTopLevelWindow();
        if (!MainWindow || !ActiveWindow
            || !SaveScreenshot(MainWindow.ToSharedRef(), TEXT("editor_result_frame_zero.png"))
            || !SaveScreenshot(ActiveWindow.ToSharedRef(), TEXT("active_result_window.png")))
        { return Finish(TEXT("Actual opened editor/Sequencer screenshot failed.")); }
        return Finish(FString());
    }
private:
    bool ReadActualViewport(UWorld* World)
    {
        const double Now = FPlatformTime::Seconds();
        if (Now < NextViewportRead) return false;
        NextViewportRead = Now + .25;
        const bool bLocal = Capture.FormatVersion == TEXT("1.3") && !Capture.bInterHandTransformKnown;
        const FString CameraPrefix = bLocal ? TEXT("Left local viewer") : TEXT("Capture camera");
        FLevelEditorViewportClient* Client = nullptr;
        ViewportCandidates.Empty();
        for (FLevelEditorViewportClient* Candidate : GEditor->GetLevelViewportClients())
        {
            if (!Candidate) continue;
            const AActor* Locked = Candidate->GetCinematicActorLock().GetLockedActor();
            const AActor* UserLocked = Candidate->GetActorLock().GetLockedActor();
            const UCameraComponent* ViewCamera = Candidate->GetCameraComponentForView();
            const FIntPoint Size = Candidate->Viewport ? Candidate->Viewport->GetSizeXY() : FIntPoint::ZeroValue;
            const TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
            Entry->SetBoolField(TEXT("same_world"), Candidate->GetWorld() == World);
            Entry->SetBoolField(TEXT("visible"), Candidate->IsVisible());
            Entry->SetBoolField(TEXT("perspective"), Candidate->IsPerspective());
            Entry->SetBoolField(TEXT("allows_cinematic_control"), Candidate->AllowsCinematicControl());
            Entry->SetBoolField(TEXT("locked_camera_view"), Candidate->bLockedCameraView);
            Entry->SetStringField(TEXT("cinematic_lock"), Locked ? Locked->GetActorLabel() : FString());
            Entry->SetStringField(TEXT("user_lock"), UserLocked ? UserLocked->GetActorLabel() : FString());
            Entry->SetStringField(TEXT("actual_view_camera"), ViewCamera ? ViewCamera->GetOwner()->GetActorLabel() : FString());
            Entry->SetArrayField(TEXT("resolution"), {MakeShared<FJsonValueNumber>(Size.X), MakeShared<FJsonValueNumber>(Size.Y)});
            ViewportCandidates.Add(MakeShared<FJsonValueObject>(Entry));
            if (!Candidate->Viewport || !Candidate->IsVisible() || !Candidate->IsPerspective()
                || Candidate->GetWorld() != World) continue;
            if (Locked && Locked->GetActorLabel().StartsWith(CameraPrefix)
                && ViewCamera && ViewCamera->GetOwner() == Locked) Client = Candidate;
        }
        if (!Client) { ViewportFailure = TEXT("No visible level viewport follows the result's display camera."); return false; }
        ViewportCameraLabel = Client->GetCinematicActorLock().GetLockedActor()->GetActorLabel();
        LastViewportSize = Client->Viewport->GetSizeXY();
        if (LastViewportSize.X <= 0 || LastViewportSize.Y <= 0) return false;
        LastViewportPixels.Empty();
        if (!Client->Viewport->ReadPixels(LastViewportPixels)
            || LastViewportPixels.Num() != LastViewportSize.X * LastViewportSize.Y)
        { ViewportFailure = TEXT("Actual level viewport ReadPixels failed."); return false; }
        bViewportRead = true;
        // Derive a region from the expected live pose and the actual viewport
        // projection. Text, toolbar icons or a distant grid cannot make a black
        // hand region pass. This test never steers a camera, forces a draw,
        // seeks a sequence or refreshes/mutates the hand pose.
        FSceneViewFamilyContext ViewFamily(FSceneViewFamily::ConstructionValues(
            Client->Viewport, Client->GetScene(), Client->EngineShowFlags));
        const FSceneView* View = Client->CalcSceneView(&ViewFamily);
        if (!View) return false;
        FBox2D ProjectedBounds(ForceInit);
        for (const FCardCapHand& Hand : Capture.Hands)
        {
            if (bLocal && Hand.Side != TEXT("left")) continue;
            for (const FVector& Point : Hand.Frames[0].JointPositionsCm)
            {
                FVector2D Pixel;
                if (!FSceneView::ProjectWorldToScreen(Point, View->UnscaledViewRect,
                    View->ViewMatrices.GetViewProjectionMatrix(), Pixel))
                { ViewportFailure = TEXT("Result hand projects behind the actual level viewport camera."); return false; }
                ProjectedBounds += Pixel;
            }
        }
        if (!ProjectedBounds.bIsValid) return false;
        const int32 MinX = FMath::Clamp(FMath::FloorToInt(ProjectedBounds.Min.X), 0, LastViewportSize.X - 1);
        const int32 MinY = FMath::Clamp(FMath::FloorToInt(ProjectedBounds.Min.Y), 0, LastViewportSize.Y - 1);
        const int32 MaxX = FMath::Clamp(FMath::CeilToInt(ProjectedBounds.Max.X), 0, LastViewportSize.X - 1);
        const int32 MaxY = FMath::Clamp(FMath::CeilToInt(ProjectedBounds.Max.Y), 0, LastViewportSize.Y - 1);
        ViewportPoseRegion = FIntRect(MinX, MinY, MaxX + 1, MaxY + 1);
        ViewportHandPixels = 0;
        int32 SupportMinX = LastViewportSize.X, SupportMinY = LastViewportSize.Y, SupportMaxX = -1, SupportMaxY = -1;
        for (int32 Y = MinY; Y <= MaxY; ++Y)
            for (int32 X = MinX; X <= MaxX; ++X)
            {
                const FColor& Pixel = LastViewportPixels[Y * LastViewportSize.X + X];
                if (Pixel.R <= 32 && Pixel.G <= 32 && Pixel.B <= 32) continue;
                ++ViewportHandPixels;
                SupportMinX = FMath::Min(SupportMinX, X); SupportMinY = FMath::Min(SupportMinY, Y);
                SupportMaxX = FMath::Max(SupportMaxX, X); SupportMaxY = FMath::Max(SupportMaxY, Y);
            }
        ViewportHandLongEdge = ViewportHandPixels ? FMath::Max(SupportMaxX - SupportMinX + 1, SupportMaxY - SupportMinY + 1) : 0;
        bViewportVisible = ViewportHandPixels >= 64 && ViewportHandLongEdge >= 24;
        ViewportFailure = bViewportVisible ? FString() : TEXT("Expected hand region in the actual level viewport is black or too small.");
        return bViewportVisible;
    }

    bool SaveScreenshot(const TSharedRef<SWidget>& Widget, const FString& Name)
    {
        TArray<FColor> Pixels;
        FIntVector Size(0, 0, 0);
        if (!FSlateApplication::Get().TakeScreenshot(Widget, Pixels, Size)
            || Size.X <= 0 || Size.Y <= 0 || Pixels.Num() != Size.X * Size.Y) { return false; }
        TArray64<uint8> Png;
        FImageUtils::PNGCompressImageArray(Size.X, Size.Y, TArrayView64<const FColor>(Pixels.GetData(), Pixels.Num()), Png);
        const FString Path = Directory / Name;
        if (!FFileHelper::SaveArrayToFile(Png, *Path)) { return false; }
        Images.Add(MakeShared<FJsonValueString>(Path));
        return true;
    }
    bool Finish(const FString& Error)
    {
        if (!Error.IsEmpty()) { Test->AddError(Error); }
        if (!Error.IsEmpty() && FSlateApplication::IsInitialized())
        {
            const TSharedPtr<SWindow> MainWindow = FGlobalTabmanager::Get()->GetRootWindow();
            const TSharedPtr<SWindow> ActiveWindow = FSlateApplication::Get().GetActiveTopLevelWindow();
            if (MainWindow) SaveScreenshot(MainWindow.ToSharedRef(), TEXT("editor_result_failed.png"));
            if (ActiveWindow) SaveScreenshot(ActiveWindow.ToSharedRef(), TEXT("active_result_failed.png"));
        }
        if (LastViewportSize.X > 0 && LastViewportSize.Y > 0 && LastViewportPixels.Num() == LastViewportSize.X * LastViewportSize.Y)
        {
            TArray64<uint8> Png;
            FImageUtils::PNGCompressImageArray(LastViewportSize.X, LastViewportSize.Y,
                TArrayView64<const FColor>(LastViewportPixels.GetData(), LastViewportPixels.Num()), Png);
            const FString Path = Directory / TEXT("actual_level_viewport.png");
            if (FFileHelper::SaveArrayToFile(Png, *Path)) Images.Add(MakeShared<FJsonValueString>(Path));
            else Test->AddError(TEXT("Actual viewport evidence PNG could not be saved."));
        }
        const TSharedRef<FJsonObject> Receipt = MakeShared<FJsonObject>();
        Receipt->SetStringField(TEXT("status"), Error.IsEmpty() ? TEXT("passed") : TEXT("failed"));
        Receipt->SetStringField(TEXT("case"), TEXT("PanelOpenLastResult"));
        Receipt->SetStringField(TEXT("expected_job_id"), ExpectedJob);
        Receipt->SetStringField(TEXT("failure"), Error);
        Receipt->SetBoolField(TEXT("actual_open_result_click_completed"), bClicked);
        Receipt->SetBoolField(TEXT("test_sets_sequence_time_or_hand_pose"), false);
        Receipt->SetBoolField(TEXT("test_sets_camera_or_forces_viewport_draw"), false);
        Receipt->SetBoolField(TEXT("actual_level_viewport_readpixels"), bViewportRead);
        Receipt->SetBoolField(TEXT("actual_level_viewport_hand_visible"), bViewportVisible);
        Receipt->SetStringField(TEXT("actual_level_viewport_camera_label"), ViewportCameraLabel);
        Receipt->SetStringField(TEXT("actual_level_viewport_visibility_failure"), ViewportFailure);
        Receipt->SetArrayField(TEXT("level_viewport_candidates"), ViewportCandidates);
        Receipt->SetNumberField(TEXT("actual_level_viewport_hand_pixels_above_32"), ViewportHandPixels);
        Receipt->SetNumberField(TEXT("actual_level_viewport_hand_long_edge_px"), ViewportHandLongEdge);
        Receipt->SetArrayField(TEXT("actual_level_viewport_resolution"), {MakeShared<FJsonValueNumber>(LastViewportSize.X), MakeShared<FJsonValueNumber>(LastViewportSize.Y)});
        Receipt->SetArrayField(TEXT("actual_level_viewport_projected_pose_region_xyxy"), {
            MakeShared<FJsonValueNumber>(ViewportPoseRegion.Min.X), MakeShared<FJsonValueNumber>(ViewportPoseRegion.Min.Y),
            MakeShared<FJsonValueNumber>(ViewportPoseRegion.Max.X), MakeShared<FJsonValueNumber>(ViewportPoseRegion.Max.Y)});
        Receipt->SetStringField(TEXT("actual_level_viewport_visibility_criterion"), TEXT("Display-only: >=64 pixels with max RGB>32 and >=24 px support long edge within actual projected hand-joint bounds; no pose or metric accuracy claim"));
        Receipt->SetBoolField(TEXT("local_hand_visibility_checked"), Capture.FormatVersion == TEXT("1.3") && !Capture.bInterHandTransformKnown);
        if (Capture.FormatVersion == TEXT("1.3")) Receipt->SetStringField(TEXT("coordinate_frame"), Capture.Provenance.CoordinateFrame);
        Receipt->SetStringField(TEXT("initial_map"), InitialMap);
        Receipt->SetStringField(TEXT("actual_opened_map"), ActualMap);
        Receipt->SetNumberField(TEXT("sequencer_global_time_ticks"), SequencerTime);
        Receipt->SetNumberField(TEXT("actual_joint_checks"), JointChecks.Num());
        Receipt->SetNumberField(TEXT("max_joint_error_to_capture_frame_zero_cm"), MaxJointErrorCm);
        Receipt->SetBoolField(TEXT("actor_identity"), bActorIdentity);
        Receipt->SetBoolField(TEXT("component_identity"), bComponentIdentity);
        Receipt->SetArrayField(TEXT("joint_checks"), JointChecks);
        Receipt->SetArrayField(TEXT("actual_slate_screenshots"), Images);
        Receipt->SetNumberField(TEXT("elapsed_seconds"), FPlatformTime::Seconds() - Started);
        if (Bridge)
        {
            Receipt->SetStringField(TEXT("sequence_asset"), Bridge->GetSnapshot().SequenceAsset);
            Receipt->SetStringField(TEXT("capture_file"), Bridge->GetSnapshot().CaptureFile);
        }
        FString Text;
        if (!FJsonSerializer::Serialize(Receipt, TJsonWriterFactory<>::Create(&Text))
            || !FFileHelper::SaveStringToFile(Text, *(Directory / TEXT("receipt.json")), FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
        { Test->AddError(TEXT("Open-result receipt could not be saved.")); }
        return true;
    }
    FAutomationTestBase* Test;
    FString ExpectedJob, Directory, InitialMap, ActualMap;
    double Started, SequencerTime = -1, MaxJointErrorCm = -1;
    double NextViewportRead = 0;
    int32 Step = 0;
    bool bOwnsDriver = false, bClicked = false, bActorIdentity = false, bComponentIdentity = false;
    bool bViewportRead = false, bViewportVisible = false;
    int32 ViewportHandPixels = 0, ViewportHandLongEdge = 0;
    FIntPoint LastViewportSize = FIntPoint::ZeroValue;
    FIntRect ViewportPoseRegion = FIntRect(0, 0, 0, 0);
    FString ViewportCameraLabel, ViewportFailure;
    TArray<FColor> LastViewportPixels;
    FCardCapCapture Capture;
    TSharedPtr<FCardCapPythonBridge> Bridge;
    TSharedPtr<SCardCapPanel> Panel;
    TSharedPtr<IAsyncAutomationDriver, ESPMode::ThreadSafe> Driver;
    TSharedPtr<IAsyncDriverSequence, ESPMode::ThreadSafe> Sequence;
    TAsyncResult<bool> Click;
    TArray<TSharedPtr<FJsonValue>> JointChecks, Images, ViewportCandidates;
};
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapPanelRealVideoTest, "CardistryCapture.Integration.PanelRealVideo",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapPanelRealVideoTest::RunTest(const FString&)
{ return CardCapPanelIntegration::Enqueue(this, CardCapPanelIntegration::ECase::RealVideo, TEXT("PanelRealVideo")); }

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapPanelCancelTest, "CardistryCapture.Integration.PanelCancel",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapPanelCancelTest::RunTest(const FString&)
{ return CardCapPanelIntegration::Enqueue(this, CardCapPanelIntegration::ECase::Cancel, TEXT("PanelCancel")); }

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapPanelInvalidPathTest, "CardistryCapture.Integration.PanelInvalidPath",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapPanelInvalidPathTest::RunTest(const FString&)
{ return CardCapPanelIntegration::Enqueue(this, CardCapPanelIntegration::ECase::InvalidPath, TEXT("PanelInvalidPath")); }

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapPanelOpenLastResultTest, "CardistryCapture.Integration.PanelOpenLastResult",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapPanelOpenLastResultTest::RunTest(const FString&)
{
    FString Evidence, Job;
    if (!FParse::Param(FCommandLine::Get(), TEXT("CardCapUITestOpenLastResult"))
        || !FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestEvidence="), Evidence)
        || !FParse::Value(FCommandLine::Get(), TEXT("CardCapUITestExpectedJob="), Job))
    { AddInfo(TEXT("NOT EXECUTED: use -CardCapUITestOpenLastResult -CardCapUITestExpectedJob=<successful job id> -CardCapUITestEvidence=<fresh directory>. This case never starts inference.")); return true; }
    if (IsRunningCommandlet() || FParse::Param(FCommandLine::Get(), TEXT("nullrhi")) || !FSlateApplication::IsInitialized())
    { AddError(TEXT("Open-result test requires a fresh interactive editor with real RHI.")); return false; }
    const FString Directory = FPaths::ConvertRelativePathToFull(Evidence) / TEXT("PanelOpenLastResult");
    if (IFileManager::Get().DirectoryExists(*Directory) || !IFileManager::Get().MakeDirectory(*Directory, true))
    { AddError(TEXT("Refusing to overwrite open-result evidence.")); return false; }
    AddCommand(new CardCapPanelIntegration::FOpenLastResultCase(this, Job, Directory));
    return true;
}
#endif
