#pragma once

#include "CoreMinimal.h"
#include "HAL/PlatformProcess.h"
#include "TickableEditorObject.h"

struct FCardCapJobOptions
{
    FString VideoPath;
    FString BoneMappingPath;
    bool bAllowBlurry = false;
    bool bRenderPreview = true;
};

struct FCardCapJobSnapshot
{
    FString JobId, Status = TEXT("idle"), Stage, Message, Error, LogPath;
    FString OutputDirectory, PreviewVideo, CaptureFile, MapAsset, SequenceAsset, AnimationAsset;
    FString ScaleConfidence, IntrinsicsSource, CoordinateFrame;
    // Retain old diagnostics for logs while presenting restored jobs in English.
    FString HistoricalError;
    bool bMessagesAreEnglish = false;
    double Progress = 0, ElapsedSeconds = 0, Fps = 0;
    int32 FramesCompleted = 0, FramesTotal = 0;
    TArray<FIntPoint> LowConfidenceRanges;
    bool IsRunning() const { return Status == TEXT("running") || Status == TEXT("cancelling"); }
    bool IsPerHandLocal() const { return CoordinateFrame == TEXT("per_hand_wrist_local"); }
};

// One owner per editor module. Closing the tab does not destroy the job.
// Heavy work lives in separate processes; Tick only polls a bounded JSON file.
class CARDISTRYCAPTUREEDITOR_API FCardCapPythonBridge final
    : public FTickableEditorObject, public TSharedFromThis<FCardCapPythonBridge>
{
public:
    FCardCapPythonBridge();
    virtual ~FCardCapPythonBridge() override;
    bool Start(const FCardCapJobOptions& Options, FString& OutError);
    void Cancel();
    void Shutdown();
    const FCardCapJobSnapshot& GetSnapshot() const { return Snapshot; }
    bool CanStart() const { return !Process.IsValid() && !bShuttingDown; }
    uint32 GetProcessId() const { return ProcessId; }
    void OpenResultScene();
    void OpenResultAnimation();
    void OpenPreview();
    void OpenOutputFolder();
    void OpenLog();
    virtual void Tick(float DeltaTime) override;
    virtual bool IsTickable() const override;
    virtual TStatId GetStatId() const override;

    // Small deterministic boundaries are shared by the implementation/tests.
    static FString QuoteArgument(const FString& Value);
    static bool ParseStatus(const FString& Json, const FString& ExpectedJob,
        FCardCapJobSnapshot& InOut, FString& OutError);

private:
    bool ReadStatus();
    bool ValidateResults(FString& OutError) const;
    void Fail(const FString& Message);
    void SaveTerminalStatus();
    void RestoreLastJob();
    void RefreshResultAssets();
    void SeekOpenedSequence();
    FString JobsRoot() const;
    FCardCapJobSnapshot Snapshot;
    FProcHandle Process;
    uint32 ProcessId = 0;
    bool bCancelRequested = false, bShuttingDown = false;
    double StartedAt = 0, LastPoll = 0, CancelledAt = 0;
    FString PendingSequenceAsset;
    int32 PendingSequenceTicks = 0;
};
