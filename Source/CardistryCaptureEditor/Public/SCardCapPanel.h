#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"

class FCardCapPythonBridge;
struct FCardCapJobSnapshot;

/** A view of the module-owned job. Destroying this widget never cancels it. */
class CARDISTRYCAPTUREEDITOR_API SCardCapPanel final : public SCompoundWidget
{
public:
    SLATE_BEGIN_ARGS(SCardCapPanel) {}
        SLATE_ARGUMENT(TSharedPtr<FCardCapPythonBridge>, Bridge)
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs);

    /** Real input/start entry points; they share the same path as the UI buttons. */
    void SetVideoPath(const FString& InPath);
    bool StartProcessing(FString& OutError);

    /** Presentation shared by live jobs and results restored from earlier versions. */
    static FText JobMessageText(const FCardCapJobSnapshot& State);

private:
    const FCardCapJobSnapshot& Snapshot() const;
    bool IsRunning() const;
    bool CanEdit() const;
    FReply BrowseVideo();
    FReply BrowseBoneMapping();
    FReply OnStartClicked();
    FReply OnCancelClicked();
    bool ChooseFile(const FText& Title, const FString& Filter, const FString& Current, FString& OutPath);
    FText StatusText() const;
    FText MessageText() const;
    FText ProgressText() const;
    FText FrameTimeText() const;
    FText ConfidenceRangesText() const;
    FText SourceNoteText() const;

    TSharedPtr<FCardCapPythonBridge> Bridge;
    FString VideoPath;
    FString BoneMappingPath;
    FString LocalError;
    bool bAllowBlurry = false;
    bool bRenderPreview = true;
};
