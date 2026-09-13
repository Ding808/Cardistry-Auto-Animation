#include "SCardCapPanel.h"

#include "CardCapPythonBridge.h"
#include "Brushes/SlateColorBrush.h"
#include "DesktopPlatformModule.h"
#include "Framework/Application/SlateApplication.h"
#include "GenericPlatform/GenericWindow.h"
#include "IDesktopPlatform.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "Rendering/DrawElements.h"
#include "Rendering/SlateLayoutTransform.h"
#include "Styling/CoreStyle.h"
#include "Styling/WidgetStyle.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SExpandableArea.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SWrapBox.h"
#include "Widgets/Notifications/SProgressBar.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/SLeafWidget.h"
#include "Widgets/SWindow.h"
#include "Widgets/Text/STextBlock.h"

#define LOCTEXT_NAMESPACE "CardCapPanel"

namespace
{
class SCardCapConfidenceBar final : public SLeafWidget
{
public:
    SLATE_BEGIN_ARGS(SCardCapConfidenceBar) {}
        SLATE_ARGUMENT(TSharedPtr<FCardCapPythonBridge>, Bridge)
    SLATE_END_ARGS()

    void Construct(const FArguments& InArgs)
    {
        Bridge = InArgs._Bridge;
        // The job owns changing data, not this widget. Paint while visible;
        // there is no widget ticker, process poll, or lifetime callback here.
        ForceVolatile(true);
    }

private:
    virtual FVector2D ComputeDesiredSize(float) const override { return FVector2D(280., 22.); }

    virtual int32 OnPaint(const FPaintArgs&, const FGeometry& Geometry, const FSlateRect&,
        FSlateWindowElementList& Elements, int32 Layer, const FWidgetStyle& Style, bool) const override
    {
        const FVector2D Size = Geometry.GetLocalSize();
        const FLinearColor Tint = Style.GetColorAndOpacityTint();
        FSlateDrawElement::MakeBox(Elements, Layer, Geometry.ToPaintGeometry(), &WhiteBrush,
            ESlateDrawEffect::None, FLinearColor(.055f, .065f, .08f, 1.f) * Tint);
        const TSharedPtr<FCardCapPythonBridge> Job = Bridge.Pin();
        if (!Job.IsValid() || Job->GetSnapshot().FramesTotal <= 0)
        {
            return Layer;
        }
        const FCardCapJobSnapshot& State = Job->GetSnapshot();
        const int32 Total = State.FramesTotal;
        for (const FIntPoint& Range : State.LowConfidenceRanges)
        {
            if (Range.Y < Range.X || Range.Y < 0 || Range.X >= Total)
            {
                continue;
            }
            const double Left = static_cast<double>(FMath::Clamp(Range.X, 0, Total - 1)) / Total * Size.X;
            const double Right = static_cast<double>(FMath::Clamp(Range.Y, 0, Total - 1) + 1) / Total * Size.X;
            FSlateDrawElement::MakeBox(Elements, Layer + 1,
                Geometry.ToPaintGeometry(FVector2D(FMath::Max(1., Right - Left), Size.Y),
                    FSlateLayoutTransform(FVector2D(Left, 0.))), &WhiteBrush,
                ESlateDrawEffect::None, FLinearColor(.92f, .48f, .08f, 1.f) * Tint);
        }
        return Layer + 1;
    }

    TWeakPtr<FCardCapPythonBridge> Bridge;
    FSlateColorBrush WhiteBrush{FLinearColor::White};
};

FString CleanPath(const FString& Value)
{
    FString Result = Value.TrimStartAndEnd();
    if (Result.Len() >= 2 && Result.StartsWith(TEXT("\"")) && Result.EndsWith(TEXT("\"")))
    {
        Result = Result.Mid(1, Result.Len() - 2).TrimStartAndEnd();
    }
    return Result;
}
}

void SCardCapPanel::Construct(const FArguments& InArgs)
{
    Bridge = InArgs._Bridge;
    SetTag(TEXT("CardCap.Panel"));
    const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("CardistryCapture"));
    if (Plugin.IsValid())
    {
        BoneMappingPath = FPaths::ConvertRelativePathToFull(FPaths::Combine(Plugin->GetBaseDir(),
            TEXT("Config/BoneMapping_UE5Mannequin.json")));
    }

    ChildSlot
    [
        SNew(SScrollBox)
        + SScrollBox::Slot()
        .Padding(20.f)
        [
            SNew(SVerticalBox)
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 7)
            [ SNew(STextBlock).Text(LOCTEXT("Title", "Cardistry Capture"))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 22)) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [ SNew(STextBlock).Text(LOCTEXT("Introduction", "Choose a video to generate hand motion for review in the editor."))
                .AutoWrapText(true).ColorAndOpacity(FLinearColor(.7f, .73f, .78f)) ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 6)
            [ SNew(STextBlock).Text(LOCTEXT("VideoLabel", "Source Video")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 14)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().FillWidth(1.f)
                [ SNew(SEditableTextBox).Tag(TEXT("CardCap.VideoPath"))
                    .HintText(LOCTEXT("VideoHint", "Choose a video or paste its full path"))
                    .Text_Lambda([this] { return FText::FromString(VideoPath); })
                    .IsReadOnly_Lambda([this] { return !CanEdit(); })
                    .OnTextChanged_Lambda([this](const FText& Text) { if (CanEdit()) { VideoPath = Text.ToString(); LocalError.Reset(); } }) ]
                + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                [ SNew(SButton).Tag(TEXT("CardCap.BrowseVideo"))
                    .Text(LOCTEXT("BrowseVideo", "Choose Video..."))
                    .IsEnabled_Lambda([this] { return CanEdit(); })
                    .OnClicked(this, &SCardCapPanel::BrowseVideo) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().AutoWidth()
                [ SNew(SButton).Tag(TEXT("CardCap.Start"))
                    .Text(LOCTEXT("Start", "Start Processing"))
                    .ContentPadding(FMargin(24, 8))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Bridge->CanStart() && !CleanPath(VideoPath).IsEmpty(); })
                    .OnClicked(this, &SCardCapPanel::OnStartClicked) ]
                + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                [ SNew(SButton).Tag(TEXT("CardCap.Cancel"))
                    .Text(LOCTEXT("Cancel", "Cancel Processing"))
                    .ContentPadding(FMargin(18, 8))
                    .IsEnabled_Lambda([this] { return IsRunning() && Snapshot().Status != TEXT("cancelling"); })
                    .OnClicked(this, &SCardCapPanel::OnCancelClicked) ]
            ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [ SNew(STextBlock).Tag(TEXT("CardCap.Status"))
                .Text(this, &SCardCapPanel::StatusText).Font(FCoreStyle::GetDefaultFontStyle("Bold", 14)) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 6)
            [
                SNew(SBox).HeightOverride(12.f)
                [ SNew(SProgressBar)
                    .Percent_Lambda([this] { return TOptional<float>(static_cast<float>(FMath::Clamp(Snapshot().Progress, 0., 1.))); })
                    .FillColorAndOpacity(FLinearColor(.12f, .56f, .78f)) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 5)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().FillWidth(1.f)
                [ SNew(STextBlock).Text(this, &SCardCapPanel::FrameTimeText).AutoWrapText(true) ]
                + SHorizontalBox::Slot().AutoWidth().Padding(10, 0, 0, 0)
                [ SNew(STextBlock).Text(this, &SCardCapPanel::ProgressText) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 12)
            [ SNew(STextBlock).Tag(TEXT("CardCap.Message"))
                .Text(this, &SCardCapPanel::MessageText).AutoWrapText(true)
                .ColorAndOpacity_Lambda([this] { return FSlateColor((!LocalError.IsEmpty() || !Snapshot().Error.IsEmpty())
                    ? FLinearColor(1.f, .42f, .32f) : FLinearColor(.7f, .73f, .78f)); }) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 16)
            [ SNew(STextBlock).Text(LOCTEXT("CloseNote", "Closing this panel will not interrupt processing."))
                .ColorAndOpacity(FLinearColor(.55f, .59f, .65f)) ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 14)
            [ SNew(SSeparator) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [ SNew(STextBlock).Text(LOCTEXT("Results", "Results"))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 14)) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 12)
            [
                SNew(SWrapBox).UseAllottedSize(true).InnerSlotPadding(FVector2D(8, 8))
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenResult")).Text(LOCTEXT("OpenScene", "Open Scene"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().MapAsset.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenResultScene(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenAnimation")).Text(LOCTEXT("OpenAnimation", "Open Animation"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().AnimationAsset.IsEmpty() && !Snapshot().IsPerHandLocal(); })
                    .ToolTipText_Lambda([this] { return Snapshot().IsPerHandLocal()
                        ? LOCTEXT("LocalAnimationUnavailable", "The relative hand positions are unknown. Review the separate left and right hand previews.") : FText::GetEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenResultAnimation(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.Preview")).Text(LOCTEXT("OpenPreview", "Preview Video"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().PreviewVideo.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenPreview(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenFolder")).Text(LOCTEXT("OpenFolder", "Result Folder"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && !Snapshot().OutputDirectory.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenOutputFolder(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenLog")).Text(LOCTEXT("OpenLog", "View Log"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && !Snapshot().LogPath.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenLog(); } return FReply::Handled(); }) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 7)
            [ SNew(STextBlock).Text(LOCTEXT("ReviewRanges", "Review the yellow sections")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 7)
            [ SNew(SCardCapConfidenceBar).Tag(TEXT("CardCap.ConfidenceBar")).Bridge(Bridge)
                .ToolTipText(LOCTEXT("ReviewTooltip", "Yellow marks sections flagged as low confidence. Other sections have not necessarily been manually verified.")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [ SNew(STextBlock).Text(this, &SCardCapPanel::ConfidenceRangesText).AutoWrapText(true) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [ SNew(STextBlock).Text(this, &SCardCapPanel::SourceNoteText).AutoWrapText(true)
                .ColorAndOpacity(FLinearColor(.72f, .67f, .52f)) ]

            + SVerticalBox::Slot().AutoHeight()
            [
                SNew(SExpandableArea).InitiallyCollapsed(true)
                .HeaderContent()
                [ SNew(STextBlock).Text(LOCTEXT("Advanced", "Advanced Settings")) ]
                .BodyContent()
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 12, 8, 5)
                    [ SNew(STextBlock).Text(LOCTEXT("MappingLabel", "Bone Mapping File (Optional)")) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 10)
                    [
                        SNew(SHorizontalBox)
                        + SHorizontalBox::Slot().FillWidth(1.f)
                        [ SNew(SEditableTextBox).Tag(TEXT("CardCap.BoneMapping"))
                            .HintText(LOCTEXT("DefaultMapping", "Leave blank to use the default mapping"))
                            .Text_Lambda([this] { return FText::FromString(BoneMappingPath); })
                            .IsReadOnly_Lambda([this] { return !CanEdit(); })
                            .OnTextChanged_Lambda([this](const FText& Text) { if (CanEdit()) { BoneMappingPath = Text.ToString(); } }) ]
                        + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                        [ SNew(SButton).Text(LOCTEXT("BrowseMapping", "Browse..."))
                            .IsEnabled_Lambda([this] { return CanEdit(); })
                            .OnClicked(this, &SCardCapPanel::BrowseBoneMapping) ]
                    ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 10)
                    [ SNew(SCheckBox).Tag(TEXT("CardCap.AllowBlurry"))
                        .IsEnabled_Lambda([this] { return CanEdit(); })
                        .IsChecked_Lambda([this] { return bAllowBlurry ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                        .OnCheckStateChanged_Lambda([this](ECheckBoxState State) { bAllowBlurry = State == ECheckBoxState::Checked; })
                        [ SNew(STextBlock).Text(LOCTEXT("AllowBlurry", "Allow drafts from blurry video")) ] ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 8)
                    [ SNew(SCheckBox).Tag(TEXT("CardCap.RenderPreview"))
                        .IsEnabled_Lambda([this] { return CanEdit(); })
                        .IsChecked_Lambda([this] { return bRenderPreview ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                        .OnCheckStateChanged_Lambda([this](ECheckBoxState State) { bRenderPreview = State == ECheckBoxState::Checked; })
                        [ SNew(STextBlock).Text(LOCTEXT("RenderPreview", "Generate a comparison video")) ] ]
                ]
            ]
        ]
    ];
}

const FCardCapJobSnapshot& SCardCapPanel::Snapshot() const
{
    static const FCardCapJobSnapshot Empty;
    return Bridge.IsValid() ? Bridge->GetSnapshot() : Empty;
}

bool SCardCapPanel::IsRunning() const { return Bridge.IsValid() && Snapshot().IsRunning(); }
bool SCardCapPanel::CanEdit() const { return Bridge.IsValid() && Bridge->CanStart(); }

void SCardCapPanel::SetVideoPath(const FString& InPath)
{
    if (CanEdit())
    {
        VideoPath = CleanPath(InPath);
        LocalError.Reset();
    }
}

bool SCardCapPanel::StartProcessing(FString& OutError)
{
    OutError.Reset();
    if (!Bridge.IsValid())
    {
        OutError = LOCTEXT("NoBridge", "The processing service is not ready. Open this panel again.").ToString();
    }
    else if (!Bridge->CanStart())
    {
        OutError = LOCTEXT("AlreadyRunning", "This video is still processing. Wait for it to finish or cancel it first.").ToString();
    }
    else if (CleanPath(VideoPath).IsEmpty())
    {
        OutError = LOCTEXT("ChooseFirst", "Choose a video first.").ToString();
    }
    else
    {
        FCardCapJobOptions Options;
        Options.VideoPath = CleanPath(VideoPath);
        Options.BoneMappingPath = CleanPath(BoneMappingPath);
        Options.bAllowBlurry = bAllowBlurry;
        Options.bRenderPreview = bRenderPreview;
        if (Bridge->Start(Options, OutError))
        {
            VideoPath = Options.VideoPath;
            LocalError.Reset();
            return true;
        }
        if (OutError.IsEmpty())
        {
            OutError = LOCTEXT("CouldNotStart", "Processing could not be started. Check the log.").ToString();
        }
    }
    LocalError = OutError;
    return false;
}

FReply SCardCapPanel::OnStartClicked()
{
    FString Error;
    StartProcessing(Error);
    return FReply::Handled();
}

FReply SCardCapPanel::OnCancelClicked()
{
    if (IsRunning()) { Bridge->Cancel(); }
    return FReply::Handled();
}

bool SCardCapPanel::ChooseFile(const FText& Title, const FString& Filter, const FString& Current, FString& OutPath)
{
    IDesktopPlatform* Desktop = FDesktopPlatformModule::Get();
    if (!Desktop)
    {
        LocalError = LOCTEXT("NoFileDialog", "The file picker could not be opened. Paste the full file path instead.").ToString();
        return false;
    }
    const TSharedPtr<SWindow> Window = FSlateApplication::Get().FindWidgetWindow(AsShared());
    const void* Handle = Window.IsValid() && Window->GetNativeWindow().IsValid()
        ? Window->GetNativeWindow()->GetOSWindowHandle() : nullptr;
    TArray<FString> Files;
    const FString Directory = Current.IsEmpty() ? FPaths::ProjectDir() : FPaths::GetPath(CleanPath(Current));
    if (Desktop->OpenFileDialog(Handle, Title.ToString(), Directory, TEXT(""), Filter, 0, Files) && !Files.IsEmpty())
    {
        OutPath = FPaths::ConvertRelativePathToFull(Files[0]);
        LocalError.Reset();
        return true;
    }
    return false;
}

FReply SCardCapPanel::BrowseVideo()
{
    if (CanEdit())
    {
        FString Selected;
        if (ChooseFile(LOCTEXT("ChooseVideoDialog", "Choose a Cardistry Video"),
            TEXT("Video Files (*.mp4;*.mov;*.m4v;*.avi;*.mkv)|*.mp4;*.mov;*.m4v;*.avi;*.mkv"), VideoPath, Selected))
        {
            SetVideoPath(Selected);
        }
    }
    return FReply::Handled();
}

FReply SCardCapPanel::BrowseBoneMapping()
{
    if (CanEdit())
    {
        FString Selected;
        if (ChooseFile(LOCTEXT("ChooseMappingDialog", "Choose a Bone Mapping File"), TEXT("JSON Files (*.json)|*.json"), BoneMappingPath, Selected))
        {
            BoneMappingPath = Selected;
        }
    }
    return FReply::Handled();
}

FText SCardCapPanel::StatusText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (!LocalError.IsEmpty() || State.Status == TEXT("failed")) { return LOCTEXT("Failed", "Processing Incomplete"); }
    if (State.Status == TEXT("cancelling")) { return LOCTEXT("Cancelling", "Cancelling..."); }
    if (State.Status == TEXT("cancelled")) { return LOCTEXT("Cancelled", "Cancelled"); }
    if (State.Status == TEXT("succeeded")) { return LOCTEXT("Succeeded", "Processing Complete"); }
    if (State.Status == TEXT("idle")) { return LOCTEXT("Idle", "Ready"); }
    if (State.Stage == TEXT("preflight")) { return LOCTEXT("Preflight", "Checking Video"); }
    if (State.Stage == TEXT("reconstruct")) { return LOCTEXT("Reconstruct", "Reconstructing Hand Motion"); }
    if (State.Stage == TEXT("solve")) { return LOCTEXT("Solve", "Solving Motion and Position"); }
    if (State.Stage == TEXT("import")) { return LOCTEXT("Import", "Preparing Editor Assets"); }
    if (State.Stage == TEXT("bake")) { return LOCTEXT("Bake", "Generating Animation"); }
    if (State.Stage == TEXT("verify")) { return LOCTEXT("Verify", "Checking Results"); }
    if (State.Stage == TEXT("preview")) { return LOCTEXT("Preview", "Preparing Preview"); }
    if (State.Stage == TEXT("complete")) { return LOCTEXT("Complete", "Finishing"); }
    return LOCTEXT("Running", "Processing...");
}

FText SCardCapPanel::MessageText() const
{
    if (!LocalError.IsEmpty()) { return FText::FromString(LocalError); }
    return JobMessageText(Snapshot());
}

FText SCardCapPanel::JobMessageText(const FCardCapJobSnapshot& State)
{
    if (!State.Error.IsEmpty())
    {
        if (!State.HistoricalError.IsEmpty() && State.Error == State.HistoricalError)
        {
            return LOCTEXT("HistoricalError", "This saved job reported an error in an earlier version. Check View Log and status.json in the Result Folder for the original details.");
        }
        return FText::FromString(State.Error);
    }
    if (State.Status == TEXT("succeeded"))
    {
        return LOCTEXT("SuccessMessage", "Processing is complete. Open the scene or preview video to review the result.");
    }
    if (State.Status == TEXT("cancelled"))
    {
        return LOCTEXT("CancelledMessage", "Cancelled. Existing files remain in the result folder.");
    }
    if (State.Status == TEXT("idle")) { return FText::GetEmpty(); }
    return FText::FromString(State.Message);
}

FText SCardCapPanel::ProgressText() const
{
    if (Snapshot().Status == TEXT("idle")) { return FText::GetEmpty(); }
    return FText::FromString(FString::Printf(TEXT("%d%%"), FMath::RoundToInt(FMath::Clamp(Snapshot().Progress, 0., 1.) * 100.)));
}

FText SCardCapPanel::FrameTimeText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (State.Status == TEXT("idle")) { return LOCTEXT("NoProgress", "Frame progress and elapsed time will appear after processing starts."); }
    const int64 Seconds = FMath::FloorToInt64(FMath::Max(0., State.ElapsedSeconds));
    const FText Elapsed = FText::Format(LOCTEXT("Elapsed", "Elapsed: {0} min {1} sec"), FText::AsNumber(Seconds / 60), FText::AsNumber(Seconds % 60));
    if (State.FramesTotal <= 0) { return Elapsed; }
    return FText::Format(LOCTEXT("FrameElapsed", "Processed {0} / {1} frames  |  {2}"),
        FText::AsNumber(FMath::Clamp(State.FramesCompleted, 0, State.FramesTotal)), FText::AsNumber(State.FramesTotal), Elapsed);
}

FText SCardCapPanel::ConfidenceRangesText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (State.LowConfidenceRanges.IsEmpty())
    {
        return State.Status == TEXT("succeeded") ? LOCTEXT("NoFlaggedRanges", "No sections have been flagged for review.")
            : LOCTEXT("RangesPending", "Frame ranges that need review will appear after processing.");
    }
    TArray<FString> Pieces;
    for (const FIntPoint& Range : State.LowConfidenceRanges)
    {
        Pieces.Add(Range.X == Range.Y ? FString::FromInt(Range.X)
            : FString::Printf(TEXT("%d–%d"), Range.X, Range.Y));
    }
    return FText::Format(LOCTEXT("Ranges", "Frame ranges (starting at 0): {0}"), FText::FromString(FString::Join(Pieces, TEXT(", "))));
}

FText SCardCapPanel::SourceNoteText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    TArray<FString> Notes;
    if (State.IsPerHandLocal())
    {
        Notes.Add(LOCTEXT("UnknownHandSpace", "The source camera parameters and relative hand positions are unknown. Review the separate left and right hand previews; they do not place both hands in a shared space").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("model-conditioned"))
    {
        Notes.Add(LOCTEXT("ConditionalCamera", "The preview uses the model's assumed camera; the source camera parameters have not been measured").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("prior-based") || State.IntrinsicsSource == TEXT("metadata_35mm_equivalent"))
    {
        Notes.Add(LOCTEXT("EstimatedCamera", "Camera parameters are estimated").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("explicit_calibration") || State.IntrinsicsSource == TEXT("adjacent_chessboard"))
    {
        Notes.Add(LOCTEXT("CalibrationCamera", "Camera parameters come from calibration").ToString());
    }
    if (State.ScaleConfidence == TEXT("unknown")) { Notes.Add(LOCTEXT("UnknownScale", "Scale is relative; actual size has not been measured").ToString()); }
    else if (State.ScaleConfidence == TEXT("low")) { Notes.Add(LOCTEXT("LowScale", "Scale confidence is low; review is recommended").ToString()); }
    else if (State.ScaleConfidence == TEXT("medium")) { Notes.Add(LOCTEXT("MediumScale", "Scale confidence is medium").ToString()); }
    else if (State.ScaleConfidence == TEXT("high")) { Notes.Add(LOCTEXT("HighScale", "Scale confidence is high").ToString()); }
    return FText::FromString(FString::Join(Notes, TEXT("; ")));
}

#undef LOCTEXT_NAMESPACE
