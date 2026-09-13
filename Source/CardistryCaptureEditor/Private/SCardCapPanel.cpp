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
            [ SNew(STextBlock).Text(LOCTEXT("Title", "花切动作捕获"))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 22)) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [ SNew(STextBlock).Text(LOCTEXT("Introduction", "选择一段视频，生成可在编辑器中查看的双手动作。"))
                .AutoWrapText(true).ColorAndOpacity(FLinearColor(.7f, .73f, .78f)) ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 6)
            [ SNew(STextBlock).Text(LOCTEXT("VideoLabel", "源视频")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 14)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().FillWidth(1.f)
                [ SNew(SEditableTextBox).Tag(TEXT("CardCap.VideoPath"))
                    .HintText(LOCTEXT("VideoHint", "选择视频，或粘贴完整路径"))
                    .Text_Lambda([this] { return FText::FromString(VideoPath); })
                    .IsReadOnly_Lambda([this] { return !CanEdit(); })
                    .OnTextChanged_Lambda([this](const FText& Text) { if (CanEdit()) { VideoPath = Text.ToString(); LocalError.Reset(); } }) ]
                + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                [ SNew(SButton).Tag(TEXT("CardCap.BrowseVideo"))
                    .Text(LOCTEXT("BrowseVideo", "选择视频…"))
                    .IsEnabled_Lambda([this] { return CanEdit(); })
                    .OnClicked(this, &SCardCapPanel::BrowseVideo) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [
                SNew(SHorizontalBox)
                + SHorizontalBox::Slot().AutoWidth()
                [ SNew(SButton).Tag(TEXT("CardCap.Start"))
                    .Text(LOCTEXT("Start", "开始处理"))
                    .ContentPadding(FMargin(24, 8))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Bridge->CanStart() && !CleanPath(VideoPath).IsEmpty(); })
                    .OnClicked(this, &SCardCapPanel::OnStartClicked) ]
                + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                [ SNew(SButton).Tag(TEXT("CardCap.Cancel"))
                    .Text(LOCTEXT("Cancel", "取消处理"))
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
            [ SNew(STextBlock).Text(LOCTEXT("CloseNote", "关闭此面板不会中断处理。"))
                .ColorAndOpacity(FLinearColor(.55f, .59f, .65f)) ]

            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 14)
            [ SNew(SSeparator) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [ SNew(STextBlock).Text(LOCTEXT("Results", "处理结果"))
                .Font(FCoreStyle::GetDefaultFontStyle("Bold", 14)) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 12)
            [
                SNew(SWrapBox).UseAllottedSize(true).InnerSlotPadding(FVector2D(8, 8))
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenResult")).Text(LOCTEXT("OpenScene", "打开场景"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().MapAsset.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenResultScene(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenAnimation")).Text(LOCTEXT("OpenAnimation", "打开动画"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().AnimationAsset.IsEmpty() && !Snapshot().IsPerHandLocal(); })
                    .ToolTipText_Lambda([this] { return Snapshot().IsPerHandLocal()
                        ? LOCTEXT("LocalAnimationUnavailable", "双手相对位置未知，请查看左右手局部预览。") : FText::GetEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenResultAnimation(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.Preview")).Text(LOCTEXT("OpenPreview", "预览视频"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && Snapshot().Status == TEXT("succeeded") && !Snapshot().PreviewVideo.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenPreview(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenFolder")).Text(LOCTEXT("OpenFolder", "结果文件夹"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && !Snapshot().OutputDirectory.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenOutputFolder(); } return FReply::Handled(); }) ]
                + SWrapBox::Slot()
                [ SNew(SButton).Tag(TEXT("CardCap.OpenLog")).Text(LOCTEXT("OpenLog", "查看日志"))
                    .IsEnabled_Lambda([this] { return Bridge.IsValid() && !Snapshot().LogPath.IsEmpty(); })
                    .OnClicked_Lambda([this] { if (Bridge.IsValid()) { Bridge->OpenLog(); } return FReply::Handled(); }) ]
            ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 7)
            [ SNew(STextBlock).Text(LOCTEXT("ReviewRanges", "黄色区段建议复核")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 7)
            [ SNew(SCardCapConfidenceBar).Tag(TEXT("CardCap.ConfidenceBar")).Bridge(Bridge)
                .ToolTipText(LOCTEXT("ReviewTooltip", "黄色表示结果中标记的低置信度区段；其余区段不代表已经人工验证。")) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 8)
            [ SNew(STextBlock).Text(this, &SCardCapPanel::ConfidenceRangesText).AutoWrapText(true) ]
            + SVerticalBox::Slot().AutoHeight().Padding(0, 0, 0, 18)
            [ SNew(STextBlock).Text(this, &SCardCapPanel::SourceNoteText).AutoWrapText(true)
                .ColorAndOpacity(FLinearColor(.72f, .67f, .52f)) ]

            + SVerticalBox::Slot().AutoHeight()
            [
                SNew(SExpandableArea).InitiallyCollapsed(true)
                .HeaderContent()
                [ SNew(STextBlock).Text(LOCTEXT("Advanced", "高级设置")) ]
                .BodyContent()
                [
                    SNew(SVerticalBox)
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 12, 8, 5)
                    [ SNew(STextBlock).Text(LOCTEXT("MappingLabel", "骨骼映射文件（可选）")) ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 10)
                    [
                        SNew(SHorizontalBox)
                        + SHorizontalBox::Slot().FillWidth(1.f)
                        [ SNew(SEditableTextBox).Tag(TEXT("CardCap.BoneMapping"))
                            .HintText(LOCTEXT("DefaultMapping", "留空时使用默认映射"))
                            .Text_Lambda([this] { return FText::FromString(BoneMappingPath); })
                            .IsReadOnly_Lambda([this] { return !CanEdit(); })
                            .OnTextChanged_Lambda([this](const FText& Text) { if (CanEdit()) { BoneMappingPath = Text.ToString(); } }) ]
                        + SHorizontalBox::Slot().AutoWidth().Padding(8, 0, 0, 0)
                        [ SNew(SButton).Text(LOCTEXT("BrowseMapping", "浏览…"))
                            .IsEnabled_Lambda([this] { return CanEdit(); })
                            .OnClicked(this, &SCardCapPanel::BrowseBoneMapping) ]
                    ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 10)
                    [ SNew(SCheckBox).Tag(TEXT("CardCap.AllowBlurry"))
                        .IsEnabled_Lambda([this] { return CanEdit(); })
                        .IsChecked_Lambda([this] { return bAllowBlurry ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                        .OnCheckStateChanged_Lambda([this](ECheckBoxState State) { bAllowBlurry = State == ECheckBoxState::Checked; })
                        [ SNew(STextBlock).Text(LOCTEXT("AllowBlurry", "允许模糊视频生成草稿")) ] ]
                    + SVerticalBox::Slot().AutoHeight().Padding(8, 0, 8, 8)
                    [ SNew(SCheckBox).Tag(TEXT("CardCap.RenderPreview"))
                        .IsEnabled_Lambda([this] { return CanEdit(); })
                        .IsChecked_Lambda([this] { return bRenderPreview ? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
                        .OnCheckStateChanged_Lambda([this](ECheckBoxState State) { bRenderPreview = State == ECheckBoxState::Checked; })
                        [ SNew(STextBlock).Text(LOCTEXT("RenderPreview", "生成对照视频")) ] ]
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
        OutError = LOCTEXT("NoBridge", "处理服务尚未就绪，请重新打开面板。").ToString();
    }
    else if (!Bridge->CanStart())
    {
        OutError = LOCTEXT("AlreadyRunning", "当前视频仍在处理中，请等待完成或先取消。").ToString();
    }
    else if (CleanPath(VideoPath).IsEmpty())
    {
        OutError = LOCTEXT("ChooseFirst", "请先选择一段视频。").ToString();
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
            OutError = LOCTEXT("CouldNotStart", "未能开始处理，请查看日志。").ToString();
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
        LocalError = LOCTEXT("NoFileDialog", "无法打开文件选择窗口，可直接粘贴文件路径。").ToString();
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
        if (ChooseFile(LOCTEXT("ChooseVideoDialog", "选择花切视频"),
            TEXT("视频文件 (*.mp4;*.mov;*.m4v;*.avi;*.mkv)|*.mp4;*.mov;*.m4v;*.avi;*.mkv"), VideoPath, Selected))
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
        if (ChooseFile(LOCTEXT("ChooseMappingDialog", "选择骨骼映射文件"), TEXT("JSON 文件 (*.json)|*.json"), BoneMappingPath, Selected))
        {
            BoneMappingPath = Selected;
        }
    }
    return FReply::Handled();
}

FText SCardCapPanel::StatusText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (!LocalError.IsEmpty() || State.Status == TEXT("failed")) { return LOCTEXT("Failed", "处理未完成"); }
    if (State.Status == TEXT("cancelling")) { return LOCTEXT("Cancelling", "正在取消…"); }
    if (State.Status == TEXT("cancelled")) { return LOCTEXT("Cancelled", "已取消"); }
    if (State.Status == TEXT("succeeded")) { return LOCTEXT("Succeeded", "处理完成"); }
    if (State.Status == TEXT("idle")) { return LOCTEXT("Idle", "准备开始"); }
    if (State.Stage == TEXT("preflight")) { return LOCTEXT("Preflight", "正在检查视频"); }
    if (State.Stage == TEXT("reconstruct")) { return LOCTEXT("Reconstruct", "正在识别双手动作"); }
    if (State.Stage == TEXT("solve")) { return LOCTEXT("Solve", "正在整理动作与位置"); }
    if (State.Stage == TEXT("import")) { return LOCTEXT("Import", "正在准备编辑器素材"); }
    if (State.Stage == TEXT("bake")) { return LOCTEXT("Bake", "正在生成动画"); }
    if (State.Stage == TEXT("verify")) { return LOCTEXT("Verify", "正在检查结果"); }
    if (State.Stage == TEXT("preview")) { return LOCTEXT("Preview", "正在准备结果预览"); }
    if (State.Stage == TEXT("complete")) { return LOCTEXT("Complete", "正在完成处理"); }
    return LOCTEXT("Running", "正在处理…");
}

FText SCardCapPanel::MessageText() const
{
    if (!LocalError.IsEmpty()) { return FText::FromString(LocalError); }
    if (!Snapshot().Error.IsEmpty()) { return FText::FromString(Snapshot().Error); }
    return FText::FromString(Snapshot().Message);
}

FText SCardCapPanel::ProgressText() const
{
    if (Snapshot().Status == TEXT("idle")) { return FText::GetEmpty(); }
    return FText::FromString(FString::Printf(TEXT("%d%%"), FMath::RoundToInt(FMath::Clamp(Snapshot().Progress, 0., 1.) * 100.)));
}

FText SCardCapPanel::FrameTimeText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (State.Status == TEXT("idle")) { return LOCTEXT("NoProgress", "开始后将显示帧数和耗时。"); }
    const int64 Seconds = FMath::FloorToInt64(FMath::Max(0., State.ElapsedSeconds));
    const FText Elapsed = FText::Format(LOCTEXT("Elapsed", "已用 {0} 分 {1} 秒"), FText::AsNumber(Seconds / 60), FText::AsNumber(Seconds % 60));
    if (State.FramesTotal <= 0) { return Elapsed; }
    return FText::Format(LOCTEXT("FrameElapsed", "已处理 {0} / {1} 帧  ·  {2}"),
        FText::AsNumber(FMath::Clamp(State.FramesCompleted, 0, State.FramesTotal)), FText::AsNumber(State.FramesTotal), Elapsed);
}

FText SCardCapPanel::ConfidenceRangesText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    if (State.LowConfidenceRanges.IsEmpty())
    {
        return State.Status == TEXT("succeeded") ? LOCTEXT("NoFlaggedRanges", "没有标记需要复核的区段。")
            : LOCTEXT("RangesPending", "处理后显示需要复核的帧范围。");
    }
    TArray<FString> Pieces;
    for (const FIntPoint& Range : State.LowConfidenceRanges)
    {
        Pieces.Add(Range.X == Range.Y ? FString::FromInt(Range.X)
            : FString::Printf(TEXT("%d–%d"), Range.X, Range.Y));
    }
    return FText::Format(LOCTEXT("Ranges", "帧范围（从 0 开始）：{0}"), FText::FromString(FString::Join(Pieces, TEXT("、"))));
}

FText SCardCapPanel::SourceNoteText() const
{
    const FCardCapJobSnapshot& State = Snapshot();
    TArray<FString> Notes;
    if (State.IsPerHandLocal())
    {
        Notes.Add(LOCTEXT("UnknownHandSpace", "原片相机参数及双手相对位置未知，请查看左右手局部预览；局部显示不代表两手处于同一空间").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("model-conditioned"))
    {
        Notes.Add(LOCTEXT("ConditionalCamera", "预览使用模型相机假设，尚未测得原片相机参数").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("prior-based") || State.IntrinsicsSource == TEXT("metadata_35mm_equivalent"))
    {
        Notes.Add(LOCTEXT("EstimatedCamera", "画面参数使用估计值").ToString());
    }
    else if (State.IntrinsicsSource == TEXT("explicit_calibration") || State.IntrinsicsSource == TEXT("adjacent_chessboard"))
    {
        Notes.Add(LOCTEXT("CalibrationCamera", "画面参数来自标定").ToString());
    }
    if (State.ScaleConfidence == TEXT("unknown")) { Notes.Add(LOCTEXT("UnknownScale", "尺寸为相对量，未测得实际大小").ToString()); }
    else if (State.ScaleConfidence == TEXT("low")) { Notes.Add(LOCTEXT("LowScale", "尺寸依据较弱，建议复核").ToString()); }
    else if (State.ScaleConfidence == TEXT("medium")) { Notes.Add(LOCTEXT("MediumScale", "尺寸依据一般").ToString()); }
    else if (State.ScaleConfidence == TEXT("high")) { Notes.Add(LOCTEXT("HighScale", "尺寸依据较充分").ToString()); }
    return FText::FromString(FString::Join(Notes, TEXT("；")));
}

#undef LOCTEXT_NAMESPACE
