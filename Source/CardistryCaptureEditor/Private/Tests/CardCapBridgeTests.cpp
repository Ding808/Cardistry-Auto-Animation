#include "CardCapPythonBridge.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "SCardCapPanel.h"
#include "Framework/Application/SlateApplication.h"
#include "Internationalization/Culture.h"
#include "Internationalization/Internationalization.h"
#include "Layout/Children.h"
#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"
#include "Widgets/Text/STextBlock.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapStatusBoundaryTest, "CardistryCapture.Editor.JobStatusContract",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCardCapStatusBoundaryTest::RunTest(const FString& Parameters)
{
    FCardCapJobSnapshot Snapshot;
    Snapshot.OutputDirectory = TEXT("C:/project/Saved/CardistryCapture/Runs/test");
    FString Error;
    const FString Running = TEXT(R"({"schema_version":1,"job_id":"test","status":"running","stage":"reconstruct","progress":0.3,"elapsed_seconds":2,"frames_completed":4,"frames_total":12})");
    TestTrue(TEXT("Running status parses"), FCardCapPythonBridge::ParseStatus(Running, TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("Unmarked legacy status has no English message guarantee"), Snapshot.bMessagesAreEnglish);
    const FString EnglishRunning = Running.Replace(TEXT("\"schema_version\":1"), TEXT("\"schema_version\":1,\"message_language\":\"en\""));
    TestTrue(TEXT("English status marker parses"), FCardCapPythonBridge::ParseStatus(EnglishRunning, TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Current English messages are identified"), Snapshot.bMessagesAreEnglish);
    Snapshot.HistoricalError = TEXT("Previous error");
    TestTrue(TEXT("An unmarked update remains compatible"), FCardCapPythonBridge::ParseStatus(Running, TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("An unmarked update does not inherit the previous language marker"), Snapshot.bMessagesAreEnglish);
    TestTrue(TEXT("A new status update clears historical error presentation"), Snapshot.HistoricalError.IsEmpty());
    TestTrue(TEXT("Running state"), Snapshot.IsRunning());
    TestEqual(TEXT("Frame progress"), Snapshot.FramesCompleted, 4);
    TestEqual(TEXT("Output ownership remains in bridge"), Snapshot.OutputDirectory, FString(TEXT("C:/project/Saved/CardistryCapture/Runs/test")));
    TestFalse(TEXT("Different job must not replace current status"), FCardCapPythonBridge::ParseStatus(Running, TEXT("other"), Snapshot, Error));
    TestEqual(TEXT("Rejected update is atomic"), Snapshot.JobId, FString(TEXT("test")));
    TestFalse(TEXT("Partial write ignored"), FCardCapPythonBridge::ParseStatus(TEXT("{\"job_id\":"), TEXT("test"), Snapshot, Error));
    const FString EmptySuccess = Running.Replace(TEXT("running"), TEXT("succeeded"));
    TestFalse(TEXT("Exit marker without outputs is not success"), FCardCapPythonBridge::ParseStatus(EmptySuccess, TEXT("test"), Snapshot, Error));
    const FString Success = TEXT(R"({"schema_version":1,"job_id":"test","status":"succeeded","stage":"complete","progress":1,"elapsed_seconds":10,"result":{"map_asset":"/CardistryCapture/GeneratedResearch/test/Review/Hands_Map","sequence_asset":"/CardistryCapture/GeneratedResearch/test/Review/Hands_Sequence","animation_asset":"/CardistryCapture/GeneratedResearch/test/Hands","capture_file":"C:/project/Saved/CardistryCapture/Runs/test/capture.json","frame_count":12,"fps":30,"scale_confidence":"low","low_confidence_ranges":[[0,2],[8,11]]}})");
    TestTrue(TEXT("Complete output contract accepted"), FCardCapPythonBridge::ParseStatus(Success, TEXT("test"), Snapshot, Error));
    const FString ProjectSuccess = Success.Replace(TEXT("/CardistryCapture/GeneratedResearch/"), TEXT("/Game/CardistryCapture/Generated/"));
    TestTrue(TEXT("Project-owned output contract accepted"), FCardCapPythonBridge::ParseStatus(ProjectSuccess, TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Project-owned map path retained"), Snapshot.MapAsset.StartsWith(TEXT("/Game/CardistryCapture/Generated/test/")));
    TestFalse(TEXT("Complete is not running"), Snapshot.IsRunning());
    TestEqual(TEXT("Missing confidence spans retained"), Snapshot.LowConfidenceRanges.Num(), 2);
    TestEqual(TEXT("Inclusive end frame"), Snapshot.LowConfidenceRanges[1].Y, 11);
    TestFalse(TEXT("Out-of-timeline low confidence rejected"), FCardCapPythonBridge::ParseStatus(Success.Replace(TEXT("8,11"), TEXT("8,12")), TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("Invalid progress rejected"), FCardCapPythonBridge::ParseStatus(Running.Replace(TEXT("0.3"), TEXT("1.3")), TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("Unknown state rejected"), FCardCapPythonBridge::ParseStatus(Running.Replace(TEXT("running"), TEXT("done")), TEXT("test"), Snapshot, Error));
    const FString LocalSuccess = Success.Replace(TEXT("\"scale_confidence\":\"low\""), TEXT("\"scale_confidence\":\"unknown\",\"coordinate_frame\":\"per_hand_wrist_local\""));
    TestTrue(TEXT("Local result coordinate frame retained"), FCardCapPythonBridge::ParseStatus(LocalSuccess, TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Local result cannot use combined raw animation viewer"), Snapshot.IsPerHandLocal());
    const FString CommonDisplay = LocalSuccess.Replace(TEXT("\"scale_confidence\":\"unknown\""),
        TEXT("\"scale_confidence\":\"unknown\",\"display_view_mode\":\"common_space\",\"display_assumed_focal_px\":1234.567891234"));
    TestTrue(TEXT("A display-only common view parses independently from capture coordinates"), FCardCapPythonBridge::ParseStatus(CommonDisplay, TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Common display is available"), Snapshot.HasCommonDisplay());
    TestTrue(TEXT("Common display does not make the source animation globally positioned"), Snapshot.IsPerHandLocal());
    TestEqual(TEXT("Display focal metadata retains double precision"), Snapshot.DisplayAssumedFocalPx, 1234.567891234);
    TestFalse(TEXT("Unknown coordinate frame rejected atomically"), FCardCapPythonBridge::ParseStatus(LocalSuccess.Replace(TEXT("per_hand_wrist_local"), TEXT("invented_joint_space")), TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Rejected frame keeps previous state"), Snapshot.IsPerHandLocal());
    TestTrue(TEXT("Legacy result remains supported"), FCardCapPythonBridge::ParseStatus(Success, TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("Legacy missing field does not inherit local restriction"), Snapshot.IsPerHandLocal());
    TestFalse(TEXT("Legacy missing display mode does not inherit a common display"), Snapshot.HasCommonDisplay());
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapArgumentBoundaryTest, "CardistryCapture.Editor.ProcessArguments",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCardCapArgumentBoundaryTest::RunTest(const FString& Parameters)
{
    TestEqual(TEXT("Unicode and spaces are one argument"), FCardCapPythonBridge::QuoteArgument(TEXT("C:\\本机 视频\\take 01.mp4")), FString(TEXT("\"C:\\本机 视频\\take 01.mp4\"")));
    TestEqual(TEXT("Trailing slash cannot escape closing quote"), FCardCapPythonBridge::QuoteArgument(TEXT("C:\\folder\\")), FString(TEXT("\"C:\\folder\\\\\"")));
    TestEqual(TEXT("Embedded quotes are argv data"), FCardCapPythonBridge::QuoteArgument(TEXT("a\"b")), FString(TEXT("\"a\\\"b\"")));
    TestEqual(TEXT("Empty argv preserved"), FCardCapPythonBridge::QuoteArgument(TEXT("")), FString(TEXT("\"\"")));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapEnglishPanelTest, "CardistryCapture.Editor.EnglishPanelUnderChineseCulture",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCardCapEnglishPanelTest::RunTest(const FString& Parameters)
{
    if (!TestTrue(TEXT("Slate is initialized"), FSlateApplication::IsInitialized())) { return false; }
    FInternationalization& Internationalization = FInternationalization::Get();
    const FString PreviousLanguage = Internationalization.GetCurrentLanguage()->GetName();
    const FString PreviousLocale = Internationalization.GetCurrentLocale()->GetName();
    ON_SCOPE_EXIT
    {
        Internationalization.SetCurrentLanguage(PreviousLanguage);
        Internationalization.SetCurrentLocale(PreviousLocale);
    };
    if (!TestTrue(TEXT("Chinese editor culture is available"), Internationalization.SetCurrentLanguageAndLocale(TEXT("zh-Hans"))))
    { return false; }

    // Inspect the real Slate widgets without starting a worker or changing any job.
    const TSharedRef<SCardCapPanel> Panel = SNew(SCardCapPanel);
    TArray<FString> Labels;
    TFunction<void(const TSharedRef<SWidget>&)> CollectLabels = [&](const TSharedRef<SWidget>& Widget)
    {
        if (Widget->GetType() == FName(TEXT("STextBlock")))
        {
            Labels.Add(StaticCastSharedRef<STextBlock>(Widget)->GetText().ToString());
        }
        FChildren* Children = Widget->GetChildren();
        for (int32 Index = 0; Children && Index < Children->Num(); ++Index)
        {
            CollectLabels(Children->GetChildAt(Index));
        }
    };
    CollectLabels(Panel);
    for (const TCHAR* Expected : {
        TEXT("Cardistry Capture"), TEXT("Source Video"), TEXT("Choose Video..."),
        TEXT("Start Processing"), TEXT("Cancel Processing"), TEXT("Results"),
        TEXT("Open Scene"), TEXT("Open Animation"), TEXT("Preview Video"),
        TEXT("Result Folder"), TEXT("View Log"), TEXT("Advanced Settings"), TEXT("Use Separate Local Preview"), TEXT("Ready")})
    {
        TestTrue(FString::Printf(TEXT("English panel label under zh-Hans: %s"), Expected), Labels.Contains(FString(Expected)));
    }

    FString Error;
    TestFalse(TEXT("Unavailable service does not start processing"), Panel->StartProcessing(Error));
    TestEqual(TEXT("Panel validation error is English"), Error,
        FString(TEXT("The processing service is not ready. Open this panel again.")));
    Labels.Reset();
    CollectLabels(Panel);
    TestTrue(TEXT("The actual panel displays the English validation error"), Labels.Contains(Error));
    TestTrue(TEXT("The actual panel displays the English failure state"), Labels.Contains(TEXT("Processing Incomplete")));

    FCardCapJobSnapshot Saved;
    Saved.Message = TEXT("\u5904\u7406\u5b8c\u6210"); // A message saved by the previous release.
    Saved.Status = TEXT("succeeded");
    TestEqual(TEXT("A saved success message uses English presentation"), SCardCapPanel::JobMessageText(Saved).ToString(),
        FString(TEXT("Processing is complete. Open the scene or preview video to review the result.")));
    Saved.Status = TEXT("cancelled");
    TestEqual(TEXT("A saved cancellation uses English presentation"), SCardCapPanel::JobMessageText(Saved).ToString(),
        FString(TEXT("Cancelled. Existing files remain in the result folder.")));
    Saved.Status = TEXT("idle");
    TestTrue(TEXT("An idle job does not display a stale saved message"), SCardCapPanel::JobMessageText(Saved).IsEmpty());
    Saved.Status = TEXT("failed");
    Saved.Error = TEXT("\u91cd\u5efa\u5931\u8d25");
    Saved.HistoricalError = Saved.Error;
    const FString OriginalError = Saved.Error;
    TestEqual(TEXT("An unmarked restored error points to its original details in English"), SCardCapPanel::JobMessageText(Saved).ToString(),
        FString(TEXT("This saved job reported an error in an earlier version. Check View Log and status.json in the Result Folder for the original details.")));
    TestEqual(TEXT("The raw historical error is preserved"), Saved.Error, OriginalError);
    Saved.Error = TEXT("Could not read C:/\u672c\u673a \u89c6\u9891/take 01.mp4");
    TestEqual(TEXT("A new English error keeps its complete Unicode path"), SCardCapPanel::JobMessageText(Saved).ToString(), Saved.Error);
    Saved.HistoricalError.Empty();
    TestEqual(TEXT("A current or marked restored error remains verbatim"), SCardCapPanel::JobMessageText(Saved).ToString(), Saved.Error);
    return true;
}
#endif
