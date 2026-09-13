#include "CardCapPythonBridge.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "Misc/AutomationTest.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapStatusBoundaryTest, "CardistryCapture.Editor.JobStatusContract",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCardCapStatusBoundaryTest::RunTest(const FString& Parameters)
{
    FCardCapJobSnapshot Snapshot;
    Snapshot.OutputDirectory = TEXT("C:/project/Saved/CardistryCapture/Runs/test");
    FString Error;
    const FString Running = TEXT(R"({"schema_version":1,"job_id":"test","status":"running","stage":"reconstruct","progress":0.3,"elapsed_seconds":2,"frames_completed":4,"frames_total":12})");
    TestTrue(TEXT("Running status parses"), FCardCapPythonBridge::ParseStatus(Running, TEXT("test"), Snapshot, Error));
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
    TestFalse(TEXT("Unknown coordinate frame rejected atomically"), FCardCapPythonBridge::ParseStatus(LocalSuccess.Replace(TEXT("per_hand_wrist_local"), TEXT("invented_joint_space")), TEXT("test"), Snapshot, Error));
    TestTrue(TEXT("Rejected frame keeps previous state"), Snapshot.IsPerHandLocal());
    TestTrue(TEXT("Legacy result remains supported"), FCardCapPythonBridge::ParseStatus(Success, TEXT("test"), Snapshot, Error));
    TestFalse(TEXT("Legacy missing field does not inherit local restriction"), Snapshot.IsPerHandLocal());
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
#endif
