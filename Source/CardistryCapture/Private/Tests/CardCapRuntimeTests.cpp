#include "CardCapCoordConvert.h"
#include "CardCapJsonParser.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Misc/AutomationTest.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace CardCapTest
{
using FObject = TSharedPtr<FJsonObject>;
using FArray = TArray<TSharedPtr<FJsonValue>>;

FArray Numbers(std::initializer_list<double> Values)
{
    FArray Result;
    for (double V : Values) Result.Add(MakeShared<FJsonValueNumber>(V));
    return Result;
}

FString Serialize(const FObject& Object)
{
    FString Result;
    FJsonSerializer::Serialize(Object.ToSharedRef(), TJsonWriterFactory<>::Create(&Result));
    return Result;
}

FObject Clone(const FObject& Object)
{
    FObject Result;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Serialize(Object)), Result);
    return Result;
}

FObject Child(const FObject& Object, const TCHAR* Key)
{
    const FObject* Result = nullptr;
    Object->TryGetObjectField(Key, Result);
    return *Result;
}

FObject First(const FObject& Object, const TCHAR* Key)
{
    const FArray* Values = nullptr; Object->TryGetArrayField(Key, Values);
    const FObject* Result = nullptr; (*Values)[0]->TryGetObject(Result);
    return *Result;
}

// Synthetic unit fixture only: exercises a parser contract, never used as video/demo evidence.
FObject Fixture(const FCardCapBoneMapping& Mapping)
{
    FObject Root;
    const FString Json = TEXT(R"JSON({
        "format_version":"1.0",
        "meta":{"source_video":"unit-fixture.mp4","fps":30,"frame_count":3,"resolution":[1280,720],"processed_at":"2026-09-12T00:00:00Z","pipeline_version":"unit-test"},
        "camera":{"intrinsics":{"fx":1000,"fy":1000,"cx":640,"cy":360},"distortion":[0,0,0,0,0],"calibrated":false},
        "scale":{"meters_per_unit":1,"anchor_method":"hand_prior","confidence":0.2,"anchor_frames":[]},
        "hands":[],"packets":[],"events":{"splits":[],"merges":[],"releases":[]},
        "quality":{"mean_hand_confidence":0.5,"low_confidence_ranges":[[0,1]],"mean_reprojection_error_px":null,"warnings":["Synthetic unit fixture; not measured footage"]}
    })JSON");
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root);
    FObject Hand = MakeShared<FJsonObject>(), Frame = MakeShared<FJsonObject>(), Rotations = MakeShared<FJsonObject>();
    Hand->SetStringField(TEXT("side"), TEXT("right"));
    Hand->SetArrayField(TEXT("mano_shape"), Numbers({0,0,0,0,0,0,0,0,0,0}));
    Frame->SetNumberField(TEXT("frame"), 0); Frame->SetNumberField(TEXT("confidence"), 0.5);
    Frame->SetArrayField(TEXT("global_trans_cm"), Numbers({10,20,30}));
    Frame->SetArrayField(TEXT("global_rot_quat"), Numbers({0,0,0,1}));
    FArray Joints;
    for (int32 I = 0; I < 21; ++I) Joints.Add(MakeShared<FJsonValueArray>(Numbers({0,0,0})));
    Frame->SetArrayField(TEXT("joint_positions_cm"), Joints);
    Frame->SetArrayField(TEXT("occluded_joints"), Numbers({8,12}));
    const auto& Names = Mapping.Hands.FindChecked(TEXT("right")).BoneNames;
    for (int32 I = 1; I < Names.Num(); ++I) Rotations->SetArrayField(Names[I].ToString(), Numbers({0,0,0,1}));
    Frame->SetObjectField(TEXT("bone_rotations"), Rotations);
    Hand->SetArrayField(TEXT("frames"), {MakeShared<FJsonValueObject>(Frame)});
    Root->SetArrayField(TEXT("hands"), {MakeShared<FJsonValueObject>(Hand)});
    return Root;
}

FObject Packet(const FString& Id)
{
    FObject Object;
    const FString Json = TEXT(R"JSON({"id":"unit","birth_frame":0,"death_frame":2,"card_count_estimate":1,"card_count_confidence":0.1,"dimensions_cm":[6.35,8.89,0.03],"frames":[{"frame":0,"confidence":0.1,"position_cm":[0,0,0],"rotation_quat":[0,0,0,1],"linear_velocity_cm_s":[0,0,0],"angular_velocity_rad_s":[0,0,0],"contact_state":"FREE","contact_bones":[],"reprojection_error_px":1.2}]})JSON");
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Object);
    Object->SetStringField(TEXT("id"), Id); return Object;
}

FObject UnknownPacket()
{
    auto P = Packet(TEXT("unknown-properties-unit-fixture"));
    P->SetField(TEXT("card_count_estimate"), MakeShared<FJsonValueNull>());
    P->SetField(TEXT("card_count_confidence"), MakeShared<FJsonValueNull>());
    auto Dimensions = Numbers({6.35, 8.89}); Dimensions.Add(MakeShared<FJsonValueNull>());
    P->SetArrayField(TEXT("dimensions_cm"), Dimensions);
    auto F = First(P, TEXT("frames"));
    for (const TCHAR* Key : {TEXT("linear_velocity_cm_s"), TEXT("angular_velocity_rad_s"), TEXT("reprojection_error_px"), TEXT("contact_bones")})
        F->SetField(Key, MakeShared<FJsonValueNull>());
    F->SetStringField(TEXT("contact_state"), TEXT("UNKNOWN"));
    return P;
}

FObject Fixture12(const FCardCapBoneMapping& Mapping)
{
    auto Root = Fixture(Mapping);
    Root->SetStringField(TEXT("format_version"), TEXT("1.2"));
    Child(Root, TEXT("camera"))->SetField(TEXT("distortion"), MakeShared<FJsonValueNull>());
    auto Scale = Child(Root, TEXT("scale"));
    for (const TCHAR* Key : {TEXT("meters_per_unit"), TEXT("anchor_method"), TEXT("confidence")})
        Scale->SetField(Key, MakeShared<FJsonValueNull>());
    FObject Provenance = MakeShared<FJsonObject>(), Validity = MakeShared<FJsonObject>();
    Provenance->SetStringField(TEXT("camera_intrinsics"), TEXT("inferred"));
    Provenance->SetStringField(TEXT("camera_distortion"), TEXT("unobservable"));
    Provenance->SetStringField(TEXT("metric_scale"), TEXT("unobservable"));
    Provenance->SetStringField(TEXT("hand_geometry"), TEXT("inferred"));
    Provenance->SetStringField(TEXT("coordinate_units"), TEXT("conditional_ue_units"));
    Provenance->SetArrayField(TEXT("assumptions"), {MakeShared<FJsonValueString>(TEXT("Synthetic contract fixture: conditional pinhole projection and arbitrary display units; not measured footage"))});
    Validity->SetBoolField(TEXT("camera_intrinsics"), true);
    Validity->SetBoolField(TEXT("camera_distortion"), false);
    Validity->SetBoolField(TEXT("metric_scale"), false);
    Root->SetObjectField(TEXT("provenance"), Provenance); Root->SetObjectField(TEXT("validity"), Validity);
    auto Frame = First(First(Root, TEXT("hands")), TEXT("frames"));
    Frame->SetStringField(TEXT("sample_kind"), TEXT("detected_model_observation"));
    auto PoseValidity = MakeShared<FJsonObject>(); PoseValidity->SetBoolField(TEXT("pose"), true);
    Frame->SetObjectField(TEXT("validity"), PoseValidity);
    return Root;
}

FObject Fixture13Local(const FCardCapBoneMapping& Mapping)
{
    auto Root = Fixture12(Mapping);
    Root->SetStringField(TEXT("format_version"), TEXT("1.3"));
    Child(Root, TEXT("camera"))->SetField(TEXT("intrinsics"), MakeShared<FJsonValueNull>());
    Child(Root, TEXT("provenance"))->SetStringField(TEXT("camera_intrinsics"), TEXT("unobservable"));
    Child(Root, TEXT("provenance"))->SetStringField(TEXT("coordinate_frame"), TEXT("per_hand_wrist_local"));
    Child(Root, TEXT("validity"))->SetBoolField(TEXT("camera_intrinsics"), false);
    Child(Root, TEXT("validity"))->SetBoolField(TEXT("inter_hand_transform"), false);
    auto Frame = First(First(Root, TEXT("hands")), TEXT("frames"));
    Frame->SetField(TEXT("global_trans_cm"), MakeShared<FJsonValueNull>());
    Child(Frame, TEXT("validity"))->SetBoolField(TEXT("global_translation"), false);
    return Root;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapCoordinateAxesTest, "CardistryCapture.M2.Coordinates.AxesAndUnits", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapCoordinateAxesTest::RunTest(const FString& Parameters)
{
    const FVector X(1,0,0), Y(0,1,0), Z(0,0,1);
    TestEqual(TEXT("spec X -> UE Y centimeters"), FCardCapCoordConvert::SpecManoPositionToUE(X), FVector(0,100,0));
    TestEqual(TEXT("spec Y -> UE Z"), FCardCapCoordConvert::SpecManoPositionToUE(Y), FVector(0,0,100));
    TestEqual(TEXT("spec Z -> UE X"), FCardCapCoordConvert::SpecManoPositionToUE(Z), FVector(100,0,0));
    TestEqual(TEXT("camera Y-down -> negative UE Z"), FCardCapCoordConvert::CameraPositionToUE(Y), FVector(0,0,-100));
    TestEqual(TEXT("camera Z-forward -> UE X"), FCardCapCoordConvert::CameraPositionToUE(Z), FVector(100,0,0));
    const FVector P(-0.25, 0.037, 1.92);
    TestEqual(TEXT("spec position inverse"), FCardCapCoordConvert::UEPositionToSpecMano(FCardCapCoordConvert::SpecManoPositionToUE(P)), P, 1e-9f);
    TestEqual(TEXT("camera position inverse"), FCardCapCoordConvert::UEPositionToCamera(FCardCapCoordConvert::CameraPositionToUE(P)), P, 1e-9f);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapCoordinateRotationTest, "CardistryCapture.M2.Coordinates.RotationBasisAndReflection", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapCoordinateRotationTest::RunTest(const FString& Parameters)
{
    FQuat Q(0.23, -0.41, 0.17, 0.86); Q.Normalize();
    const FQuat Spec = FCardCapCoordConvert::SpecManoRotationToUE(Q);
    const FQuat Camera = FCardCapCoordConvert::CameraRotationToUE(Q);
    for (const FVector V : {FVector(1,0,0), FVector(0,1,0), FVector(0,0,1), FVector(0.3,-0.2,0.9)})
    {
        TestEqual(TEXT("spec C R = R_UE C"), FCardCapCoordConvert::SpecManoPositionToUE(Q.RotateVector(V)), Spec.RotateVector(FCardCapCoordConvert::SpecManoPositionToUE(V)), 1e-8f);
        TestEqual(TEXT("camera reflected C R = R_UE C"), FCardCapCoordConvert::CameraPositionToUE(Q.RotateVector(V)), Camera.RotateVector(FCardCapCoordConvert::CameraPositionToUE(V)), 1e-8f);
    }
    TestTrue(TEXT("spec quaternion inverse"), FCardCapCoordConvert::UERotationToSpecMano(Spec).Equals(Q, 1e-12));
    TestTrue(TEXT("camera quaternion inverse"), FCardCapCoordConvert::UERotationToCamera(Camera).Equals(Q, 1e-12));
    const double H = FMath::Sqrt(0.5);
    const FQuat AroundCameraY(0,H,0,H);
    const FQuat Correct = FCardCapCoordConvert::CameraRotationToUE(AroundCameraY);
    const FQuat WrongPositionOnly(0,0,-H,H);
    const FVector V(100,0,0);
    TestFalse(TEXT("reflection cannot permute quaternion like a position"), Correct.RotateVector(V).Equals(WrongPositionOnly.RotateVector(V), 1e-8));
    TestEqual(TEXT("basis transform preserves unit norm"), Camera.SizeSquared(), 1.0, 1e-12);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParserValidTest, "CardistryCapture.M2.Parser.CompleteContract", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParserValidTest::RunTest(const FString& Parameters)
{
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load actual mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto Root = CardCapTest::Fixture(Mapping); FCardCapCapture Capture;
    if (!TestTrue(TEXT("minimal complete document with empty packets/events"), FCardCapJsonParser::ParseString(CardCapTest::Serialize(Root), Capture, Error))) { AddError(Error); return false; }
    TestEqual(TEXT("fps retained"), Capture.Meta.Fps, 30.0);
    TestEqual(TEXT("exact 21 joints"), Capture.Hands[0].Frames[0].JointPositionsCm.Num(), 21);
    TestEqual(TEXT("configured 15 bones"), Capture.Hands[0].Frames[0].BoneRotations.Num(), 15);
    TestEqual(TEXT("UE position not converted twice"), Capture.Hands[0].Frames[0].GlobalTransCm, FVector(10,20,30));
    TestFalse(TEXT("explicit unmeasured quality remains unset"), Capture.Quality.MeanReprojectionErrorPx.IsSet());
    // Exercise every packet/event object branch; these values remain unit fixture data.
    Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(CardCapTest::Packet(TEXT("a"))), MakeShared<FJsonValueObject>(CardCapTest::Packet(TEXT("b"))), MakeShared<FJsonValueObject>(CardCapTest::Packet(TEXT("c")))});
    CardCapTest::FObject Events;
    FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(FString(TEXT(R"JSON({"splits":[{"frame":1,"source":"a","results":["b","c"]}],"merges":[{"frame":2,"sources":["b","c"],"result":"a"}],"releases":[{"frame":1,"packet":"b","release_velocity_cm_s":[0,120,340]}]})JSON"))), Events);
    Root->SetObjectField(TEXT("events"), Events);
    CardCapTest::Child(Root, TEXT("quality"))->SetNumberField(TEXT("mean_reprojection_error_px"), 1.2);
    if (!TestTrue(TEXT("complete nonempty packets/events"), FCardCapJsonParser::ParseString(CardCapTest::Serialize(Root), Capture, Error))) { AddError(Error); return false; }
    TestEqual(TEXT("release velocity retained"), Capture.Events.Releases[0].ReleaseVelocityCmS, FVector(0,120,340));
    TestEqual(TEXT("measured quality retained"), Capture.Quality.MeanReprojectionErrorPx.GetValue(), 1.2);
    // Sparse frames are legal and remain sparse for the baker to handle explicitly.
    auto Hand = CardCapTest::First(Root, TEXT("hands")); Hand->SetArrayField(TEXT("frames"), {});
    TestTrue(TEXT("missing hand samples can be represented by a sparse sequence"), FCardCapJsonParser::ParseString(CardCapTest::Serialize(Root), Capture, Error));
    TestEqual(TEXT("no implicit fill-forward"), Capture.Hands[0].Frames.Num(), 0);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParserRejectTest, "CardistryCapture.M2.Parser.RejectMalformed", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParserRejectTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Valid = Fixture(Mapping);
    auto Reject = [this](const FObject& Root, const FString& ExpectedPath)
    {
        FCardCapCapture Capture; Capture.Meta.Fps = 123.0; FString E;
        TestFalse(*ExpectedPath, FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
        TestTrue(TEXT("diagnostic identifies offending field"), E.Contains(ExpectedPath));
        TestEqual(TEXT("failure leaves prior capture untouched"), Capture.Meta.Fps, 123.0);
    };
    for (const TCHAR* Key : {TEXT("format_version"),TEXT("meta"),TEXT("camera"),TEXT("scale"),TEXT("hands"),TEXT("packets"),TEXT("events"),TEXT("quality")})
    { auto O = Clone(Valid); O->RemoveField(Key); Reject(O, FString(TEXT("$.")) + Key); }
    { auto O = Clone(Valid); O->SetStringField(TEXT("format_version"), TEXT("2.0")); Reject(O, TEXT("$.format_version")); }
    { auto O = Clone(Valid); Child(O,TEXT("meta"))->SetStringField(TEXT("fps"), TEXT("30")); Reject(O,TEXT("$.meta.fps")); }
    { auto O = Clone(Valid); Child(O,TEXT("meta"))->SetNumberField(TEXT("frame_count"), 1.5); Reject(O,TEXT("$.meta.frame_count")); }
    { auto O = Clone(Valid); Child(O,TEXT("meta"))->SetNumberField(TEXT("fps"), 0); Reject(O,TEXT("$.meta.fps")); }
    { auto O = Clone(Valid); First(O,TEXT("hands"))->SetArrayField(TEXT("mano_shape"), Numbers({0,0})); Reject(O,TEXT("$.hands[0].mano_shape")); }
    { auto O = Clone(Valid); auto H = First(O,TEXT("hands")); O->SetArrayField(TEXT("hands"), {MakeShared<FJsonValueObject>(H),MakeShared<FJsonValueObject>(H)}); Reject(O,TEXT("$.hands[1].side")); }
    for (const TCHAR* Key : {TEXT("frame"),TEXT("confidence"),TEXT("global_trans_cm"),TEXT("global_rot_quat"),TEXT("joint_positions_cm"),TEXT("bone_rotations"),TEXT("occluded_joints")})
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->RemoveField(Key); Reject(O,FString(TEXT("$.hands[0].frames[0].")) + Key); }
    { auto O = Clone(Valid); auto H = First(O,TEXT("hands")); auto F = First(H,TEXT("frames")); H->SetArrayField(TEXT("frames"), {MakeShared<FJsonValueObject>(F),MakeShared<FJsonValueObject>(F)}); Reject(O,TEXT("$.hands[0].frames[1].frame")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetNumberField(TEXT("frame"), 3); Reject(O,TEXT(".frame")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetNumberField(TEXT("confidence"), 1.1); Reject(O,TEXT(".confidence")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetArrayField(TEXT("global_rot_quat"), Numbers({0,0,0,0})); Reject(O,TEXT(".global_rot_quat")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetArrayField(TEXT("global_rot_quat"), Numbers({0,0,0,2})); Reject(O,TEXT(".global_rot_quat")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetArrayField(TEXT("joint_positions_cm"), {}); Reject(O,TEXT(".joint_positions_cm")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetArrayField(TEXT("occluded_joints"), Numbers({21})); Reject(O,TEXT(".occluded_joints[0]")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetArrayField(TEXT("occluded_joints"), Numbers({8,8})); Reject(O,TEXT(".occluded_joints[1]")); }
    { auto O = Clone(Valid); auto B = Child(First(First(O,TEXT("hands")),TEXT("frames")),TEXT("bone_rotations")); B->RemoveField(Mapping.Hands.FindChecked(TEXT("right")).BoneNames[1].ToString()); B->SetArrayField(TEXT("invalid_bone"),Numbers({0,0,0,1})); Reject(O,TEXT(".bone_rotations.invalid_bone")); }
    { auto O = Clone(Valid); Child(O,TEXT("quality"))->RemoveField(TEXT("mean_reprojection_error_px")); Reject(O,TEXT("$.quality.mean_reprojection_error_px")); }
    { auto O = Clone(Valid); Child(O,TEXT("quality"))->SetNumberField(TEXT("mean_reprojection_error_px"),-1); Reject(O,TEXT("$.quality.mean_reprojection_error_px")); }
    { auto O = Clone(Valid); Child(O,TEXT("quality"))->SetArrayField(TEXT("low_confidence_ranges"), {MakeShared<FJsonValueArray>(Numbers({2,1}))}); Reject(O,TEXT(".low_confidence_ranges[0]")); }
    { auto O = Clone(Valid); Child(O,TEXT("events"))->RemoveField(TEXT("releases")); Reject(O,TEXT("$.events.releases")); }
    { auto O = Clone(Valid); auto P = Packet(TEXT("a")); First(P,TEXT("frames"))->RemoveField(TEXT("linear_velocity_cm_s")); O->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)}); Reject(O,TEXT("$.packets[0].frames[0].linear_velocity_cm_s")); }
    { auto O = Clone(Valid); auto P = Packet(TEXT("a")); First(P,TEXT("frames"))->SetNumberField(TEXT("frame"),3); O->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)}); Reject(O,TEXT("$.packets[0].frames[0].frame")); }
    { auto O = Clone(Valid); auto P = Packet(TEXT("a")); First(P,TEXT("frames"))->SetStringField(TEXT("contact_state"),TEXT("BOGUS")); O->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)}); Reject(O,TEXT(".contact_state")); }
    FCardCapCapture Capture;
    TestFalse(TEXT("duplicate JSON object keys are rejected before map overwrite"), FCardCapJsonParser::ParseString(TEXT("{\"format_version\":\"1.0\",\"format_version\":\"1.0\"}"), Capture, Error));
    TestTrue(TEXT("duplicate key has location"), Error.Contains(TEXT("duplicate object key")) && Error.Contains(TEXT("line")));
    const FString ValidText = Serialize(Valid);
    FString Nonfinite = ValidText.Replace(TEXT("\"fps\":30"), TEXT("\"fps\":1e9999")).Replace(TEXT("\"fps\": 30"), TEXT("\"fps\": 1e9999"));
    TestTrue(TEXT("overflow fixture actually changes the fps token"), Nonfinite != ValidText);
    TestFalse(TEXT("overflow JSON number is rejected"), FCardCapJsonParser::ParseString(Nonfinite, Capture, Error));
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser11ValuesTest, "CardistryCapture.M3.Parser11.NullableValues", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser11ValuesTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto Root = Fixture(Mapping); Root->SetStringField(TEXT("format_version"), TEXT("1.1"));
    Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(UnknownPacket())});
    FCardCapCapture Capture;
    if (!TestTrue(TEXT("1.1 explicit unknown packet properties"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error))) { AddError(Error); return false; }
    const auto& P = Capture.Packets[0]; const auto& F = P.Frames[0];
    TestFalse(TEXT("unknown count is unset"), P.CardCountEstimate.IsSet());
    TestFalse(TEXT("unknown count confidence is unset"), P.CardCountConfidence.IsSet());
    TestEqual(TEXT("known width retained"), P.DimensionsCm.Width, 6.35);
    TestEqual(TEXT("known height retained"), P.DimensionsCm.Height, 8.89);
    TestFalse(TEXT("unknown thickness is unset"), P.DimensionsCm.Thickness.IsSet());
    TestFalse(TEXT("unknown linear velocity is unset"), F.LinearVelocityCmS.IsSet());
    TestFalse(TEXT("unknown angular velocity is unset"), F.AngularVelocityRadS.IsSet());
    TestFalse(TEXT("unmeasured reprojection is unset"), F.ReprojectionErrorPx.IsSet());
    TestFalse(TEXT("unassessed contact bones are unset"), F.ContactBones.IsSet());
    TestTrue(TEXT("unknown contact is not free"), F.ContactState == ECardCapContactState::Unknown);
    TestEqual(TEXT("packet holes are not filled"), P.Frames.Num(), 1);
    First(Root, TEXT("packets"))->SetArrayField(TEXT("frames"), {});
    if (!TestTrue(TEXT("empty packet pose sequence remains legal"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error))) return false;
    TestEqual(TEXT("no invented packet poses"), Capture.Packets[0].Frames.Num(), 0);
    // Both versions preserve genuine zeros and empty contact lists as set values.
    for (const TCHAR* Version : {TEXT("1.0"), TEXT("1.1")})
    {
        Root->SetStringField(TEXT("format_version"), Version);
        auto Known = Packet(TEXT("known-unit-fixture"));
        Known->SetNumberField(TEXT("card_count_confidence"), 0);
        First(Known, TEXT("frames"))->SetNumberField(TEXT("reprojection_error_px"), 0);
        Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(Known)});
        if (!TestTrue(TEXT("fully specified packet supported in both versions"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error))) { AddError(Error); return false; }
        const auto& K = Capture.Packets[0]; const auto& KF = K.Frames[0];
        if (!TestTrue(TEXT("known fields remain set"), K.CardCountEstimate.IsSet() && K.CardCountConfidence.IsSet() && K.DimensionsCm.Thickness.IsSet()
            && KF.LinearVelocityCmS.IsSet() && KF.AngularVelocityRadS.IsSet() && KF.ReprojectionErrorPx.IsSet() && KF.ContactBones.IsSet())) return false;
        TestEqual(TEXT("known count retained"), K.CardCountEstimate.GetValue(), 1);
        TestEqual(TEXT("zero count confidence is not null"), K.CardCountConfidence.GetValue(), 0.0);
        TestEqual(TEXT("known thickness retained"), K.DimensionsCm.Thickness.GetValue(), 0.03);
        TestEqual(TEXT("zero linear velocity is not unknown"), KF.LinearVelocityCmS.GetValue(), FVector::ZeroVector);
        TestEqual(TEXT("zero angular velocity is not unknown"), KF.AngularVelocityRadS.GetValue(), FVector::ZeroVector);
        TestEqual(TEXT("zero reprojection is not unknown"), KF.ReprojectionErrorPx.GetValue(), 0.0);
        TestEqual(TEXT("known empty contact list retained"), KF.ContactBones.GetValue().Num(), 0);
        TestTrue(TEXT("explicit free state retained"), KF.ContactState == ECardCapContactState::Free);
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser11VersionTest, "CardistryCapture.M3.Parser11.LegacyBoundary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser11VersionTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto CheckBoundary = [this, &Mapping](const FObject& PacketObject, const FString& Path)
    {
        auto Root = Fixture(Mapping); Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(PacketObject)});
        FCardCapCapture Capture; Capture.Meta.Fps = 123; FString E;
        TestFalse(*Path, FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
        TestTrue(TEXT("1.0 rejection identifies field"), E.Contains(Path));
        TestEqual(TEXT("1.0 failure preserves output"), Capture.Meta.Fps, 123.0);
        Root->SetStringField(TEXT("format_version"), TEXT("1.1"));
        if (!TestTrue(TEXT("same explicit unknown is accepted only in 1.1"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, E))) AddError(E);
    };
    { auto P = Packet(TEXT("a")); P->SetField(TEXT("card_count_estimate"), MakeShared<FJsonValueNull>()); P->SetField(TEXT("card_count_confidence"), MakeShared<FJsonValueNull>()); CheckBoundary(P, TEXT(".card_count_estimate")); }
    { auto P = Packet(TEXT("a")); auto D = Numbers({6.35, 8.89}); D.Add(MakeShared<FJsonValueNull>()); P->SetArrayField(TEXT("dimensions_cm"), D); CheckBoundary(P, TEXT(".dimensions_cm[2]")); }
    for (const TCHAR* Key : {TEXT("linear_velocity_cm_s"), TEXT("angular_velocity_rad_s"), TEXT("reprojection_error_px"), TEXT("contact_bones")})
    {
        auto P = Packet(TEXT("a")); First(P, TEXT("frames"))->SetField(Key, MakeShared<FJsonValueNull>());
        CheckBoundary(P, FString(TEXT(".")) + Key);
    }
    { auto P = Packet(TEXT("a")); First(P, TEXT("frames"))->SetStringField(TEXT("contact_state"), TEXT("UNKNOWN")); CheckBoundary(P, TEXT(".contact_state")); }
    // Existing FString == semantics accepted case variants before 1.1. Preserve
    // those four states while requiring exact spelling only for new UNKNOWN.
    struct FLegacyState { const TCHAR* Text; ECardCapContactState Expected; };
    const FLegacyState LegacyStates[] = {
        {TEXT("gripped"), ECardCapContactState::Gripped}, {TEXT("FrEe"), ECardCapContactState::Free},
        {TEXT("resting"), ECardCapContactState::Resting}, {TEXT("sLiDiNg"), ECardCapContactState::Sliding}};
    for (const TCHAR* Version : {TEXT("1.0"), TEXT("1.1")})
    {
        for (const auto& State : LegacyStates)
        {
            auto Root = Fixture(Mapping); Root->SetStringField(TEXT("format_version"), Version);
            auto P = Packet(TEXT("a")); First(P, TEXT("frames"))->SetStringField(TEXT("contact_state"), State.Text);
            Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)});
            FCardCapCapture Capture;
            if (!TestTrue(TEXT("legacy state case variants remain accepted"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error))) { AddError(Error); return false; }
            TestTrue(TEXT("legacy case variant maps to the same state"), Capture.Packets[0].Frames[0].ContactState == State.Expected);
        }
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser11RejectTest, "CardistryCapture.M3.Parser11.RejectMalformed", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser11RejectTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto Reject = [this, &Mapping](const FObject& PacketObject, const FString& Path)
    {
        auto Root = Fixture(Mapping); Root->SetStringField(TEXT("format_version"), TEXT("1.1"));
        Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(PacketObject)});
        FCardCapCapture Capture; Capture.Meta.Fps = 123; FString E;
        TestFalse(*Path, FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
        TestTrue(TEXT("1.1 rejection identifies field"), E.Contains(Path));
        TestEqual(TEXT("1.1 failure preserves output"), Capture.Meta.Fps, 123.0);
    };
    for (const TCHAR* Key : {TEXT("card_count_estimate"), TEXT("card_count_confidence"), TEXT("dimensions_cm")})
    { auto P = UnknownPacket(); P->RemoveField(Key); Reject(P, FString(TEXT(".")) + Key); }
    for (const TCHAR* Key : {TEXT("linear_velocity_cm_s"), TEXT("angular_velocity_rad_s"), TEXT("reprojection_error_px"), TEXT("contact_state"), TEXT("contact_bones")})
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->RemoveField(Key); Reject(P, FString(TEXT(".")) + Key); }
    // Count and its confidence cannot disagree about whether an estimate exists.
    { auto P = UnknownPacket(); P->SetNumberField(TEXT("card_count_estimate"), 1); Reject(P, TEXT(".card_count_confidence")); }
    { auto P = UnknownPacket(); P->SetNumberField(TEXT("card_count_confidence"), 0); Reject(P, TEXT(".card_count_confidence")); }
    for (double Bad : {0.0, 1.5})
    { auto P = Packet(TEXT("a")); P->SetNumberField(TEXT("card_count_estimate"), Bad); Reject(P, TEXT(".card_count_estimate")); }
    { auto P = Packet(TEXT("a")); P->SetNumberField(TEXT("card_count_confidence"), 1.1); Reject(P, TEXT(".card_count_confidence")); }
    { auto P = Packet(TEXT("a")); P->SetBoolField(TEXT("card_count_estimate"), true); Reject(P, TEXT(".card_count_estimate")); }
    for (int32 Axis = 0; Axis < 3; ++Axis)
    {
        auto P = Packet(TEXT("a")); auto D = Numbers({6.35, 8.89, .03}); D[Axis] = MakeShared<FJsonValueNumber>(0);
        P->SetArrayField(TEXT("dimensions_cm"), D); Reject(P, TEXT(".dimensions_cm"));
        if (Axis < 2) { D[Axis] = MakeShared<FJsonValueNull>(); P->SetArrayField(TEXT("dimensions_cm"), D); Reject(P, TEXT(".dimensions_cm")); }
    }
    { auto P = UnknownPacket(); P->SetField(TEXT("dimensions_cm"), MakeShared<FJsonValueNull>()); Reject(P, TEXT(".dimensions_cm")); }
    for (const TCHAR* Key : {TEXT("linear_velocity_cm_s"), TEXT("angular_velocity_rad_s")})
    {
        auto P = UnknownPacket(); auto F = First(P, TEXT("frames")); auto V = Numbers({0, 0}); V.Add(MakeShared<FJsonValueNull>());
        F->SetArrayField(Key, V); Reject(P, FString(TEXT(".")) + Key);
        F->SetArrayField(Key, Numbers({0, 0})); Reject(P, FString(TEXT(".")) + Key);
        F->SetStringField(Key, TEXT("unknown")); Reject(P, FString(TEXT(".")) + Key);
    }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetNumberField(TEXT("reprojection_error_px"), -1); Reject(P, TEXT(".reprojection_error_px")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetBoolField(TEXT("reprojection_error_px"), false); Reject(P, TEXT(".reprojection_error_px")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetField(TEXT("contact_state"), MakeShared<FJsonValueNull>()); Reject(P, TEXT(".contact_state")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetStringField(TEXT("contact_state"), TEXT("unknown")); Reject(P, TEXT(".contact_state")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetStringField(TEXT("contact_state"), TEXT("Unknown")); Reject(P, TEXT(".contact_state")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetArrayField(TEXT("contact_bones"), {MakeShared<FJsonValueString>(TEXT("not_a_mapped_bone"))}); Reject(P, TEXT(".contact_bones[0]")); }
    { auto P = UnknownPacket(); auto N = MakeShared<FJsonValueString>(Mapping.Hands.FindChecked(TEXT("right")).BoneNames[0].ToString()); First(P, TEXT("frames"))->SetArrayField(TEXT("contact_bones"), {N, N}); Reject(P, TEXT(".contact_bones[1]")); }
    // Nullability of properties never permits an invented/partial pose.
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetField(TEXT("position_cm"), MakeShared<FJsonValueNull>()); Reject(P, TEXT(".position_cm")); }
    { auto P = UnknownPacket(); First(P, TEXT("frames"))->SetArrayField(TEXT("rotation_quat"), Numbers({0, 0, 0, 0})); Reject(P, TEXT(".rotation_quat")); }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser12ValuesTest, "CardistryCapture.P0.Parser12.NullableProvenanceAndMasks", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser12ValuesTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto Root = Fixture12(Mapping); FCardCapCapture Capture;
    auto Parse = [this, &Root, &Capture, &Error]()
    {
        const bool Ok = FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error);
        if (!TestTrue(TEXT("1.2 contract parses"), Ok)) AddError(Error);
        return Ok;
    };
    if (!Parse()) return false;
    TestTrue(TEXT("conditional numeric K is available, not calibrated"), Capture.Camera.bHasIntrinsics && !Capture.Camera.bCalibrated);
    TestFalse(TEXT("unknown D stays unset"), Capture.Camera.bHasDistortion);
    TestFalse(TEXT("unknown meters per unit stays unset"), Capture.Scale.bHasMetersPerUnit);
    TestFalse(TEXT("unknown confidence stays unset"), Capture.Scale.bHasConfidence);
    TestFalse(TEXT("unknown anchor stays unset"), Capture.Scale.bHasAnchorMethod);
    TestTrue(TEXT("conditional K is inferred"), Capture.Provenance.CameraIntrinsics == ECardCapProvenance::Inferred);
    TestEqual(TEXT("conditional units retained"), Capture.Provenance.CoordinateUnits, FString(TEXT("conditional_ue_units")));
    TestEqual(TEXT("stored display coordinate is neither scaled nor aligned"), Capture.Hands[0].Frames[0].GlobalTransCm, FVector(10,20,30));
    auto F = First(First(Root, TEXT("hands")), TEXT("frames"));
    const TCHAR* Kinds[] = {TEXT("detected_model_observation"), TEXT("tracked_roi_model_observation"), TEXT("interpolated"), TEXT("endpoint_hold"), TEXT("missing")};
    const ECardCapSampleKind Expected[] = {ECardCapSampleKind::DetectedModelObservation, ECardCapSampleKind::TrackedRoiModelObservation, ECardCapSampleKind::Interpolated, ECardCapSampleKind::EndpointHold, ECardCapSampleKind::Missing};
    for (int32 I = 0; I < UE_ARRAY_COUNT(Kinds); ++I)
    {
        F->SetStringField(TEXT("sample_kind"), Kinds[I]); Child(F, TEXT("validity"))->SetBoolField(TEXT("pose"), I != 4);
        if (!Parse()) return false;
        TestTrue(TEXT("sample kind preserved without claiming an observation"), Capture.Hands[0].Frames[0].SampleKind == Expected[I]);
        TestEqual(TEXT("pose availability preserved"), Capture.Hands[0].Frames[0].bPoseValid, I != 4);
    }
    auto P = UnknownPacket(); P->SetField(TEXT("dimensions_cm"), MakeShared<FJsonValueNull>());
    Root->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)});
    auto Release = MakeShared<FJsonObject>(); Release->SetNumberField(TEXT("frame"), 1);
    Release->SetStringField(TEXT("packet"), TEXT("unknown-properties-unit-fixture"));
    Release->SetField(TEXT("release_velocity_cm_s"), MakeShared<FJsonValueNull>());
    Child(Root, TEXT("events"))->SetArrayField(TEXT("releases"), {MakeShared<FJsonValueObject>(Release)});
    if (!Parse()) return false;
    TestFalse(TEXT("null width unavailable"), Capture.Packets[0].DimensionsCm.bHasWidth);
    TestFalse(TEXT("null height unavailable"), Capture.Packets[0].DimensionsCm.bHasHeight);
    TestFalse(TEXT("null release velocity is not a zero velocity"), Capture.Events.Releases[0].bHasReleaseVelocity);
    P->SetArrayField(TEXT("dimensions_cm"), {MakeShared<FJsonValueNull>(), MakeShared<FJsonValueNumber>(8.89), MakeShared<FJsonValueNull>()});
    Release->SetArrayField(TEXT("release_velocity_cm_s"), Numbers({0,0,0}));
    if (!Parse()) return false;
    TestTrue(TEXT("partial dimensions preserve known height only"), !Capture.Packets[0].DimensionsCm.bHasWidth && Capture.Packets[0].DimensionsCm.bHasHeight);
    TestEqual(TEXT("known height retained"), Capture.Packets[0].DimensionsCm.Height, 8.89);
    TestTrue(TEXT("genuine zero release remains supplied"), Capture.Events.Releases[0].bHasReleaseVelocity);
    TestEqual(TEXT("genuine zero release retained"), Capture.Events.Releases[0].ReleaseVelocityCmS, FVector::ZeroVector);
    Child(Root, TEXT("camera"))->SetField(TEXT("intrinsics"), MakeShared<FJsonValueNull>());
    Child(Root, TEXT("validity"))->SetBoolField(TEXT("camera_intrinsics"), false);
    Child(Root, TEXT("provenance"))->SetStringField(TEXT("camera_intrinsics"), TEXT("unobservable"));
    if (!Parse()) return false;
    TestFalse(TEXT("unknown K stays unavailable"), Capture.Camera.bHasIntrinsics);
    auto Scale = Child(Root, TEXT("scale")); Scale->SetNumberField(TEXT("meters_per_unit"), 1);
    Scale->SetStringField(TEXT("anchor_method"), TEXT("user_measured"));
    Child(Root, TEXT("validity"))->SetBoolField(TEXT("metric_scale"), true);
    Child(Root, TEXT("provenance"))->SetStringField(TEXT("metric_scale"), TEXT("user_measured"));
    Child(Root, TEXT("provenance"))->SetStringField(TEXT("coordinate_units"), TEXT("ue_centimeters"));
    if (!Parse()) return false;
    TestTrue(TEXT("explicit measurement can supply scale without fabricating numeric probability"), Capture.Scale.bHasMetersPerUnit && Capture.Scale.bHasAnchorMethod && !Capture.Scale.bHasConfidence);
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser12RejectTest, "CardistryCapture.P0.Parser12.RejectContradictoryClaims", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser12RejectTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Valid = Fixture12(Mapping);
    auto Reject = [this](const FObject& Root, const FString& Expected)
    {
        FCardCapCapture Capture; Capture.Meta.Fps = 123; FString E;
        TestFalse(*Expected, FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
        TestTrue(TEXT("error identifies contradictory or malformed claim"), E.Contains(Expected));
        TestEqual(TEXT("failed 1.2 parse is transactional"), Capture.Meta.Fps, 123.0);
    };
    for (const TCHAR* Key : {TEXT("provenance"), TEXT("validity")})
    { auto O = Clone(Valid); O->RemoveField(Key); Reject(O, FString(TEXT("$.")) + Key); }
    for (const TCHAR* Key : {TEXT("camera_intrinsics"), TEXT("camera_distortion"), TEXT("metric_scale"), TEXT("hand_geometry"), TEXT("coordinate_units"), TEXT("assumptions")})
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->RemoveField(Key); Reject(O,FString(TEXT("$.provenance.")) + Key); }
    for (const TCHAR* Key : {TEXT("camera_intrinsics"), TEXT("camera_distortion"), TEXT("metric_scale"), TEXT("hand_geometry")})
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetStringField(Key, TEXT("Inferred")); Reject(O,FString(TEXT("$.provenance.")) + Key); }
    for (const TCHAR* Key : {TEXT("camera_intrinsics"), TEXT("camera_distortion"), TEXT("metric_scale")})
    {
        auto O = Clone(Valid); Child(O,TEXT("validity"))->SetBoolField(Key, FString(Key) != TEXT("camera_intrinsics")); Reject(O,FString(TEXT("$.validity.")) + Key);
        O = Clone(Valid); Child(O,TEXT("validity"))->SetNumberField(Key, 1); Reject(O,FString(TEXT("$.validity.")) + Key);
    }
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetStringField(TEXT("camera_intrinsics"), TEXT("unobservable")); Reject(O,TEXT("$.provenance.camera_intrinsics")); }
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetStringField(TEXT("coordinate_units"), TEXT("ue_centimeters")); Reject(O,TEXT("$.provenance.coordinate_units")); }
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetArrayField(TEXT("assumptions"), {}); Reject(O,TEXT("$.provenance.assumptions")); }
    { auto O = Clone(Valid); Child(O,TEXT("camera"))->SetBoolField(TEXT("calibrated"), true); Reject(O,TEXT("$.camera.calibrated")); }
    { auto O = Clone(Valid); Child(O,TEXT("scale"))->SetNumberField(TEXT("confidence"), 0); Reject(O,TEXT("$.scale")); }
    { auto O = Clone(Valid); Child(O,TEXT("scale"))->SetArrayField(TEXT("anchor_frames"), Numbers({0})); Reject(O,TEXT("$.scale")); }
    { auto O = Clone(Valid); Child(Child(O,TEXT("camera")),TEXT("intrinsics"))->SetField(TEXT("fx"), MakeShared<FJsonValueNull>()); Reject(O,TEXT("$.camera.intrinsics.fx")); }
    for (const TCHAR* Key : {TEXT("sample_kind"), TEXT("validity")})
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->RemoveField(Key); Reject(O,FString(TEXT("$.hands[0].frames[0].")) + Key); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetStringField(TEXT("sample_kind"), TEXT("detected")); Reject(O,TEXT(".sample_kind")); }
    { auto O = Clone(Valid); First(First(O,TEXT("hands")),TEXT("frames"))->SetStringField(TEXT("sample_kind"), TEXT("missing")); Reject(O,TEXT(".validity.pose")); }
    { auto O = Clone(Valid); Child(First(First(O,TEXT("hands")),TEXT("frames")),TEXT("validity"))->SetBoolField(TEXT("pose"), false); Reject(O,TEXT(".validity.pose")); }
    { auto O = Clone(Valid); auto P = UnknownPacket(); P->SetArrayField(TEXT("dimensions_cm"), {MakeShared<FJsonValueNull>(), MakeShared<FJsonValueNumber>(0), MakeShared<FJsonValueNull>()}); O->SetArrayField(TEXT("packets"), {MakeShared<FJsonValueObject>(P)}); Reject(O,TEXT(".dimensions_cm")); }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser12LegacyTest, "CardistryCapture.P0.Parser12.LegacyNullBoundary", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser12LegacyTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    for (const TCHAR* Version : {TEXT("1.0"), TEXT("1.1")})
    {
        auto Reject = [this, &Mapping, Version](const TCHAR* Object, const TCHAR* Key)
        {
            auto Root = Fixture(Mapping); Root->SetStringField(TEXT("format_version"), Version);
            Child(Root, Object)->SetField(Key, MakeShared<FJsonValueNull>());
            FCardCapCapture Capture; FString E;
            TestFalse(TEXT("1.2 null expansion is rejected in legacy format"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
            TestTrue(TEXT("legacy null rejection identifies field"), E.Contains(Key));
        };
        Reject(TEXT("camera"), TEXT("intrinsics")); Reject(TEXT("camera"), TEXT("distortion"));
        Reject(TEXT("scale"), TEXT("meters_per_unit")); Reject(TEXT("scale"), TEXT("confidence")); Reject(TEXT("scale"), TEXT("anchor_method"));
        auto Root = Fixture(Mapping); Root->SetStringField(TEXT("format_version"), Version);
        Child(Root, TEXT("scale"))->SetStringField(TEXT("anchor_method"), TEXT("user_measured"));
        FCardCapCapture Capture;
        TestFalse(TEXT("new anchor enum does not relax old versions"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error));
        auto Legacy = Fixture(Mapping); Legacy->SetStringField(TEXT("format_version"), Version);
        if (!TestTrue(TEXT("legacy files still require no 1.2 provenance or masks"), FCardCapJsonParser::ParseString(Serialize(Legacy), Capture, Error))) { AddError(Error); return false; }
        TestTrue(TEXT("legacy supplied values have true storage masks"), Capture.Camera.bHasIntrinsics && Capture.Camera.bHasDistortion && Capture.Scale.bHasMetersPerUnit && Capture.Scale.bHasAnchorMethod && Capture.Scale.bHasConfidence);
    }
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser13LocalTest, "CardistryCapture.P0.Parser13.LocalUnknownSpace", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser13LocalTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    auto Root = Fixture13Local(Mapping); FCardCapCapture Capture;
    if (!TestTrue(TEXT("local pose survives null source K and global position"), FCardCapJsonParser::ParseString(Serialize(Root), Capture, Error))) { AddError(Error); return false; }
    TestFalse(TEXT("no source K invented"), Capture.Camera.bHasIntrinsics);
    TestFalse(TEXT("no shared hand space inferred"), Capture.bInterHandTransformKnown);
    TestFalse(TEXT("null translation stays unavailable"), Capture.Hands[0].Frames[0].bHasGlobalTranslation);
    TestTrue(TEXT("local model pose remains available"), Capture.Hands[0].Frames[0].bPoseValid);
    TestFalse(TEXT("no full-image reprojection claimed"), Capture.Quality.MeanReprojectionErrorPx.IsSet());
    TestEqual(TEXT("coordinate frame retained"), Capture.Provenance.CoordinateFrame, FString(TEXT("per_hand_wrist_local")));
    auto Legacy = Clone(Root); Legacy->SetStringField(TEXT("format_version"), TEXT("1.2"));
    TestFalse(TEXT("1.3 global-translation null cannot be renamed into 1.2"), FCardCapJsonParser::ParseString(Serialize(Legacy), Capture, Error));
    TestTrue(TEXT("legacy boundary identifies global translation"), Error.Contains(TEXT(".global_trans_cm")));
    auto Shared = Fixture12(Mapping); Shared->SetStringField(TEXT("format_version"), TEXT("1.3"));
    Child(Shared, TEXT("provenance"))->SetStringField(TEXT("coordinate_frame"), TEXT("shared_camera"));
    Child(Shared, TEXT("validity"))->SetBoolField(TEXT("inter_hand_transform"), true);
    Child(First(First(Shared, TEXT("hands")), TEXT("frames")), TEXT("validity"))->SetBoolField(TEXT("global_translation"), true);
    if (!TestTrue(TEXT("explicit source camera preserves shared numeric translations"), FCardCapJsonParser::ParseString(Serialize(Shared), Capture, Error))) { AddError(Error); return false; }
    TestTrue(TEXT("supplied shared position remains available"), Capture.Hands[0].Frames[0].bHasGlobalTranslation && Capture.bInterHandTransformKnown);
    TestEqual(TEXT("shared translation not recentered"), Capture.Hands[0].Frames[0].GlobalTransCm, FVector(10,20,30));
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapParser13RejectTest, "CardistryCapture.P0.Parser13.RejectFalseSharedGeometry", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapParser13RejectTest::RunTest(const FString& Parameters)
{
    using namespace CardCapTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Valid = Fixture13Local(Mapping);
    auto Reject = [this](const FObject& Root, const FString& Expected)
    {
        FCardCapCapture Capture; Capture.Meta.Fps = 123; FString E;
        TestFalse(*Expected, FCardCapJsonParser::ParseString(Serialize(Root), Capture, E));
        TestTrue(TEXT("contradiction identifies its field"), E.Contains(Expected));
        TestEqual(TEXT("failed spatial parse is transactional"), Capture.Meta.Fps, 123.0);
    };
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->RemoveField(TEXT("coordinate_frame")); Reject(O,TEXT(".coordinate_frame")); }
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetStringField(TEXT("coordinate_frame"),TEXT("assumed_joint_space")); Reject(O,TEXT(".coordinate_frame")); }
    { auto O = Clone(Valid); Child(O,TEXT("validity"))->RemoveField(TEXT("inter_hand_transform")); Reject(O,TEXT(".inter_hand_transform")); }
    { auto O = Clone(Valid); Child(O,TEXT("validity"))->SetBoolField(TEXT("inter_hand_transform"),true); Reject(O,TEXT(".inter_hand_transform")); }
    { auto O = Clone(Valid); Child(O,TEXT("provenance"))->SetStringField(TEXT("coordinate_frame"),TEXT("shared_camera")); Child(O,TEXT("validity"))->SetBoolField(TEXT("inter_hand_transform"),true); Reject(O,TEXT(".camera.intrinsics")); }
    { auto O = Clone(Valid); auto F = First(First(O,TEXT("hands")),TEXT("frames")); Child(F,TEXT("validity"))->RemoveField(TEXT("global_translation")); Reject(O,TEXT(".validity.global_translation")); }
    { auto O = Clone(Valid); auto F = First(First(O,TEXT("hands")),TEXT("frames")); Child(F,TEXT("validity"))->SetBoolField(TEXT("global_translation"),true); Reject(O,TEXT(".validity.global_translation")); }
    { auto O = Clone(Valid); auto F = First(First(O,TEXT("hands")),TEXT("frames")); F->SetArrayField(TEXT("global_trans_cm"),Numbers({0,0,0})); Child(F,TEXT("validity"))->SetBoolField(TEXT("global_translation"),true); Reject(O,TEXT(".global_trans_cm")); }
    { auto O = Clone(Valid); Child(O,TEXT("quality"))->SetNumberField(TEXT("mean_reprojection_error_px"),0); Reject(O,TEXT(".mean_reprojection_error_px")); }
    { auto O = Clone(Valid); Child(O,TEXT("camera"))->SetArrayField(TEXT("distortion"),Numbers({0,0,0,0,0})); Child(O,TEXT("validity"))->SetBoolField(TEXT("camera_distortion"),true); Child(O,TEXT("provenance"))->SetStringField(TEXT("camera_distortion"),TEXT("calibrated")); Reject(O,TEXT(".camera.distortion")); }
    { auto O = Clone(Valid); O->SetArrayField(TEXT("packets"),{MakeShared<FJsonValueObject>(Packet(TEXT("a")))}); Reject(O,TEXT(".packets")); }
    { auto O = Clone(Valid); auto F = First(First(O,TEXT("hands")),TEXT("frames")); FArray Joints; for (int32 I = 0; I < 21; ++I) Joints.Add(MakeShared<FJsonValueArray>(Numbers({1,0,0}))); F->SetArrayField(TEXT("joint_positions_cm"),Joints); Reject(O,TEXT(".joint_positions_cm[0]")); }
    return true;
}
#endif
