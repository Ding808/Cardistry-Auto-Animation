#include "CardCapRetargeter.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "CardCapJsonParser.h"
#include "Misc/AutomationTest.h"
#include "ReferenceSkeleton.h"

namespace CardCapRetargetTest
{
FQuat Rotation(double X, double Y, double Z, double W)
{
    FQuat Q(X,Y,Z,W); Q.Normalize(); return Q;
}

// Deliberately nonidentity bind rotations and an unmapped metacarpal test actual
// ancestor composition. This is not footage or a demonstration asset.
FReferenceSkeleton MakeSkeleton(const FCardCapBoneMapping& Mapping, bool WrongHierarchy = false, bool NonunitScale = false)
{
    FReferenceSkeleton Skeleton;
    {
        FReferenceSkeletonModifier Modifier(Skeleton, nullptr);
        const FTransform RootPose(Rotation(0,0,0.2,0.98), FVector(3,4,5), FVector::OneVector);
        Modifier.Add(FMeshBoneInfo(Mapping.RootBone, Mapping.RootBone.ToString(), INDEX_NONE), RootPose);
        int32 Count = 1;
        for (const FString Side : {FString(TEXT("left")), FString(TEXT("right"))})
        {
            const auto& Names = Mapping.Hands.FindChecked(Side).BoneNames;
            TArray<int32> Indices; Indices.SetNum(16);
            Indices[0] = Count++;
            Modifier.Add(FMeshBoneInfo(Names[0], Names[0].ToString(), 0),
                FTransform(Rotation(0.1,0,0,0.99), FVector(10, Side == TEXT("left") ? -15 : 15, 7), FVector::OneVector));
            for (int32 J = 1; J < 16; ++J)
            {
                int32 Parent = Indices[Mapping.ManoParents[J]];
                if (J == 1)
                {
                    const FName Extra(*(TEXT("unit_metacarpal_") + Side));
                    Modifier.Add(FMeshBoneInfo(Extra, Extra.ToString(), Parent),
                        FTransform(Rotation(0,0.15,0,0.98), FVector(1,2,3), FVector::OneVector));
                    Parent = Count++;
                }
                if (WrongHierarchy && Side == TEXT("right") && J == 2) Parent = Indices[0];
                Indices[J] = Count++;
                Modifier.Add(FMeshBoneInfo(Names[J], Names[J].ToString(), Parent),
                    FTransform(Rotation(0,0,0.04*J,1), FVector(1+0.1*J,2,0.5),
                        NonunitScale && Side == TEXT("right") && J == 1 ? FVector(1,2,1) : FVector::OneVector));
            }
        }
    }
    return Skeleton;
}

TArray<FTransform> Components(const FReferenceSkeleton& Skeleton, const TArray<FTransform>& Local)
{
    TArray<FTransform> Out; Out.SetNum(Local.Num());
    for (int32 I = 0; I < Local.Num(); ++I)
    {
        const int32 Parent = Skeleton.GetParentIndex(I);
        Out[I] = Parent == INDEX_NONE ? Local[I] : Local[I] * Out[Parent];
    }
    return Out;
}

FCardCapCapture Capture(const FCardCapBoneMapping& Mapping, const FReferenceSkeleton& Skeleton)
{
    FCardCapCapture C; C.FormatVersion = TEXT("1.0"); C.BoneMapping = Mapping;
    C.Meta.Fps = 30; C.Meta.FrameCount = 2;
    const auto Ref = Components(Skeleton, Skeleton.GetRefBonePose());
    for (const FString Side : {FString(TEXT("left")), FString(TEXT("right"))})
    {
        const auto& Names = Mapping.Hands.FindChecked(Side).BoneNames;
        FCardCapHand H; H.Side = Side; H.ManoShape.Init(0,10);
        for (int32 F = 0; F < C.Meta.FrameCount; ++F)
        {
            FCardCapHandFrame Frame; Frame.Frame = F; Frame.Confidence = 0.5;
            Frame.JointPositionsCm.Init(FVector::ZeroVector,21);
            Frame.GlobalTransCm = F == 0 ? Ref[Skeleton.FindBoneIndex(Names[0])].GetTranslation() : FVector(100,200,300);
            Frame.GlobalRotQuat = F == 0 ? FQuat::Identity : Rotation(0.1,0.3,-0.15,0.8);
            for (int32 J = 1; J < 16; ++J) Frame.BoneRotations.Add(Names[J], FQuat::Identity);
            if (F == 1)
            {
                Frame.BoneRotations[Names[1]] = Rotation(0,0,0.3,0.9);
                Frame.BoneRotations[Names[2]] = Rotation(-0.2,0,0,0.8);
            }
            H.Frames.Add(MoveTemp(Frame));
        }
        C.Hands.Add(MoveTemp(H));
    }
    return C;
}
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapRetargetPoseTest, "CardistryCapture.M2.Retarget.ReferenceAndAnimatedParents", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapRetargetPoseTest::RunTest(const FString& Parameters)
{
    using namespace CardCapRetargetTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load actual mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Target = MakeSkeleton(Mapping);
    auto Input = Capture(Mapping, Target);
    FCardCapRetargetResult Result;
    if (!TestTrue(TEXT("retarget nonidentity reference plus extra metacarpals"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Result, Error))) { AddError(Error); return false; }
    TestEqual(TEXT("full frame count"), Result.Frames.Num(), 2);
    TestEqual(TEXT("every target bone has a local transform"), Result.Frames[1].LocalTransforms.Num(), Target.GetNum());
    const auto& RefLocal = Target.GetRefBonePose();
    const auto Ref = Components(Target, RefLocal);
    const auto Animated = Components(Target, Result.Frames[1].LocalTransforms);
    TSet<FName> Wrists;
    for (const auto& H : Input.Hands) Wrists.Add(Mapping.Hands.FindChecked(H.Side).BoneNames[0]);
    for (int32 B = 0; B < Target.GetNum(); ++B)
    {
        TestTrue(TEXT("identity source exactly recovers reference local pose"), Result.Frames[0].LocalTransforms[B].Equals(RefLocal[B],1e-8));
        TestEqual(TEXT("reference scale retained"), Result.Frames[1].LocalTransforms[B].GetScale3D(), RefLocal[B].GetScale3D(),1e-9f);
        if (!Wrists.Contains(Target.GetBoneName(B)))
        {
            TestEqual(TEXT("nonwrist local translation remains reference"), Result.Frames[1].LocalTransforms[B].GetTranslation(), RefLocal[B].GetTranslation(),1e-9f);
            const int32 Parent = Target.GetParentIndex(B);
            if (Parent != INDEX_NONE)
                TestEqual(TEXT("parent-child length remains reference"), (Animated[B].GetTranslation()-Animated[Parent].GetTranslation()).Length(), RefLocal[B].GetTranslation().Length(),1e-8);
        }
    }
    for (int32 H = 0; H < Input.Hands.Num(); ++H)
    {
        const auto& Names = Mapping.Hands.FindChecked(Input.Hands[H].Side).BoneNames;
        const auto& Frame = Input.Hands[H].Frames[1];
        TArray<FQuat> Accum; Accum.SetNum(16); Accum[0] = Frame.GlobalRotQuat;
        for (int32 J = 0; J < 16; ++J)
        {
            if (J > 0) Accum[J] = Accum[Mapping.ManoParents[J]] * Frame.BoneRotations.FindChecked(Names[J]);
            const int32 B = Target.FindBoneIndex(Names[J]);
            const FQuat Expected = Accum[J] * Ref[B].GetRotation();
            TestTrue(TEXT("component rotations include full source chain and target reference"), Animated[B].GetRotation().Equals(Expected,1e-8));
            if (J == 0) TestEqual(TEXT("wrist reaches requested component position despite nonidentity root"), Animated[B].GetTranslation(), Frame.GlobalTransCm,1e-8f);
        }
    }
    return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapRetargetRejectTest, "CardistryCapture.M2.Retarget.RejectIncompleteAndInvalid", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapRetargetRejectTest::RunTest(const FString& Parameters)
{
    using namespace CardCapRetargetTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Target = MakeSkeleton(Mapping); const auto Valid = Capture(Mapping, Target);
    auto Reject = [this](const FCardCapCapture& Input, const FReferenceSkeleton& Skeleton, const FString& Expected)
    {
        FCardCapRetargetResult Result; Result.FrameCount = 123; FString E;
        TestFalse(*Expected, FCardCapRetargeter::BuildLocalPoseFrames(Input, Skeleton, Result, E));
        TestTrue(TEXT("actionable validation error"), E.Contains(Expected));
        TestEqual(TEXT("failure is transactional"), Result.FrameCount,123);
    };
    { auto C = Valid; C.Hands[0].Frames.RemoveAt(1); Reject(C,Target,TEXT("sparse")); }
    { auto C = Valid; C.Hands[0].Frames[1].Frame = 0; Reject(C,Target,TEXT("unique and ordered")); }
    { auto C = Valid; C.Hands[1].Side = C.Hands[0].Side; Reject(C,Target,TEXT("duplicate side")); }
    { auto C = Valid; C.Hands[0].Frames[1].GlobalRotQuat = FQuat(0,0,0,0); Reject(C,Target,TEXT("nonunit")); }
    { auto C = Valid; C.Hands[0].Frames[1].BoneRotations.Remove(Mapping.Hands.FindChecked(TEXT("left")).BoneNames[1]); Reject(C,Target,TEXT("fifteen rotations")); }
    { auto C = Valid; auto& Names = C.BoneMapping.Hands.FindChecked(TEXT("left")).BoneNames; Names[1] = Names[0]; Reject(C,Target,TEXT("duplicate")); }
    { auto C = Valid; C.BoneMapping.Hands.FindChecked(TEXT("left")).BoneNames[1] = FName(TEXT("unit_missing_bone")); Reject(C,Target,TEXT("missing")); }
    { auto C = Valid; C.BoneMapping.RootBone = C.BoneMapping.Hands.FindChecked(TEXT("left")).BoneNames[0]; Reject(C,Target,TEXT("root_bone")); }
    { auto C = Valid; C.BoneMapping.ManoParents[2] = 2; Reject(C,Target,TEXT("parent must precede")); }
    Reject(Valid,MakeSkeleton(Mapping,true),TEXT("nearest mapped ancestor"));
    Reject(Valid,MakeSkeleton(Mapping,false,true),TEXT("unit positive scale"));
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapRetarget12ValidityTest, "CardistryCapture.P0.Retarget.ConditionalUnitsAndMissingMask", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapRetarget12ValidityTest::RunTest(const FString& Parameters)
{
    using namespace CardCapRetargetTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Target = MakeSkeleton(Mapping); auto Input = Capture(Mapping, Target);
    FCardCapRetargetResult Legacy, Conditional;
    if (!TestTrue(TEXT("legacy fixture retargets"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Legacy, Error))) return false;
    Input.FormatVersion = TEXT("1.2"); Input.Provenance.CoordinateUnits = TEXT("conditional_ue_units");
    Input.Provenance.MetricScale = ECardCapProvenance::Unobservable;
    for (auto& Hand : Input.Hands)
        for (auto& Frame : Hand.Frames) Frame.SampleKind = ECardCapSampleKind::Interpolated;
    if (!TestTrue(TEXT("explicitly conditional non-observation samples remain visible"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Conditional, Error))) { AddError(Error); return false; }
    for (int32 F = 0; F < Legacy.Frames.Num(); ++F)
        for (int32 B = 0; B < Target.GetNum(); ++B)
            TestTrue(TEXT("null metric claim makes no actor, component or bone-scale correction"), Conditional.Frames[F].LocalTransforms[B].Equals(Legacy.Frames[F].LocalTransforms[B], 0));
    Input.Hands[0].Frames[1].SampleKind = ECardCapSampleKind::Missing;
    Input.Hands[0].Frames[1].bPoseValid = false;
    FCardCapRetargetResult Rejected; Rejected.FrameCount = 123;
    TestFalse(TEXT("masked missing pose is never consumed as a real or conditional sample"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Rejected, Error));
    TestTrue(TEXT("missing sample error identifies its mask"), Error.Contains(TEXT(".validity.pose")));
    TestEqual(TEXT("missing pose rejection remains transactional"), Rejected.FrameCount, 123);
    return true;
}
IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapRetarget13LocalTest, "CardistryCapture.P0.Retarget.PerHandLocalOrigins", EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)
bool FCardCapRetarget13LocalTest::RunTest(const FString& Parameters)
{
    using namespace CardCapRetargetTest;
    FCardCapBoneMapping Mapping; FString Error;
    if (!TestTrue(TEXT("load mapping"), FCardCapJsonParser::LoadBoneMapping(FString(), Mapping, Error))) { AddError(Error); return false; }
    const auto Target = MakeSkeleton(Mapping); auto Input = Capture(Mapping, Target);
    FCardCapRetargetResult Legacy, Local;
    if (!FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Legacy, Error)) { AddError(Error); return false; }
    Input.FormatVersion = TEXT("1.3"); Input.Provenance.CoordinateFrame = TEXT("per_hand_wrist_local");
    Input.bInterHandTransformKnown = false;
    for (auto& Hand : Input.Hands)
        for (auto& Frame : Hand.Frames) { Frame.bHasGlobalTranslation = false; Frame.GlobalTransCm = FVector(123,456,789); }
    if (!TestTrue(TEXT("unknown translations allow independently local animations"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Local, Error))) { AddError(Error); return false; }
    for (int32 F = 0; F < Input.Meta.FrameCount; ++F)
    {
        const auto ComponentPose = Components(Target, Local.Frames[F].LocalTransforms);
        for (const auto& Hand : Input.Hands)
            TestTrue(TEXT("each wrist has display-only local origin"), ComponentPose[Target.FindBoneIndex(Mapping.Hands[Hand.Side].BoneNames[0])].GetTranslation().IsNearlyZero(1.e-8));
        for (int32 Bone = 0; Bone < Target.GetNum(); ++Bone)
        {
            TestTrue(TEXT("local preview preserves every source rotation"), Local.Frames[F].LocalTransforms[Bone].GetRotation().Equals(Legacy.Frames[F].LocalTransforms[Bone].GetRotation(),1.e-12));
            TestEqual(TEXT("local preview does not hide or align by bone scale"), Local.Frames[F].LocalTransforms[Bone].GetScale3D(), Legacy.Frames[F].LocalTransforms[Bone].GetScale3D());
        }
    }
    TestEqual(TEXT("display origin never written back to capture storage"), Input.Hands[0].Frames[0].GlobalTransCm, FVector(123,456,789));
    Input.Hands[0].Frames[0].bHasGlobalTranslation = true;
    TestFalse(TEXT("local mode rejects a supplied shared-space position"), FCardCapRetargeter::BuildLocalPoseFrames(Input, Target, Local, Error));
    TestTrue(TEXT("rejection identifies global translation"), Error.Contains(TEXT(".global_trans_cm")));
    return true;
}
#endif
