#include "CardCapRetargeter.h"
#include "ReferenceSkeleton.h"

namespace CardCapRetarget
{
bool FiniteVector(const FVector& V)
{
    return FMath::IsFinite(V.X) && FMath::IsFinite(V.Y) && FMath::IsFinite(V.Z);
}

bool UnitQuaternion(const FQuat& Q)
{
    return FMath::IsFinite(Q.X) && FMath::IsFinite(Q.Y) && FMath::IsFinite(Q.Z) && FMath::IsFinite(Q.W)
        && FMath::IsFinite(Q.SizeSquared()) && FMath::Abs(Q.SizeSquared() - 1.0) <= 1e-4;
}

bool UnitTransform(const FTransform& T)
{
    return FiniteVector(T.GetTranslation()) && FiniteVector(T.GetScale3D())
        && T.GetScale3D().Equals(FVector::OneVector, 1e-6) && UnitQuaternion(T.GetRotation());
}

struct FBinding
{
    int32 Hand = INDEX_NONE;
    int32 Joint = INDEX_NONE;
};
}

bool FCardCapRetargeter::BuildLocalPoseFrames(const FCardCapCapture& Capture,
    const FReferenceSkeleton& Target, FCardCapRetargetResult& OutResult, FString& OutError)
{
    using namespace CardCapRetarget;
    OutError.Reset();
    auto Fail = [&OutError](const FString& Path, const FString& Reason)
    {
        OutError = Path + TEXT(": ") + Reason;
        return false;
    };
    if (Capture.Meta.FrameCount <= 0 || !FMath::IsFinite(Capture.Meta.Fps) || Capture.Meta.Fps <= 0)
        return Fail(TEXT("capture.meta"), TEXT("positive finite fps and frame count are required"));
    if (Capture.Hands.IsEmpty() || Capture.Hands.Num() > 2)
        return Fail(TEXT("capture.hands"), TEXT("expected one or two prepared hand sequences"));
    const bool bSpatial13 = Capture.FormatVersion == TEXT("1.3");
    const bool bLocalPreview = bSpatial13 && Capture.Provenance.CoordinateFrame == TEXT("per_hand_wrist_local");
    if (bSpatial13 && (bLocalPreview == Capture.bInterHandTransformKnown ||
        (!bLocalPreview && Capture.Provenance.CoordinateFrame != TEXT("shared_camera"))))
        return Fail(TEXT("capture.provenance.coordinate_frame"), TEXT("local/shared coordinate frame disagrees with spatial validity mask"));
    const auto& Mapping = Capture.BoneMapping;
    if (Mapping.ManoParents.Num() != 16 || Mapping.ManoJointOrder.Num() != 16 || Mapping.LandmarkIndices.Num() != 16
        || Mapping.ManoParents[0] != INDEX_NONE)
        return Fail(TEXT("mapping"), TEXT("requires sixteen MANO joints, parents and landmark indices, wrist parent -1"));
    TSet<FString> JointNames; TSet<int32> Landmarks;
    for (int32 J = 0; J < 16; ++J)
    {
        if (Mapping.ManoJointOrder[J].IsEmpty() || JointNames.Contains(Mapping.ManoJointOrder[J])
            || Mapping.LandmarkIndices[J] < 0 || Mapping.LandmarkIndices[J] > 20 || Landmarks.Contains(Mapping.LandmarkIndices[J]))
            return Fail(FString::Printf(TEXT("mapping.joints[%d]"), J), TEXT("duplicate/invalid joint name or landmark index"));
        JointNames.Add(Mapping.ManoJointOrder[J]); Landmarks.Add(Mapping.LandmarkIndices[J]);
        if (J > 0 && (Mapping.ManoParents[J] < 0 || Mapping.ManoParents[J] >= J))
            return Fail(FString::Printf(TEXT("mapping.mano_parents[%d]"), J), TEXT("parent must precede child"));
    }
    const int32 BoneCount = Target.GetNum();
    const auto& RefLocal = Target.GetRefBonePose();
    if (BoneCount == 0 || RefLocal.Num() != BoneCount)
        return Fail(TEXT("target"), TEXT("reference skeleton is empty or has inconsistent pose size"));
    const int32 RootIndex = Target.FindBoneIndex(Mapping.RootBone);
    if (Mapping.RootBone.IsNone() || RootIndex != 0 || Target.GetParentIndex(RootIndex) != INDEX_NONE)
        return Fail(TEXT("mapping.root_bone"), TEXT("must identify the target's single root at index zero"));

    FCardCapRetargetResult Candidate;
    Candidate.Fps = Capture.Meta.Fps; Candidate.FrameCount = Capture.Meta.FrameCount;
    TArray<FTransform> RefComponent; RefComponent.SetNum(BoneCount);
    TSet<FName> TargetNames;
    for (int32 B = 0; B < BoneCount; ++B)
    {
        const FName Name = Target.GetBoneName(B);
        const int32 Parent = Target.GetParentIndex(B);
        const FString Path = FString::Printf(TEXT("target.bones[%d](%s)"), B, *Name.ToString());
        if (Name.IsNone() || TargetNames.Contains(Name)) return Fail(Path, TEXT("missing or duplicate bone name"));
        if ((B == 0 && Parent != INDEX_NONE) || (B > 0 && (Parent < 0 || Parent >= B)))
            return Fail(Path, TEXT("expected a single root and parent-before-child order"));
        if (!UnitTransform(RefLocal[B])) return Fail(Path, TEXT("reference transform must be finite with unit rotation and unit positive scale"));
        TargetNames.Add(Name); Candidate.BoneNames.Add(Name);
        // UE transform multiplication is local * parent; quaternion composition is parent * local.
        RefComponent[B] = Parent == INDEX_NONE ? RefLocal[B] : RefLocal[B] * RefComponent[Parent];
        if (!UnitTransform(RefComponent[B])) return Fail(Path, TEXT("reference component transform became invalid"));
    }

    TArray<TArray<int32>> HandBones; HandBones.SetNum(Capture.Hands.Num());
    TArray<FBinding> Bindings; Bindings.SetNum(BoneCount);
    TSet<FString> SeenSides; TSet<FName> MappedNames;
    for (int32 H = 0; H < Capture.Hands.Num(); ++H)
    {
        const auto& Hand = Capture.Hands[H];
        const FString Path = FString::Printf(TEXT("capture.hands[%d](%s)"), H, *Hand.Side);
        if ((Hand.Side != TEXT("left") && Hand.Side != TEXT("right")) || SeenSides.Contains(Hand.Side))
            return Fail(Path, TEXT("invalid or duplicate side"));
        SeenSides.Add(Hand.Side);
        const FCardCapHandBoneMapping* HandMapping = Mapping.Hands.Find(Hand.Side);
        if (!HandMapping || HandMapping->BoneNames.Num() != 16)
            return Fail(Path, TEXT("side mapping must contain wrist plus fifteen finger bones"));
        if (Hand.Frames.Num() != Capture.Meta.FrameCount)
            return Fail(Path + TEXT(".frames"), TEXT("sparse sequence: prepare and explicitly label every frame before retargeting"));
        for (int32 J = 0; J < 16; ++J)
        {
            const FName Name = HandMapping->BoneNames[J];
            const int32 Bone = Target.FindBoneIndex(Name);
            if (Name.IsNone() || MappedNames.Contains(Name) || Bone == INDEX_NONE || Bone == RootIndex)
                return Fail(Path + TEXT(".mapping.") + Name.ToString(), TEXT("missing, duplicate or root-overlapping mapped bone"));
            MappedNames.Add(Name); HandBones[H].Add(Bone); Bindings[Bone] = {H, J};
        }
        for (int32 F = 0; F < Hand.Frames.Num(); ++F)
        {
            const auto& Frame = Hand.Frames[F];
            const FString FP = FString::Printf(TEXT("%s.frames[%d]"), *Path, F);
            if (Frame.Frame != F) return Fail(FP, TEXT("full frames must be unique and ordered exactly 0..frame_count-1"));
            if ((Capture.FormatVersion == TEXT("1.2") || bSpatial13) && (!Frame.bPoseValid || Frame.SampleKind == ECardCapSampleKind::Missing))
                return Fail(FP + TEXT(".validity.pose"), TEXT("missing pose cannot be baked as an observed or conditional pose; prepare an explicitly sourced output first"));
            if (bSpatial13 && Frame.bHasGlobalTranslation == bLocalPreview)
                return Fail(FP + TEXT(".global_trans_cm"), TEXT("local animation requires unknown source translation; shared animation requires a supplied translation"));
            if (!FiniteVector(Frame.GlobalTransCm) || !UnitQuaternion(Frame.GlobalRotQuat))
                return Fail(FP, TEXT("wrist translation/rotation is nonfinite or nonunit"));
            if (Frame.BoneRotations.Num() != 15) return Fail(FP + TEXT(".bone_rotations"), TEXT("expected fifteen rotations"));
            for (int32 J = 1; J < 16; ++J)
            {
                const FQuat* Q = Frame.BoneRotations.Find(HandMapping->BoneNames[J]);
                if (!Q || !UnitQuaternion(*Q))
                    return Fail(FP + TEXT(".bone_rotations.") + HandMapping->BoneNames[J].ToString(), TEXT("missing, nonfinite or nonunit rotation"));
            }
        }
    }
    // Permit unmapped intermediate metacarpals, but not a different mapped ancestor.
    for (int32 H = 0; H < HandBones.Num(); ++H)
    {
        for (int32 J = 0; J < 16; ++J)
        {
            const int32 Bone = HandBones[H][J];
            const int32 ExpectedAncestor = J == 0 ? INDEX_NONE : HandBones[H][Mapping.ManoParents[J]];
            int32 Parent = Target.GetParentIndex(Bone);
            while (Parent != INDEX_NONE && Bindings[Parent].Hand == INDEX_NONE) Parent = Target.GetParentIndex(Parent);
            if (Parent != ExpectedAncestor)
                return Fail(TEXT("target.hierarchy.") + Target.GetBoneName(Bone).ToString(), TEXT("nearest mapped ancestor disagrees with configured MANO topology"));
        }
    }

    Candidate.Frames.Reserve(Candidate.FrameCount);
    for (int32 F = 0; F < Candidate.FrameCount; ++F)
    {
        TArray<TArray<FQuat>> SourceComponent; SourceComponent.SetNum(Capture.Hands.Num());
        for (int32 H = 0; H < Capture.Hands.Num(); ++H)
        {
            const auto& Frame = Capture.Hands[H].Frames[F];
            auto& Source = SourceComponent[H]; Source.SetNum(16);
            Source[0] = Frame.GlobalRotQuat; Source[0].Normalize();
            for (int32 J = 1; J < 16; ++J)
            {
                Source[J] = Source[Mapping.ManoParents[J]] * Frame.BoneRotations.FindChecked(Target.GetBoneName(HandBones[H][J]));
                Source[J].Normalize();
            }
        }
        FCardCapRetargetFrame ResultFrame; ResultFrame.Frame = F; ResultFrame.LocalTransforms = RefLocal;
        TArray<FTransform> AnimatedComponent; AnimatedComponent.SetNum(BoneCount);
        for (int32 B = 0; B < BoneCount; ++B)
        {
            const int32 Parent = Target.GetParentIndex(B);
            FTransform& Local = ResultFrame.LocalTransforms[B];
            const FBinding& Binding = Bindings[B];
            if (Binding.Hand != INDEX_NONE)
            {
                FQuat Desired = SourceComponent[Binding.Hand][Binding.Joint] * RefComponent[B].GetRotation();
                Desired.Normalize();
                FQuat LocalRotation = Parent == INDEX_NONE ? Desired : AnimatedComponent[Parent].GetRotation().Inverse() * Desired;
                LocalRotation.Normalize();
                if (F > 0)
                {
                    const FQuat Previous = Candidate.Frames[F - 1].LocalTransforms[B].GetRotation();
                    if (LocalRotation.X*Previous.X + LocalRotation.Y*Previous.Y + LocalRotation.Z*Previous.Z + LocalRotation.W*Previous.W < 0)
                        LocalRotation = FQuat(-LocalRotation.X, -LocalRotation.Y, -LocalRotation.Z, -LocalRotation.W);
                }
                Local.SetRotation(LocalRotation);
                if (Binding.Joint == 0)
                {
                    // Local preview origin is a display coordinate, never an
                    // estimate of where either hand is in a shared space.
                    const FVector DesiredPosition = bLocalPreview ? FVector::ZeroVector : Capture.Hands[Binding.Hand].Frames[F].GlobalTransCm;
                    Local.SetTranslation(Parent == INDEX_NONE ? DesiredPosition : AnimatedComponent[Parent].InverseTransformPosition(DesiredPosition));
                }
            }
            AnimatedComponent[B] = Parent == INDEX_NONE ? Local : Local * AnimatedComponent[Parent];
            if (!UnitTransform(Local) || !UnitTransform(AnimatedComponent[B]))
                return Fail(FString::Printf(TEXT("output.frames[%d].bones[%d]"), F, B), TEXT("retargeting produced a nonfinite or nonunit transform"));
        }
        Candidate.Frames.Add(MoveTemp(ResultFrame));
    }
    OutResult = MoveTemp(Candidate);
    return true;
}
