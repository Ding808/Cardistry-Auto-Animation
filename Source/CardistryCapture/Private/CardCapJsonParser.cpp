#include "CardCapJsonParser.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace CardCapJson
{
using FValue = TSharedPtr<FJsonValue>;
using FObject = TSharedPtr<FJsonObject>;
using FArray = TArray<FValue>;

struct FReader
{
    FString& Error;
    explicit FReader(FString& InError) : Error(InError) {}
    bool Fail(const FString& Path, const FString& Message)
    {
        if (Error.IsEmpty()) Error = Path + TEXT(": ") + Message;
        return false;
    }
    bool Require(bool Condition, const FString& Path, const FString& Message)
    {
        return Condition || Fail(Path, Message);
    }
    FValue Field(const FObject& Object, const TCHAR* Key, const FString& Path)
    {
        const FValue* Value = Object->Values.Find(Key);
        if (!Value) { Fail(Path + TEXT(".") + Key, TEXT("required field is missing")); return nullptr; }
        return *Value;
    }
    bool Object(const FValue& Value, FObject& Out, const FString& Path)
    {
        const FObject* Ptr = nullptr;
        if (!Value.IsValid() || Value->Type != EJson::Object || !Value->TryGetObject(Ptr) || !Ptr->IsValid())
            return Fail(Path, TEXT("expected object"));
        Out = *Ptr;
        return true;
    }
    bool ObjectField(const FObject& Parent, const TCHAR* Key, FObject& Out, const FString& Path)
    { return Object(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key); }
    bool Array(const FValue& Value, const FArray*& Out, const FString& Path, int32 Count = INDEX_NONE)
    {
        if (!Value.IsValid() || Value->Type != EJson::Array || !Value->TryGetArray(Out))
            return Fail(Path, TEXT("expected array"));
        return Count == INDEX_NONE || Require(Out->Num() == Count, Path, FString::Printf(TEXT("expected exactly %d entries"), Count));
    }
    bool ArrayField(const FObject& Parent, const TCHAR* Key, const FArray*& Out, const FString& Path, int32 Count = INDEX_NONE)
    { return Array(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key, Count); }
    bool Number(const FValue& Value, double& Out, const FString& Path)
    {
        // UE JsonValue can coerce strings/bools to numbers; forbid that explicitly.
        if (!Value.IsValid() || Value->Type != EJson::Number || !Value->TryGetNumber(Out) || !FMath::IsFinite(Out))
            return Fail(Path, TEXT("expected finite JSON number"));
        return true;
    }
    bool NumberField(const FObject& Parent, const TCHAR* Key, double& Out, const FString& Path)
    { return Number(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key); }
    bool Integer(const FValue& Value, int32& Out, const FString& Path, int32 Min = 0, int32 Max = MAX_int32)
    {
        double NumberValue = 0;
        if (!Number(Value, NumberValue, Path)) return false;
        if (NumberValue < Min || NumberValue > Max || FMath::FloorToDouble(NumberValue) != NumberValue)
            return Fail(Path, FString::Printf(TEXT("expected integer in [%d,%d]"), Min, Max));
        Out = static_cast<int32>(NumberValue);
        return true;
    }
    bool IntegerField(const FObject& Parent, const TCHAR* Key, int32& Out, const FString& Path, int32 Min = 0, int32 Max = MAX_int32)
    { return Integer(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key, Min, Max); }
    bool String(const FValue& Value, FString& Out, const FString& Path, bool AllowEmpty = false)
    {
        if (!Value.IsValid() || Value->Type != EJson::String || !Value->TryGetString(Out))
            return Fail(Path, TEXT("expected string"));
        return AllowEmpty || Require(!Out.IsEmpty(), Path, TEXT("must not be empty"));
    }
    bool StringField(const FObject& Parent, const TCHAR* Key, FString& Out, const FString& Path)
    { return String(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key); }
    bool BoolField(const FObject& Parent, const TCHAR* Key, bool& Out, const FString& Path)
    {
        FValue Value = Field(Parent, Key, Path);
        return (Value.IsValid() && Value->Type == EJson::Boolean && Value->TryGetBool(Out)) || Fail(Path + TEXT(".") + Key, TEXT("expected boolean"));
    }
    bool Score(const FObject& Parent, const TCHAR* Key, double& Out, const FString& Path)
    {
        return NumberField(Parent, Key, Out, Path) && Require(Out >= 0 && Out <= 1, Path + TEXT(".") + Key, TEXT("expected value in [0,1]"));
    }
    bool Numbers(const FValue& Value, TArray<double>& Out, const FString& Path, int32 Count = INDEX_NONE)
    {
        const FArray* Values = nullptr;
        if (!Array(Value, Values, Path, Count)) return false;
        for (int32 I = 0; I < Values->Num(); ++I)
        {
            double N = 0;
            if (!Number((*Values)[I], N, Index(Path, I))) return false;
            Out.Add(N);
        }
        return true;
    }
    bool Integers(const FValue& Value, TArray<int32>& Out, const FString& Path, int32 Min, int32 Max, bool Unique = true, int32 Count = INDEX_NONE)
    {
        const FArray* Values = nullptr;
        if (!Array(Value, Values, Path, Count)) return false;
        TSet<int32> Seen;
        for (int32 I = 0; I < Values->Num(); ++I)
        {
            int32 N = 0;
            if (!Integer((*Values)[I], N, Index(Path, I), Min, Max)) return false;
            if (Unique && Seen.Contains(N)) return Fail(Index(Path, I), TEXT("duplicate index"));
            Seen.Add(N); Out.Add(N);
        }
        return true;
    }
    bool Strings(const FValue& Value, TArray<FString>& Out, const FString& Path, bool Unique = false, int32 Count = INDEX_NONE)
    {
        const FArray* Values = nullptr;
        if (!Array(Value, Values, Path, Count)) return false;
        TSet<FString> Seen;
        for (int32 I = 0; I < Values->Num(); ++I)
        {
            FString S;
            if (!String((*Values)[I], S, Index(Path, I))) return false;
            if (Unique && Seen.Contains(S)) return Fail(Index(Path, I), TEXT("duplicate string"));
            Seen.Add(S); Out.Add(S);
        }
        return true;
    }
    bool Vector(const FValue& Value, FVector& Out, const FString& Path)
    {
        TArray<double> N;
        if (!Numbers(Value, N, Path, 3)) return false;
        Out = FVector(N[0], N[1], N[2]); return true;
    }
    bool VectorField(const FObject& Parent, const TCHAR* Key, FVector& Out, const FString& Path)
    { return Vector(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key); }
    bool Quaternion(const FValue& Value, FQuat& Out, const FString& Path)
    {
        TArray<double> N;
        if (!Numbers(Value, N, Path, 4)) return false;
        const double NormSquared = N[0]*N[0] + N[1]*N[1] + N[2]*N[2] + N[3]*N[3];
        if (!FMath::IsFinite(NormSquared) || FMath::Abs(NormSquared - 1.0) > 1e-4)
            return Fail(Path, TEXT("expected unit quaternion (squared-norm tolerance 1e-4); zero/unnormalized rotations are invalid"));
        Out = FQuat(N[0], N[1], N[2], N[3]);
        return true;
    }
    bool QuatField(const FObject& Parent, const TCHAR* Key, FQuat& Out, const FString& Path)
    { return Quaternion(Field(Parent, Key, Path), Out, Path + TEXT(".") + Key); }
    // Required keys are still read with Field. Only explicit null in 1.1 is
    // unknown; the strict existing readers validate every non-null value.
    bool NullableNumber(const FValue& Value, TOptional<double>& Out, const FString& Path, bool AllowNull)
    {
        if (AllowNull && Value.IsValid() && Value->Type == EJson::Null) { Out.Reset(); return true; }
        double N = 0;
        if (!Number(Value, N, Path)) return false;
        Out = N; return true;
    }
    bool NullableInteger(const FValue& Value, TOptional<int32>& Out, const FString& Path, bool AllowNull, int32 Min)
    {
        if (AllowNull && Value.IsValid() && Value->Type == EJson::Null) { Out.Reset(); return true; }
        int32 N = 0;
        if (!Integer(Value, N, Path, Min)) return false;
        Out = N; return true;
    }
    bool NullableVector(const FValue& Value, TOptional<FVector>& Out, const FString& Path, bool AllowNull)
    {
        if (AllowNull && Value.IsValid() && Value->Type == EJson::Null) { Out.Reset(); return true; }
        FVector V = FVector::ZeroVector;
        if (!Vector(Value, V, Path)) return false;
        Out = V; return true;
    }
    bool MaskedNumber(const FValue& Value, double& Out, bool& HasValue, const FString& Path, bool AllowNull)
    {
        TOptional<double> N;
        if (!NullableNumber(Value, N, Path, AllowNull)) return false;
        HasValue = N.IsSet();
        if (HasValue) Out = N.GetValue();
        return true;
    }
    bool MaskedVector(const FValue& Value, FVector& Out, bool& HasValue, const FString& Path, bool AllowNull)
    {
        TOptional<FVector> V;
        if (!NullableVector(Value, V, Path, AllowNull)) return false;
        HasValue = V.IsSet();
        if (HasValue) Out = V.GetValue();
        return true;
    }
    bool MaskedString(const FValue& Value, FString& Out, bool& HasValue, const FString& Path, bool AllowNull)
    {
        if (AllowNull && Value.IsValid() && Value->Type == EJson::Null) { HasValue = false; return true; }
        HasValue = true;
        return String(Value, Out, Path);
    }
    bool Provenance(const FObject& Parent, const TCHAR* Key, ECardCapProvenance& Out, const FString& Path)
    {
        FString Source;
        if (!StringField(Parent, Key, Source, Path)) return false;
        if (Source.Equals(TEXT("observed"), ESearchCase::CaseSensitive)) Out = ECardCapProvenance::Observed;
        else if (Source.Equals(TEXT("calibrated"), ESearchCase::CaseSensitive)) Out = ECardCapProvenance::Calibrated;
        else if (Source.Equals(TEXT("user_measured"), ESearchCase::CaseSensitive)) Out = ECardCapProvenance::UserMeasured;
        else if (Source.Equals(TEXT("inferred"), ESearchCase::CaseSensitive)) Out = ECardCapProvenance::Inferred;
        else if (Source.Equals(TEXT("unobservable"), ESearchCase::CaseSensitive)) Out = ECardCapProvenance::Unobservable;
        else return Fail(Path + TEXT(".") + Key, TEXT("expected observed, calibrated, user_measured, inferred or unobservable"));
        return true;
    }
    bool SampleKind(const FObject& Parent, ECardCapSampleKind& Out, const FString& Path)
    {
        FString Kind;
        if (!StringField(Parent, TEXT("sample_kind"), Kind, Path)) return false;
        if (Kind.Equals(TEXT("detected_model_observation"), ESearchCase::CaseSensitive)) Out = ECardCapSampleKind::DetectedModelObservation;
        else if (Kind.Equals(TEXT("tracked_roi_model_observation"), ESearchCase::CaseSensitive)) Out = ECardCapSampleKind::TrackedRoiModelObservation;
        else if (Kind.Equals(TEXT("interpolated"), ESearchCase::CaseSensitive)) Out = ECardCapSampleKind::Interpolated;
        else if (Kind.Equals(TEXT("endpoint_hold"), ESearchCase::CaseSensitive)) Out = ECardCapSampleKind::EndpointHold;
        else if (Kind.Equals(TEXT("missing"), ESearchCase::CaseSensitive)) Out = ECardCapSampleKind::Missing;
        else return Fail(Path + TEXT(".sample_kind"), TEXT("unknown sample kind"));
        return true;
    }
    static FString Index(const FString& Path, int32 I) { return FString::Printf(TEXT("%s[%d]"), *Path, I); }
};

// FJsonObject is a map and otherwise silently replaces duplicate object keys.
bool Deserialize(const FString& Json, FObject& Out, FString& Error)
{
    struct FScope { bool bObject; TSet<FString> Keys; };
    TArray<FScope> Stack;
    auto Scan = TJsonReaderFactory<>::Create(Json);
    EJsonNotation Notation;
    while (Scan->ReadNext(Notation))
    {
        const bool bEnd = Notation == EJsonNotation::ObjectEnd || Notation == EJsonNotation::ArrayEnd;
        if (Notation == EJsonNotation::Error) break;
        if (!bEnd && !Stack.IsEmpty() && Stack.Last().bObject)
        {
            const FString& Key = Scan->GetIdentifier();
            if (Stack.Last().Keys.Contains(Key))
            {
                Error = FString::Printf(TEXT("JSON line %u character %u: duplicate object key '%s'"), Scan->GetLineNumber(), Scan->GetCharacterNumber(), *Key);
                return false;
            }
            Stack.Last().Keys.Add(Key);
        }
        if (Notation == EJsonNotation::ObjectStart || Notation == EJsonNotation::ArrayStart)
            Stack.Add({Notation == EJsonNotation::ObjectStart, {}});
        else if (bEnd && !Stack.IsEmpty()) Stack.Pop();
    }
    auto Reader = TJsonReaderFactory<>::Create(Json);
    if (!FJsonSerializer::Deserialize(Reader, Out) || !Out.IsValid())
    {
        Error = TEXT("$: invalid JSON object: ") + Reader->GetErrorMessage(); return false;
    }
    return true;
}

bool ReadMapping(const FObject& Root, FCardCapBoneMapping& Out, FString& Error)
{
    FReader R(Error);
    int32 Version = 0;
    FString RootName;
    FObject Hands;
    if (!R.IntegerField(Root, TEXT("schema_version"), Version, TEXT("mapping"), 1, 1) ||
        !R.StringField(Root, TEXT("root_bone"), RootName, TEXT("mapping")) ||
        !R.Strings(R.Field(Root, TEXT("mano_joint_order"), TEXT("mapping")), Out.ManoJointOrder, TEXT("mapping.mano_joint_order"), true, 16) ||
        !R.Integers(R.Field(Root, TEXT("mano_parents"), TEXT("mapping")), Out.ManoParents, TEXT("mapping.mano_parents"), -1, 15, false, 16) ||
        !R.Integers(R.Field(Root, TEXT("landmark_indices"), TEXT("mapping")), Out.LandmarkIndices, TEXT("mapping.landmark_indices"), 0, 20, true, 16) ||
        !R.ObjectField(Root, TEXT("hands"), Hands, TEXT("mapping"))) return false;
    Out.RootBone = FName(*RootName);
    if (!R.Require(!Out.RootBone.IsNone(), TEXT("mapping.root_bone"), TEXT("invalid bone name")) ||
        !R.Require(Out.ManoParents[0] == -1, TEXT("mapping.mano_parents[0]"), TEXT("wrist must have parent -1"))) return false;
    for (int32 I = 1; I < 16; ++I)
        if (!R.Require(Out.ManoParents[I] >= 0 && Out.ManoParents[I] < I, FReader::Index(TEXT("mapping.mano_parents"), I), TEXT("parent must precede child"))) return false;
    TSet<FName> AllNames; AllNames.Add(Out.RootBone);
    for (const FString Side : {FString(TEXT("left")), FString(TEXT("right"))})
    {
        const FString P = TEXT("mapping.hands.") + Side;
        FObject H; TArray<FString> Names; FCardCapHandBoneMapping Mapping;
        if (!R.ObjectField(Hands, *Side, H, TEXT("mapping.hands")) ||
            !R.Strings(R.Field(H, TEXT("bone_names"), P), Names, P + TEXT(".bone_names"), true, 16) ||
            !R.VectorField(H, TEXT("rest_offset_camera_m"), Mapping.RestOffsetCameraM, P)) return false;
        for (int32 I = 0; I < Names.Num(); ++I)
        {
            const FName Name(*Names[I]);
            if (!R.Require(!Name.IsNone() && !AllNames.Contains(Name), FReader::Index(P + TEXT(".bone_names"), I), TEXT("invalid or duplicate bone name"))) return false;
            AllNames.Add(Name); Mapping.BoneNames.Add(Name);
        }
        Out.Hands.Add(Side, MoveTemp(Mapping));
    }
    return true;
}

bool ReadCapture(const FObject& Root, FCardCapCapture& Out, FString& Error)
{
    FReader R(Error);
    if (!R.StringField(Root, TEXT("format_version"), Out.FormatVersion, TEXT("$")) ||
        !R.Require(Out.FormatVersion == TEXT("1.0") || Out.FormatVersion == TEXT("1.1") || Out.FormatVersion == TEXT("1.2") || Out.FormatVersion == TEXT("1.3"), TEXT("$.format_version"), TEXT("unsupported format; expected 1.0, 1.1, 1.2 or 1.3"))) return false;
    const bool bSpatial13 = Out.FormatVersion == TEXT("1.3");
    const bool bProvenance12 = Out.FormatVersion == TEXT("1.2") || bSpatial13;
    const bool bNullablePackets = Out.FormatVersion == TEXT("1.1") || bProvenance12;
    FObject Meta, Camera, Intrinsics, Scale, Events, Quality;
    const FArray *Resolution = nullptr, *Hands = nullptr, *Packets = nullptr;
    const FArray *Splits = nullptr, *Merges = nullptr, *Releases = nullptr, *Ranges = nullptr;
    if (!R.ObjectField(Root, TEXT("meta"), Meta, TEXT("$")) ||
        !R.StringField(Meta, TEXT("source_video"), Out.Meta.SourceVideo, TEXT("$.meta")) ||
        !R.NumberField(Meta, TEXT("fps"), Out.Meta.Fps, TEXT("$.meta")) ||
        !R.Require(Out.Meta.Fps > 0, TEXT("$.meta.fps"), TEXT("must be positive")) ||
        !R.IntegerField(Meta, TEXT("frame_count"), Out.Meta.FrameCount, TEXT("$.meta"), 1) ||
        !R.ArrayField(Meta, TEXT("resolution"), Resolution, TEXT("$.meta"), 2) ||
        !R.Integer((*Resolution)[0], Out.Meta.Resolution.X, TEXT("$.meta.resolution[0]"), 1) ||
        !R.Integer((*Resolution)[1], Out.Meta.Resolution.Y, TEXT("$.meta.resolution[1]"), 1) ||
        !R.StringField(Meta, TEXT("processed_at"), Out.Meta.ProcessedAt, TEXT("$.meta")) ||
        !R.StringField(Meta, TEXT("pipeline_version"), Out.Meta.PipelineVersion, TEXT("$.meta"))) return false;
    const int32 LastFrame = Out.Meta.FrameCount - 1;
    if (!R.ObjectField(Root, TEXT("camera"), Camera, TEXT("$"))) return false;
    const FValue KValue = R.Field(Camera, TEXT("intrinsics"), TEXT("$.camera"));
    if (!KValue.IsValid()) return false;
    Out.Camera.bHasIntrinsics = !(bProvenance12 && KValue->Type == EJson::Null);
    if (Out.Camera.bHasIntrinsics &&
        (!R.Object(KValue, Intrinsics, TEXT("$.camera.intrinsics")) ||
         !R.NumberField(Intrinsics, TEXT("fx"), Out.Camera.Fx, TEXT("$.camera.intrinsics")) ||
         !R.NumberField(Intrinsics, TEXT("fy"), Out.Camera.Fy, TEXT("$.camera.intrinsics")) ||
         !R.NumberField(Intrinsics, TEXT("cx"), Out.Camera.Cx, TEXT("$.camera.intrinsics")) ||
         !R.NumberField(Intrinsics, TEXT("cy"), Out.Camera.Cy, TEXT("$.camera.intrinsics")) ||
         !R.Require(Out.Camera.Fx > 0 && Out.Camera.Fy > 0, TEXT("$.camera.intrinsics"), TEXT("focal lengths must be positive")))) return false;
    const FValue DValue = R.Field(Camera, TEXT("distortion"), TEXT("$.camera"));
    if (!DValue.IsValid()) return false;
    Out.Camera.bHasDistortion = !(bProvenance12 && DValue->Type == EJson::Null);
    if (Out.Camera.bHasDistortion)
    {
        if (!R.Numbers(DValue, Out.Camera.Distortion, TEXT("$.camera.distortion"))) return false;
        const int32 D = Out.Camera.Distortion.Num();
        if (!R.Require(D == 4 || D == 5 || D == 8 || D == 12 || D == 14, TEXT("$.camera.distortion"), TEXT("expected OpenCV coefficient count 4,5,8,12 or 14"))) return false;
    }
    if (!R.BoolField(Camera, TEXT("calibrated"), Out.Camera.bCalibrated, TEXT("$.camera")) ||
        !R.ObjectField(Root, TEXT("scale"), Scale, TEXT("$")) ||
        !R.MaskedNumber(R.Field(Scale, TEXT("meters_per_unit"), TEXT("$.scale")), Out.Scale.MetersPerUnit, Out.Scale.bHasMetersPerUnit, TEXT("$.scale.meters_per_unit"), bProvenance12) ||
        !R.Require(!Out.Scale.bHasMetersPerUnit || Out.Scale.MetersPerUnit > 0, TEXT("$.scale.meters_per_unit"), TEXT("must be positive when supplied")) ||
        !R.MaskedString(R.Field(Scale, TEXT("anchor_method"), TEXT("$.scale")), Out.Scale.AnchorMethod, Out.Scale.bHasAnchorMethod, TEXT("$.scale.anchor_method"), bProvenance12) ||
        !R.Require(!Out.Scale.bHasAnchorMethod || Out.Scale.AnchorMethod == TEXT("card_pnp") || Out.Scale.AnchorMethod == TEXT("hand_prior") || (bProvenance12 && Out.Scale.AnchorMethod.Equals(TEXT("user_measured"), ESearchCase::CaseSensitive)), TEXT("$.scale.anchor_method"), TEXT("expected card_pnp, hand_prior, or (1.2 only) user_measured")) ||
        !R.MaskedNumber(R.Field(Scale, TEXT("confidence"), TEXT("$.scale")), Out.Scale.Confidence, Out.Scale.bHasConfidence, TEXT("$.scale.confidence"), bProvenance12) ||
        !R.Require(!Out.Scale.bHasConfidence || (Out.Scale.Confidence >= 0 && Out.Scale.Confidence <= 1), TEXT("$.scale.confidence"), TEXT("expected value in [0,1]")) ||
        !R.Integers(R.Field(Scale, TEXT("anchor_frames"), TEXT("$.scale")), Out.Scale.AnchorFrames, TEXT("$.scale.anchor_frames"), 0, LastFrame)) return false;
    if (bProvenance12)
    {
        FObject Provenance, Validity;
        bool HasK = false, HasD = false, HasScale = false;
        if (!R.ObjectField(Root, TEXT("provenance"), Provenance, TEXT("$")) ||
            !R.Provenance(Provenance, TEXT("camera_intrinsics"), Out.Provenance.CameraIntrinsics, TEXT("$.provenance")) ||
            !R.Provenance(Provenance, TEXT("camera_distortion"), Out.Provenance.CameraDistortion, TEXT("$.provenance")) ||
            !R.Provenance(Provenance, TEXT("metric_scale"), Out.Provenance.MetricScale, TEXT("$.provenance")) ||
            !R.Provenance(Provenance, TEXT("hand_geometry"), Out.Provenance.HandGeometry, TEXT("$.provenance")) ||
            !R.StringField(Provenance, TEXT("coordinate_units"), Out.Provenance.CoordinateUnits, TEXT("$.provenance")) ||
            !R.Strings(R.Field(Provenance, TEXT("assumptions"), TEXT("$.provenance")), Out.Provenance.Assumptions, TEXT("$.provenance.assumptions")) ||
            !R.ObjectField(Root, TEXT("validity"), Validity, TEXT("$")) ||
            !R.BoolField(Validity, TEXT("camera_intrinsics"), HasK, TEXT("$.validity")) ||
            !R.BoolField(Validity, TEXT("camera_distortion"), HasD, TEXT("$.validity")) ||
            !R.BoolField(Validity, TEXT("metric_scale"), HasScale, TEXT("$.validity"))) return false;
        if (!R.Require(HasK == Out.Camera.bHasIntrinsics, TEXT("$.validity.camera_intrinsics"), TEXT("mask must match null/non-null field")) ||
            !R.Require(HasD == Out.Camera.bHasDistortion, TEXT("$.validity.camera_distortion"), TEXT("mask must match null/non-null field")) ||
            !R.Require(HasScale == Out.Scale.bHasMetersPerUnit, TEXT("$.validity.metric_scale"), TEXT("mask must match null/non-null field")) ||
            !R.Require(HasK == (Out.Provenance.CameraIntrinsics != ECardCapProvenance::Unobservable), TEXT("$.provenance.camera_intrinsics"), TEXT("unobservable requires null; a supplied value requires a source")) ||
            !R.Require(HasD == (Out.Provenance.CameraDistortion != ECardCapProvenance::Unobservable), TEXT("$.provenance.camera_distortion"), TEXT("unobservable requires null; a supplied value requires a source")) ||
            !R.Require(HasScale == (Out.Provenance.MetricScale != ECardCapProvenance::Unobservable), TEXT("$.provenance.metric_scale"), TEXT("unobservable requires null; a supplied value requires a source")) ||
            !R.Require(!Out.Camera.bCalibrated || (HasK && Out.Provenance.CameraIntrinsics == ECardCapProvenance::Calibrated), TEXT("$.camera.calibrated"), TEXT("calibrated flag requires calibrated intrinsics provenance")) ||
            !R.Require(HasScale ? Out.Scale.bHasAnchorMethod : (!Out.Scale.bHasAnchorMethod && !Out.Scale.bHasConfidence && Out.Scale.AnchorFrames.IsEmpty()), TEXT("$.scale"), TEXT("unknown metric scale requires null anchor/confidence and empty anchor frames; known scale requires an anchor")) ||
            !R.Require(Out.Provenance.CoordinateUnits.Equals(HasScale ? TEXT("ue_centimeters") : TEXT("conditional_ue_units"), ESearchCase::CaseSensitive), TEXT("$.provenance.coordinate_units"), TEXT("units must agree with metric scale validity")) ||
            !R.Require(HasScale || !Out.Provenance.Assumptions.IsEmpty(), TEXT("$.provenance.assumptions"), TEXT("conditional geometry requires explicit assumptions"))) return false;
        if (bSpatial13)
        {
            if (!R.StringField(Provenance, TEXT("coordinate_frame"), Out.Provenance.CoordinateFrame, TEXT("$.provenance")) ||
                !R.BoolField(Validity, TEXT("inter_hand_transform"), Out.bInterHandTransformKnown, TEXT("$.validity"))) return false;
            const bool bLocal = Out.Provenance.CoordinateFrame.Equals(TEXT("per_hand_wrist_local"), ESearchCase::CaseSensitive);
            const bool bShared = Out.Provenance.CoordinateFrame.Equals(TEXT("shared_camera"), ESearchCase::CaseSensitive);
            if (!R.Require(bLocal || bShared, TEXT("$.provenance.coordinate_frame"), TEXT("expected per_hand_wrist_local or shared_camera")) ||
                !R.Require(bShared == Out.bInterHandTransformKnown, TEXT("$.validity.inter_hand_transform"), TEXT("per-hand local coordinates cannot declare a shared spatial relation")) ||
                !R.Require(bShared == Out.Camera.bHasIntrinsics, TEXT("$.camera.intrinsics"), TEXT("shared coordinates require an evidenced source camera; local display cameras are never source intrinsics")) ||
                !R.Require(!bLocal || !Out.Camera.bHasDistortion, TEXT("$.camera.distortion"), TEXT("per-hand local output must not publish a shared lens-distortion model"))) return false;
        }
    }
    if (!R.ArrayField(Root, TEXT("hands"), Hands, TEXT("$"))) return false;
    if (!R.Require(Hands->Num() >= 1 && Hands->Num() <= 2, TEXT("$.hands"), TEXT("expected one or two unique hand sides"))) return false;
    TSet<FString> Sides;
    TSet<FName> ContactBoneNames;
    for (const auto& Mapping : Out.BoneMapping.Hands)
        for (const FName Name : Mapping.Value.BoneNames) ContactBoneNames.Add(Name);
    for (int32 H = 0; H < Hands->Num(); ++H)
    {
        const FString P = FReader::Index(TEXT("$.hands"), H);
        FObject Hand; const FArray* Frames = nullptr; FCardCapHand Parsed;
        if (!R.Object((*Hands)[H], Hand, P) || !R.StringField(Hand, TEXT("side"), Parsed.Side, P) ||
            !R.Require(Out.BoneMapping.Hands.Contains(Parsed.Side), P + TEXT(".side"), TEXT("expected left or right")) ||
            !R.Require(!Sides.Contains(Parsed.Side), P + TEXT(".side"), TEXT("duplicate hand side")) ||
            !R.Numbers(R.Field(Hand, TEXT("mano_shape"), P), Parsed.ManoShape, P + TEXT(".mano_shape"), 10) ||
            !R.ArrayField(Hand, TEXT("frames"), Frames, P)) return false;
        Sides.Add(Parsed.Side);
        const auto& BoneNames = Out.BoneMapping.Hands.FindChecked(Parsed.Side).BoneNames;
        TSet<int32> SeenFrames;
        for (int32 I = 0; I < Frames->Num(); ++I)
        {
            const FString Q = FReader::Index(P + TEXT(".frames"), I);
            FObject Frame, Rotations; const FArray* Joints = nullptr; FCardCapHandFrame F;
            if (!R.Object((*Frames)[I], Frame, Q) || !R.IntegerField(Frame, TEXT("frame"), F.Frame, Q, 0, LastFrame) ||
                !R.Require(!SeenFrames.Contains(F.Frame), Q + TEXT(".frame"), TEXT("duplicate hand frame")) ||
                !R.Score(Frame, TEXT("confidence"), F.Confidence, Q) ||
                !R.MaskedVector(R.Field(Frame, TEXT("global_trans_cm"), Q), F.GlobalTransCm, F.bHasGlobalTranslation, Q + TEXT(".global_trans_cm"), bSpatial13) ||
                !R.QuatField(Frame, TEXT("global_rot_quat"), F.GlobalRotQuat, Q) ||
                !R.ArrayField(Frame, TEXT("joint_positions_cm"), Joints, Q, 21) ||
                !R.ObjectField(Frame, TEXT("bone_rotations"), Rotations, Q) ||
                !R.Integers(R.Field(Frame, TEXT("occluded_joints"), Q), F.OccludedJoints, Q + TEXT(".occluded_joints"), 0, 20)) return false;
            if (bProvenance12)
            {
                FObject Validity;
                if (!R.SampleKind(Frame, F.SampleKind, Q) ||
                    !R.ObjectField(Frame, TEXT("validity"), Validity, Q) ||
                    !R.BoolField(Validity, TEXT("pose"), F.bPoseValid, Q + TEXT(".validity")) ||
                    !R.Require(F.bPoseValid == (F.SampleKind != ECardCapSampleKind::Missing), Q + TEXT(".validity.pose"), TEXT("missing must be invalid; supplied conditional poses must be valid"))) return false;
                if (bSpatial13)
                {
                    bool GlobalValid = false;
                    if (!R.BoolField(Validity, TEXT("global_translation"), GlobalValid, Q + TEXT(".validity")) ||
                        !R.Require(GlobalValid == F.bHasGlobalTranslation, Q + TEXT(".validity.global_translation"), TEXT("mask must match null/non-null global translation")) ||
                        !R.Require(GlobalValid == Out.bInterHandTransformKnown, Q + TEXT(".global_trans_cm"), TEXT("local hand poses require null global translation; shared camera poses require numeric global translation"))) return false;
                }
            }
            SeenFrames.Add(F.Frame);
            for (int32 J = 0; J < 21; ++J)
            {
                FVector V;
                if (!R.Vector((*Joints)[J], V, FReader::Index(Q + TEXT(".joint_positions_cm"), J))) return false;
                F.JointPositionsCm.Add(V);
            }
            if (bSpatial13 && !Out.bInterHandTransformKnown &&
                !R.Require(F.JointPositionsCm[0].IsNearlyZero(1.e-8), Q + TEXT(".joint_positions_cm[0]"), TEXT("per-hand local geometry must be wrist-relative; the origin is not a global position"))) return false;
            if (!R.Require(Rotations->Values.Num() == 15, Q + TEXT(".bone_rotations"), TEXT("expected exactly the 15 configured finger bones"))) return false;
            for (const auto& Pair : Rotations->Values)
            {
                const FName Name(*Pair.Key); const int32 BoneIndex = BoneNames.IndexOfByKey(Name);
                if (!R.Require(BoneIndex > 0 && BoneNames[BoneIndex].ToString() == Pair.Key, Q + TEXT(".bone_rotations.") + Pair.Key, TEXT("unknown bone or wrong hand; expected configured exact spelling"))) return false;
                FQuat Rotation;
                if (!R.Quaternion(Pair.Value, Rotation, Q + TEXT(".bone_rotations.") + Pair.Key)) return false;
                F.BoneRotations.Add(Name, Rotation);
            }
            Parsed.Frames.Add(MoveTemp(F));
        }
        // Sparse hand frames are allowed; holes remain holes. Never synthesize poses here.
        Parsed.Frames.Sort([](const FCardCapHandFrame& A, const FCardCapHandFrame& B) { return A.Frame < B.Frame; });
        Out.Hands.Add(MoveTemp(Parsed));
    }
    if (!R.ArrayField(Root, TEXT("packets"), Packets, TEXT("$"))) return false;
    if (bSpatial13 && !Out.bInterHandTransformKnown &&
        !R.Require(Packets->IsEmpty(), TEXT("$.packets"), TEXT("per-hand local previews cannot claim shared-space packet poses"))) return false;
    TSet<FString> PacketIds;
    for (int32 I = 0; I < Packets->Num(); ++I)
    {
        const FString P = FReader::Index(TEXT("$.packets"), I);
        FObject Packet; const FArray *Frames = nullptr, *Dimensions = nullptr; FCardCapPacket Parsed;
        if (!R.Object((*Packets)[I], Packet, P) || !R.StringField(Packet, TEXT("id"), Parsed.Id, P) ||
            !R.Require(!PacketIds.Contains(Parsed.Id), P + TEXT(".id"), TEXT("duplicate packet id")) ||
            !R.IntegerField(Packet, TEXT("birth_frame"), Parsed.BirthFrame, P, 0, LastFrame) ||
            !R.IntegerField(Packet, TEXT("death_frame"), Parsed.DeathFrame, P, Parsed.BirthFrame, LastFrame) ||
            !R.NullableInteger(R.Field(Packet, TEXT("card_count_estimate"), P), Parsed.CardCountEstimate, P + TEXT(".card_count_estimate"), bNullablePackets, 1) ||
            !R.NullableNumber(R.Field(Packet, TEXT("card_count_confidence"), P), Parsed.CardCountConfidence, P + TEXT(".card_count_confidence"), bNullablePackets) ||
            !R.Require(!Parsed.CardCountConfidence.IsSet() || (Parsed.CardCountConfidence.GetValue() >= 0 && Parsed.CardCountConfidence.GetValue() <= 1), P + TEXT(".card_count_confidence"), TEXT("expected value in [0,1]")) ||
            !R.Require(Parsed.CardCountEstimate.IsSet() == Parsed.CardCountConfidence.IsSet(), P + TEXT(".card_count_confidence"), TEXT("card count and its confidence must both be numbers or both be null"))) return false;
        const FValue DimensionsValue = R.Field(Packet, TEXT("dimensions_cm"), P);
        if (!DimensionsValue.IsValid()) return false;
        if (!(bProvenance12 && DimensionsValue->Type == EJson::Null))
        {
            if (!R.Array(DimensionsValue, Dimensions, P + TEXT(".dimensions_cm"), 3) ||
                !R.MaskedNumber((*Dimensions)[0], Parsed.DimensionsCm.Width, Parsed.DimensionsCm.bHasWidth, P + TEXT(".dimensions_cm[0]"), bProvenance12) ||
                !R.MaskedNumber((*Dimensions)[1], Parsed.DimensionsCm.Height, Parsed.DimensionsCm.bHasHeight, P + TEXT(".dimensions_cm[1]"), bProvenance12) ||
                !R.NullableNumber((*Dimensions)[2], Parsed.DimensionsCm.Thickness, P + TEXT(".dimensions_cm[2]"), bNullablePackets) ||
                !R.Require((!Parsed.DimensionsCm.bHasWidth || Parsed.DimensionsCm.Width > 0) && (!Parsed.DimensionsCm.bHasHeight || Parsed.DimensionsCm.Height > 0) && (!Parsed.DimensionsCm.Thickness.IsSet() || Parsed.DimensionsCm.Thickness.GetValue() > 0), P + TEXT(".dimensions_cm"), TEXT("known dimensions must be positive; 1.1 allows null thickness, 1.2 allows any null dimension"))) return false;
        }
        if (!R.ArrayField(Packet, TEXT("frames"), Frames, P)) return false;
        PacketIds.Add(Parsed.Id); TSet<int32> SeenFrames;
        for (int32 J = 0; J < Frames->Num(); ++J)
        {
            const FString Q = FReader::Index(P + TEXT(".frames"), J);
            FObject Frame; FCardCapPacketFrame F; FString State; TArray<FString> Bones;
            if (!R.Object((*Frames)[J], Frame, Q) || !R.IntegerField(Frame, TEXT("frame"), F.Frame, Q, Parsed.BirthFrame, Parsed.DeathFrame) ||
                !R.Require(!SeenFrames.Contains(F.Frame), Q + TEXT(".frame"), TEXT("duplicate packet frame")) ||
                !R.Score(Frame, TEXT("confidence"), F.Confidence, Q) ||
                !R.VectorField(Frame, TEXT("position_cm"), F.PositionCm, Q) ||
                !R.QuatField(Frame, TEXT("rotation_quat"), F.RotationQuat, Q) ||
                !R.NullableVector(R.Field(Frame, TEXT("linear_velocity_cm_s"), Q), F.LinearVelocityCmS, Q + TEXT(".linear_velocity_cm_s"), bNullablePackets) ||
                !R.NullableVector(R.Field(Frame, TEXT("angular_velocity_rad_s"), Q), F.AngularVelocityRadS, Q + TEXT(".angular_velocity_rad_s"), bNullablePackets) ||
                !R.StringField(Frame, TEXT("contact_state"), State, Q) ||
                !R.NullableNumber(R.Field(Frame, TEXT("reprojection_error_px"), Q), F.ReprojectionErrorPx, Q + TEXT(".reprojection_error_px"), bNullablePackets) ||
                !R.Require(!F.ReprojectionErrorPx.IsSet() || F.ReprojectionErrorPx.GetValue() >= 0, Q + TEXT(".reprojection_error_px"), TEXT("must be nonnegative"))) return false;
            const FValue BoneValue = R.Field(Frame, TEXT("contact_bones"), Q);
            if (!BoneValue.IsValid()) return false;
            const bool bUnknownBones = bNullablePackets && BoneValue->Type == EJson::Null;
            if (!bUnknownBones && !R.Strings(BoneValue, Bones, Q + TEXT(".contact_bones"), true)) return false;
            SeenFrames.Add(F.Frame);
            if (State == TEXT("GRIPPED")) F.ContactState = ECardCapContactState::Gripped;
            else if (State == TEXT("FREE")) F.ContactState = ECardCapContactState::Free;
            else if (State == TEXT("RESTING")) F.ContactState = ECardCapContactState::Resting;
            else if (State == TEXT("SLIDING")) F.ContactState = ECardCapContactState::Sliding;
            // Keep legacy states' FString == case-insensitive acceptance above.
            // UNKNOWN is new in 1.1 and explicitly requires its canonical spelling.
            else if (bNullablePackets && State.Equals(TEXT("UNKNOWN"), ESearchCase::CaseSensitive)) F.ContactState = ECardCapContactState::Unknown;
            else return R.Fail(Q + TEXT(".contact_state"), TEXT("unknown contact state"));
            TArray<FName> ParsedBones;
            for (int32 K = 0; K < Bones.Num(); ++K)
            {
                const FName Name(*Bones[K]);
                if (!R.Require(ContactBoneNames.Contains(Name) && Name.ToString() == Bones[K], FReader::Index(Q + TEXT(".contact_bones"), K), TEXT("bone is absent from mapping"))) return false;
                ParsedBones.Add(Name);
            }
            if (!bUnknownBones) F.ContactBones = MoveTemp(ParsedBones);
            Parsed.Frames.Add(MoveTemp(F));
        }
        Parsed.Frames.Sort([](const FCardCapPacketFrame& A, const FCardCapPacketFrame& B) { return A.Frame < B.Frame; });
        Out.Packets.Add(MoveTemp(Parsed));
    }
    if (!R.ObjectField(Root, TEXT("events"), Events, TEXT("$")) ||
        !R.ArrayField(Events, TEXT("splits"), Splits, TEXT("$.events")) ||
        !R.ArrayField(Events, TEXT("merges"), Merges, TEXT("$.events")) ||
        !R.ArrayField(Events, TEXT("releases"), Releases, TEXT("$.events"))) return false;
    TSet<FString> SeenEvents;
    for (int32 I = 0; I < Splits->Num(); ++I)
    {
        const FString P = FReader::Index(TEXT("$.events.splits"), I); FObject O; FCardCapSplitEvent E;
        if (!R.Object((*Splits)[I], O, P) || !R.IntegerField(O, TEXT("frame"), E.Frame, P, 0, LastFrame) ||
            !R.StringField(O, TEXT("source"), E.Source, P) ||
            !R.Require(PacketIds.Contains(E.Source), P + TEXT(".source"), TEXT("unknown packet id")) ||
            !R.Strings(R.Field(O, TEXT("results"), P), E.Results, P + TEXT(".results"), true) ||
            !R.Require(E.Results.Num() >= 2, P + TEXT(".results"), TEXT("split requires at least two results"))) return false;
        for (int32 J = 0; J < E.Results.Num(); ++J)
            if (!R.Require(PacketIds.Contains(E.Results[J]) && E.Results[J] != E.Source, FReader::Index(P + TEXT(".results"), J), TEXT("unknown packet or self split"))) return false;
        const FString Key = FString::Printf(TEXT("split:%d:%s"), E.Frame, *E.Source);
        if (!R.Require(!SeenEvents.Contains(Key), P, TEXT("duplicate split event"))) return false;
        SeenEvents.Add(Key); Out.Events.Splits.Add(MoveTemp(E));
    }
    for (int32 I = 0; I < Merges->Num(); ++I)
    {
        const FString P = FReader::Index(TEXT("$.events.merges"), I); FObject O; FCardCapMergeEvent E;
        if (!R.Object((*Merges)[I], O, P) || !R.IntegerField(O, TEXT("frame"), E.Frame, P, 0, LastFrame) ||
            !R.StringField(O, TEXT("result"), E.Result, P) ||
            !R.Require(PacketIds.Contains(E.Result), P + TEXT(".result"), TEXT("unknown packet id")) ||
            !R.Strings(R.Field(O, TEXT("sources"), P), E.Sources, P + TEXT(".sources"), true) ||
            !R.Require(E.Sources.Num() >= 2, P + TEXT(".sources"), TEXT("merge requires at least two sources"))) return false;
        for (int32 J = 0; J < E.Sources.Num(); ++J)
            if (!R.Require(PacketIds.Contains(E.Sources[J]) && E.Sources[J] != E.Result, FReader::Index(P + TEXT(".sources"), J), TEXT("unknown packet or self merge"))) return false;
        const FString Key = FString::Printf(TEXT("merge:%d:%s"), E.Frame, *E.Result);
        if (!R.Require(!SeenEvents.Contains(Key), P, TEXT("duplicate merge event"))) return false;
        SeenEvents.Add(Key); Out.Events.Merges.Add(MoveTemp(E));
    }
    for (int32 I = 0; I < Releases->Num(); ++I)
    {
        const FString P = FReader::Index(TEXT("$.events.releases"), I); FObject O; FCardCapReleaseEvent E;
        if (!R.Object((*Releases)[I], O, P) || !R.IntegerField(O, TEXT("frame"), E.Frame, P, 0, LastFrame) ||
            !R.StringField(O, TEXT("packet"), E.Packet, P) ||
            !R.Require(PacketIds.Contains(E.Packet), P + TEXT(".packet"), TEXT("unknown packet id"))) return false;
        TOptional<FVector> ReleaseVelocity;
        if (!R.NullableVector(R.Field(O, TEXT("release_velocity_cm_s"), P), ReleaseVelocity, P + TEXT(".release_velocity_cm_s"), bProvenance12)) return false;
        E.bHasReleaseVelocity = ReleaseVelocity.IsSet();
        if (E.bHasReleaseVelocity) E.ReleaseVelocityCmS = ReleaseVelocity.GetValue();
        const FString Key = FString::Printf(TEXT("release:%d:%s"), E.Frame, *E.Packet);
        if (!R.Require(!SeenEvents.Contains(Key), P, TEXT("duplicate release event"))) return false;
        SeenEvents.Add(Key); Out.Events.Releases.Add(MoveTemp(E));
    }
    if (!R.ObjectField(Root, TEXT("quality"), Quality, TEXT("$")) ||
        !R.Score(Quality, TEXT("mean_hand_confidence"), Out.Quality.MeanHandConfidence, TEXT("$.quality")) ||
        !R.ArrayField(Quality, TEXT("low_confidence_ranges"), Ranges, TEXT("$.quality")) ||
        !R.Strings(R.Field(Quality, TEXT("warnings"), TEXT("$.quality")), Out.Quality.Warnings, TEXT("$.quality.warnings"))) return false;
    FValue MeanError = R.Field(Quality, TEXT("mean_reprojection_error_px"), TEXT("$.quality"));
    if (!MeanError.IsValid()) return false;
    if (MeanError->Type != EJson::Null)
    {
        if (bSpatial13 && !Out.bInterHandTransformKnown)
            return R.Fail(TEXT("$.quality.mean_reprojection_error_px"), TEXT("unknown source camera and global translations cannot have a full-image 3D reprojection score; crop-model diagnostics must be separate"));
        double Number = 0;
        if (!R.Number(MeanError, Number, TEXT("$.quality.mean_reprojection_error_px")) ||
            !R.Require(Number >= 0, TEXT("$.quality.mean_reprojection_error_px"), TEXT("must be nonnegative or null when unmeasured"))) return false;
        Out.Quality.MeanReprojectionErrorPx = Number;
    }
    int32 PreviousEnd = -1;
    for (int32 I = 0; I < Ranges->Num(); ++I)
    {
        const FString P = FReader::Index(TEXT("$.quality.low_confidence_ranges"), I);
        const FArray* Range = nullptr; FIntPoint Pair = FIntPoint::ZeroValue;
        if (!R.Array((*Ranges)[I], Range, P, 2) || !R.Integer((*Range)[0], Pair.X, P + TEXT("[0]"), 0, LastFrame) ||
            !R.Integer((*Range)[1], Pair.Y, P + TEXT("[1]"), Pair.X, LastFrame) ||
            !R.Require(Pair.X > PreviousEnd, P, TEXT("ranges must be ordered and nonoverlapping"))) return false;
        PreviousEnd = Pair.Y; Out.Quality.LowConfidenceRanges.Add(Pair);
    }
    return Error.IsEmpty();
}
}

bool FCardCapJsonParser::LoadBoneMapping(const FString& Filename, FCardCapBoneMapping& OutMapping, FString& OutError)
{
    OutError.Reset(); FString Path = Filename;
    if (Path.IsEmpty())
    {
        const TSharedPtr<IPlugin> Plugin = IPluginManager::Get().FindPlugin(TEXT("CardistryCapture"));
        if (!Plugin.IsValid()) { OutError = TEXT("mapping: CardistryCapture plugin location is unavailable"); return false; }
        Path = FPaths::Combine(Plugin->GetBaseDir(), TEXT("Config/BoneMapping_UE5Mannequin.json"));
    }
    FString Json;
    if (!FFileHelper::LoadFileToString(Json, *Path)) { OutError = TEXT("mapping: cannot read ") + Path; return false; }
    CardCapJson::FObject Root; FCardCapBoneMapping Candidate;
    if (!CardCapJson::Deserialize(Json, Root, OutError) || !CardCapJson::ReadMapping(Root, Candidate, OutError)) return false;
    OutMapping = MoveTemp(Candidate); return true;
}

bool FCardCapJsonParser::ParseString(const FString& Json, FCardCapCapture& OutCapture, FString& OutError, const FString& BoneMappingPath)
{
    OutError.Reset(); FCardCapCapture Candidate; CardCapJson::FObject Root;
    if (!CardCapJson::Deserialize(Json, Root, OutError) ||
        !LoadBoneMapping(BoneMappingPath, Candidate.BoneMapping, OutError) ||
        !CardCapJson::ReadCapture(Root, Candidate, OutError)) return false;
    OutCapture = MoveTemp(Candidate); return true;
}

bool FCardCapJsonParser::ParseFile(const FString& Filename, FCardCapCapture& OutCapture, FString& OutError, const FString& BoneMappingPath)
{
    OutError.Reset(); FString Json;
    if (!FFileHelper::LoadFileToString(Json, *Filename)) { OutError = TEXT("$: cannot read ") + Filename; return false; }
    return ParseString(Json, OutCapture, OutError, BoneMappingPath);
}
