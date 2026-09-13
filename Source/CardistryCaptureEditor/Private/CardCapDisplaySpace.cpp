#include "CardCapDisplaySpace.h"
#include "CardCapSequenceBuilder.h"
#include "Components/SceneComponent.h"
#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "openssl/sha.h"

namespace
{
bool Fail(FString& Error, const TCHAR* Message) { Error = Message; return false; }
bool HashFile(const FString& Path, FString& Hash)
{
    TArray<uint8> Bytes;
    if (!FFileHelper::LoadFileToArray(Bytes, *Path)) return false;
    uint8 Digest[SHA256_DIGEST_LENGTH];
    if (!SHA256(Bytes.GetData(), Bytes.Num(), Digest)) return false;
    Hash = BytesToHex(Digest, SHA256_DIGEST_LENGTH).ToLower();
    return true;
}
bool Number(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, double& Value)
{ return Object && Object->TryGetNumberField(Key, Value) && FMath::IsFinite(Value); }
bool Vector(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, FVector& Value)
{
    const TArray<TSharedPtr<FJsonValue>>* Array = nullptr;
    if (!Object || !Object->TryGetArrayField(Key, Array) || Array->Num() != 3) return false;
    return (*Array)[0]->TryGetNumber(Value.X) && (*Array)[1]->TryGetNumber(Value.Y)
        && (*Array)[2]->TryGetNumber(Value.Z) && !Value.ContainsNaN();
}
void FrameBounds(ACameraActor* Camera, const FBox& Bounds, FIntPoint Resolution)
{
    if (!Camera || !Bounds.IsValid || Resolution.X < 1 || Resolution.Y < 1) return;
    UCameraComponent* View = Camera->GetCameraComponent();
    View->SetFieldOfView(60.f);
    View->SetAspectRatio(double(Resolution.X) / Resolution.Y);
    View->bConstrainAspectRatio = true;
    const double HalfAngle = FMath::Min(FMath::DegreesToRadians(30.), FMath::Atan(FMath::Tan(FMath::DegreesToRadians(30.)) / View->AspectRatio));
    const double Distance = Bounds.GetExtent().Size() / FMath::Sin(HalfAngle) * 1.15;
    const FVector Location = Bounds.GetCenter() + FVector(-1., -1., .65).GetSafeNormal() * Distance;
    Camera->SetActorLocation(Location);
    Camera->SetActorRotation(FRotationMatrix::MakeFromX(Bounds.GetCenter() - Location).Rotator());
}
}

bool FCardCapDisplaySpaceData::Load(const FString& ConfigFile, const FString& CaptureFile, int32 FrameCount,
    double Fps, FIntPoint ExpectedResolution, FCardCapDisplaySpaceData& Out, FString& Error)
{
    FCardCapDisplaySpaceData Next;
    FString Json, Version, Mode, Basis, Units, BoundCaptureHash;
    TSharedPtr<FJsonObject> Root;
    bool bDisplayOnly = false;
    double Count = 0, Rate = 0;
    if (!FFileHelper::LoadFileToString(Json, *ConfigFile)
        || !FJsonSerializer::Deserialize(TJsonReaderFactory<>::Create(Json), Root) || !Root)
        return Fail(Error, TEXT("The display configuration is not readable JSON."));
    if (!Root->TryGetStringField(TEXT("format_version"), Version) || Version != TEXT("cardcap.display_space/1.0")
        || !Root->TryGetBoolField(TEXT("display_only"), bDisplayOnly) || !bDisplayOnly
        || !Root->TryGetStringField(TEXT("mode"), Mode) || Mode != TEXT("assumed_common_camera")
        || !Root->TryGetStringField(TEXT("coordinate_basis"), Basis) || Basis != TEXT("ue_x_forward_y_right_z_up")
        || !Root->TryGetStringField(TEXT("coordinate_units"), Units) || Units != TEXT("conditional_ue_display_units")
        || !Number(Root, TEXT("frame_count"), Count) || Count != FrameCount
        || !Number(Root, TEXT("fps"), Rate) || !FMath::IsNearlyEqual(Rate, Fps, 1.e-6)
        || !Number(Root, TEXT("display_assumed_focal_px"), Next.AssumedFocalPx) || Next.AssumedFocalPx <= 0
        || !Number(Root, TEXT("neutral_hand_length_display_units"), Next.HandLength) || Next.HandLength <= 0)
        return Fail(Error, TEXT("Display configuration must declare compatible display-only assumptions, frame timing, focal length and hand length."));
    const TArray<TSharedPtr<FJsonValue>>* ResolutionValues = nullptr;
    const TArray<TSharedPtr<FJsonValue>>* Principal = nullptr;
    double W = 0, H = 0;
    if (!Root->TryGetArrayField(TEXT("resolution"), ResolutionValues) || ResolutionValues->Num() != 2
        || !(*ResolutionValues)[0]->TryGetNumber(W) || W != ExpectedResolution.X
        || !(*ResolutionValues)[1]->TryGetNumber(H) || H != ExpectedResolution.Y
        || !Root->TryGetArrayField(TEXT("principal_point_px"), Principal) || Principal->Num() != 2
        || !(*Principal)[0]->TryGetNumber(Next.PrincipalPoint.X) || !(*Principal)[1]->TryGetNumber(Next.PrincipalPoint.Y)
        || !FMath::IsFinite(Next.PrincipalPoint.X) || !FMath::IsFinite(Next.PrincipalPoint.Y))
        return Fail(Error, TEXT("Display resolution or principal point does not match the source."));
    Next.Resolution = ExpectedResolution;
    if (!Root->TryGetStringField(TEXT("capture_sha256"), BoundCaptureHash)
        || !HashFile(CaptureFile, Next.CaptureSha256) || Next.CaptureSha256 != BoundCaptureHash
        || !HashFile(ConfigFile, Next.ConfigSha256))
        return Fail(Error, TEXT("The display configuration is not bound to this exact capture file."));
    Next.ConfigPath = FPaths::ConvertRelativePathToFull(ConfigFile);
    const TArray<TSharedPtr<FJsonValue>>* Frames = nullptr;
    if (!Root->TryGetArrayField(TEXT("frames"), Frames) || Frames->Num() != FrameCount)
        return Fail(Error, TEXT("Display configuration requires one entry for every source frame."));
    for (int32 Frame = 0; Frame < FrameCount; ++Frame)
    {
        const TSharedPtr<FJsonObject>* FrameObject = nullptr;
        if (!(*Frames)[Frame]->TryGetObject(FrameObject) || !FrameObject || !*FrameObject)
            return Fail(Error, TEXT("Each display frame must be a JSON object."));
        const TSharedPtr<FJsonObject> Item = *FrameObject;
        const TSharedPtr<FJsonObject>* Hands = nullptr;
        double Index = -1, Ratio = -1;
        if (!Number(Item, TEXT("frame"), Index) || Index != Frame
            || !Number(Item, TEXT("wrist_distance_over_hand_length"), Ratio) || Ratio < 0
            || !Item->TryGetObjectField(TEXT("hands"), Hands) || !Hands || !*Hands)
            return Fail(Error, TEXT("Display frame indices, hand entries or wrist ratios are invalid."));
        FVector Translations[2];
        for (int32 Side = 0; Side < 2; ++Side)
        {
            const TSharedPtr<FJsonObject>* Hand = nullptr;
            FVector Constant, PerFocal;
            if (!(*Hands)->TryGetObjectField(Side ? TEXT("right") : TEXT("left"), Hand) || !Hand || !*Hand
                || !Vector(*Hand, TEXT("wrist_translation_constant_display_units"), Constant)
                || !Vector(*Hand, TEXT("wrist_translation_per_focal_px_display_units"), PerFocal)
                || !Vector(*Hand, TEXT("wrist_translation_display_units"), Translations[Side])
                || !Translations[Side].Equals(Constant + Next.AssumedFocalPx * PerFocal, 1.e-5))
                return Fail(Error, TEXT("Display hand translations do not match their explicit focal coefficients."));
            (Side ? Next.RightConstant : Next.LeftConstant).Add(Constant);
            (Side ? Next.RightPerFocal : Next.LeftPerFocal).Add(PerFocal);
        }
        if (!FMath::IsNearlyEqual(FVector::Distance(Translations[0], Translations[1]) / Next.HandLength, Ratio, 1.e-6))
            return Fail(Error, TEXT("Display wrist distance does not match the declared hand length and translations."));
    }
    Out = MoveTemp(Next);
    Error.Reset();
    return true;
}

ACardCapDisplaySpaceActor::ACardCapDisplaySpaceActor()
{
    RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("DisplayRoot"));
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.bStartWithTickEnabled = true;
}

void ACardCapDisplaySpaceActor::Configure(const FCardCapDisplaySpaceData& Data)
{
    DisplayAssumedFocalPx = Data.AssumedFocalPx;
    LastValidFocal = DisplayAssumedFocalPx;
    HandLength = Data.HandLength;
    Resolution = Data.Resolution;
    PrincipalPoint = Data.PrincipalPoint;
    LeftConstant = Data.LeftConstant; LeftPerFocal = Data.LeftPerFocal;
    RightConstant = Data.RightConstant; RightPerFocal = Data.RightPerFocal;
    DisplayConfigSha256 = Data.ConfigSha256;
    ApplyDisplay();
}
FVector ACardCapDisplaySpaceActor::HandTranslation(bool bRight, int32 Frame) const
{
    const auto& Constant = bRight ? RightConstant : LeftConstant;
    const auto& PerFocal = bRight ? RightPerFocal : LeftPerFocal;
    return Constant.IsValidIndex(Frame) && PerFocal.IsValidIndex(Frame)
        ? Constant[Frame] + DisplayAssumedFocalPx * PerFocal[Frame] : FVector::ZeroVector;
}
void ACardCapDisplaySpaceActor::SetDisplayFrame(float Value)
{
    if (FMath::IsFinite(Value)) DisplayFrame = Value;
    ApplyDisplay();
}
void ACardCapDisplaySpaceActor::ApplyDisplay()
{
    if (!FMath::IsFinite(DisplayAssumedFocalPx) || DisplayAssumedFocalPx <= 0)
        DisplayAssumedFocalPx = LastValidFocal;
    if (!LeftHand || !RightHand || !DisplayCamera || !OverviewCamera || LeftConstant.IsEmpty()
        || DisplayAssumedFocalPx <= 0 || HandLength <= 0 || Resolution.X < 1 || Resolution.Y < 1) return;
    LastValidFocal = DisplayAssumedFocalPx;
    const int32 Frame = FMath::Clamp(FMath::FloorToInt(DisplayFrame), 0, LeftConstant.Num() - 1);
    const FVector Left = HandTranslation(false, Frame), Right = HandTranslation(true, Frame);
    WristDistanceOverHandLength = FVector::Distance(Left, Right) / HandLength;
    LeftHand->SetActorLocation(bShowLocalHands ? FVector::ZeroVector : Left);
    RightHand->SetActorLocation(bShowLocalHands ? FVector::ZeroVector : Right);
    LeftHand->GetSkeletalMeshComponent()->SetVisibility(!bShowLocalHands || !bShowRightLocalHand);
    RightHand->GetSkeletalMeshComponent()->SetVisibility(!bShowLocalHands || bShowRightLocalHand);
    auto* View = CastChecked<UCardCapResearchCameraComponent>(DisplayCamera->GetCameraComponent());
    if (bShowLocalHands)
    {
        FrameBounds(DisplayCamera, bShowRightLocalHand ? RightBounds : LeftBounds, Resolution);
        View->PrincipalPointOffset = FVector2D::ZeroVector;
        DisplayCamera->SetActorLabel(TEXT("Common display camera (optional local hand view; display only)"), false);
    }
    else
    {
        DisplayCamera->SetActorTransform(FTransform::Identity);
        View->SetFieldOfView(FMath::RadiansToDegrees(2. * FMath::Atan(Resolution.X / (2. * DisplayAssumedFocalPx))));
        View->SetAspectRatio(double(Resolution.X) / Resolution.Y);
        View->bConstrainAspectRatio = true;
        View->PrincipalPointOffset = FVector2D(1. - 2. * PrincipalPoint.X / Resolution.X, 2. * PrincipalPoint.Y / Resolution.Y - 1.);
        DisplayCamera->SetActorLabel(FString::Printf(TEXT("Common display camera (assumed %.3f px; display only; uncalibrated)"), DisplayAssumedFocalPx), false);
    }
    if (LastOverviewFocal != DisplayAssumedFocalPx)
    {
        FBox CommonBounds(ForceInit);
        for (int32 Index = 0; Index < LeftConstant.Num(); ++Index)
        {
            CommonBounds += LeftBounds.ShiftBy(HandTranslation(false, Index));
            CommonBounds += RightBounds.ShiftBy(HandTranslation(true, Index));
        }
        FrameBounds(OverviewCamera, CommonBounds, Resolution);
        LastOverviewFocal = DisplayAssumedFocalPx;
    }
}
void ACardCapDisplaySpaceActor::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);
    ApplyDisplay();
}
#if WITH_EDITOR
void ACardCapDisplaySpaceActor::PostEditChangeProperty(FPropertyChangedEvent& Event)
{
    Super::PostEditChangeProperty(Event);
    ApplyDisplay();
}
#endif
