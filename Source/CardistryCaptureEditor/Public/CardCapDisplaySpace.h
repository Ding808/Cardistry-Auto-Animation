#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CardCapDisplaySpace.generated.h"

class ACardCapResearchHandsActor;
class ACardCapResearchCameraActor;
class ACameraActor;

/** Explicit viewing assumptions, kept separate from capture and animation data. */
struct CARDISTRYCAPTUREEDITOR_API FCardCapDisplaySpaceData
{
    FString ConfigPath, ConfigSha256, CaptureSha256;
    double AssumedFocalPx = 0, HandLength = 0;
    FIntPoint Resolution = FIntPoint::ZeroValue;
    FVector2D PrincipalPoint = FVector2D::ZeroVector;
    TArray<FVector> LeftConstant, LeftPerFocal, RightConstant, RightPerFocal;

    static bool Load(const FString& ConfigFile, const FString& CaptureFile, int32 FrameCount,
        double Fps, FIntPoint Resolution, FCardCapDisplaySpaceData& Out, FString& Error);
};

/** Display-only scene controller. Editing it never changes source K or animation keys. */
UCLASS()
class CARDISTRYCAPTUREEDITOR_API ACardCapDisplaySpaceActor : public AActor
{
    GENERATED_BODY()
public:
    ACardCapDisplaySpaceActor();
    UPROPERTY(EditAnywhere, Category="Common Space Preview", meta=(DisplayName="Assumed Focal Length (px)", ClampMin="0.001", ToolTip="Display assumption only. The source camera remains uncalibrated."))
    double DisplayAssumedFocalPx = 0;
    UPROPERTY(VisibleAnywhere, Category="Common Space Preview", meta=(DisplayName="Wrist Distance / Hand Length"))
    double WristDistanceOverHandLength = 0;
    UPROPERTY(VisibleAnywhere, Interp, Category="Common Space Preview", meta=(DisplayName="Display Frame"))
    float DisplayFrame = 0;
    UPROPERTY(EditAnywhere, Category="Common Space Preview", meta=(DisplayName="Show Local Hands"))
    bool bShowLocalHands = false;
    UPROPERTY(EditAnywhere, Category="Common Space Preview", meta=(DisplayName="Show Right Local Hand", EditCondition="bShowLocalHands"))
    bool bShowRightLocalHand = false;
    UPROPERTY(VisibleAnywhere, Category="Common Space Preview")
    FString DisplayNotice = TEXT("Common space uses an assumed focal length. Display only; uncalibrated. Source camera and physical scale remain unknown.");
    UPROPERTY(VisibleAnywhere, Category="Common Space Preview") FString DisplayConfigSha256;
    UPROPERTY() TObjectPtr<ACardCapResearchHandsActor> LeftHand;
    UPROPERTY() TObjectPtr<ACardCapResearchHandsActor> RightHand;
    UPROPERTY() TObjectPtr<ACardCapResearchCameraActor> DisplayCamera;
    UPROPERTY() TObjectPtr<ACameraActor> OverviewCamera;
    UPROPERTY() TArray<FVector> LeftConstant;
    UPROPERTY() TArray<FVector> LeftPerFocal;
    UPROPERTY() TArray<FVector> RightConstant;
    UPROPERTY() TArray<FVector> RightPerFocal;
    UPROPERTY() double HandLength = 0;
    UPROPERTY() FIntPoint Resolution = FIntPoint::ZeroValue;
    UPROPERTY() FVector2D PrincipalPoint = FVector2D::ZeroVector;
    UPROPERTY() FBox LeftBounds = FBox(ForceInit);
    UPROPERTY() FBox RightBounds = FBox(ForceInit);

    void Configure(const FCardCapDisplaySpaceData& Data);
    // Sequencer invokes native property setters through AActor::ProcessEvent;
    // editor review worlds require this permission before gameplay initializes actors.
    UFUNCTION(CallInEditor) void SetDisplayFrame(float Value);
    void ApplyDisplay();
    FVector HandTranslation(bool bRight, int32 Frame) const;
    virtual void Tick(float DeltaTime) override;
    virtual bool ShouldTickIfViewportsOnly() const override { return true; }
#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& Event) override;
#endif
private:
    double LastValidFocal = 0;
    double LastOverviewFocal = -1;
};
