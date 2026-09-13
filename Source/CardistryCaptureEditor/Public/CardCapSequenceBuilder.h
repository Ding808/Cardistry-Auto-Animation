#pragma once

#include "CoreMinimal.h"
#include "Animation/SkeletalMeshActor.h"
#include "Camera/CameraActor.h"
#include "Camera/CameraComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "CardCapSequenceBuilder.generated.h"

class UAnimSequence;
class ULevelSequence;
class ALevelSequenceActor;
class FJsonObject;

// These editor-only classes persist in local research maps. They are not a
// runtime/cooked content dependency or an end-user capture UI.
UCLASS()
class CARDISTRYCAPTUREEDITOR_API UCardCapResearchCameraComponent : public UCameraComponent
{
    GENERATED_BODY()
public:
    UPROPERTY() FVector2D PrincipalPointOffset = FVector2D::ZeroVector;
    virtual void GetCameraView(float DeltaTime, FMinimalViewInfo& DesiredView) override;
};

UCLASS()
class CARDISTRYCAPTUREEDITOR_API ACardCapResearchCameraActor : public ACameraActor
{
    GENERATED_BODY()
public:
    ACardCapResearchCameraActor(const FObjectInitializer& ObjectInitializer);
};

// Imported MANO meshes can have generated physics bodies, but their stationary
// shared root does not follow either wrist. Keep research preview bounds tied
// to both animated wrists, independent of reference/physics-body bounds quality.
UCLASS()
class CARDISTRYCAPTUREEDITOR_API UCardCapResearchSkeletalMeshComponent : public USkeletalMeshComponent
{
    GENERATED_BODY()
public:
    UPROPERTY() double BindVertexToBoneRadiusCm = 50.0;
    UPROPERTY() bool bSeparateLocalHands = false;
    UPROPERTY() TArray<int32> LocalLeftMaterialIds;
    UPROPERTY() TArray<int32> LocalRightMaterialIds;
    UPROPERTY(EditAnywhere, Category="Local Hand Preview", meta=(EditCondition="bSeparateLocalHands", DisplayName="Show right hand locally"))
    bool bShowRightLocalHand = false;
    void SetLocalHandView(bool bRight);
    virtual void OnRegister() override;
#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif
    virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;
};

UCLASS()
class CARDISTRYCAPTUREEDITOR_API ACardCapResearchHandsActor : public ASkeletalMeshActor
{
    GENERATED_BODY()
public:
    ACardCapResearchHandsActor(const FObjectInitializer& ObjectInitializer);
};

struct CARDISTRYCAPTUREEDITOR_API FCardCapSequenceBuildSettings
{
    FString PackageDirectory;
    FString AssetName = TEXT("Cardistry_Hands");
    FFrameRate DisplayRate = FFrameRate(30, 1);
    int32 FrameCount = 0;
    FIntPoint Resolution = FIntPoint(1280, 720);
    // Shared/legacy mode consumes the source projection. In 1.3 local mode these
    // fields remain unset; independent display-only viewers never become K.
    double Fx = 0, Fy = 0, Cx = 0, Cy = 0;
    bool bCalibrated = false;
    FString CameraSourceLabel; // Empty preserves the legacy 1.0/1.1 label.
    FString CoordinateUnits = TEXT("ue_centimeters");
    FString DistortionPolicy;
    bool bPerHandLocalPreview = false;
    FName LeftWristBone, RightWristBone;
    // Camera-local UE +X forward, +Y image-right, +Z image-up; units cm.
    // The exported capture already uses this basis, so both default to identity.
    FTransform CameraTransform = FTransform::Identity;
    FTransform MeshTransform = FTransform::Identity;
    float NearClipCm = 0.1f;
};

struct CARDISTRYCAPTUREEDITOR_API FCardCapSequenceBuildResult
{
    UWorld* World = nullptr;
    ULevelSequence* Sequence = nullptr;
    ACardCapResearchHandsActor* HandsActor = nullptr;
    ACardCapResearchCameraActor* CameraActor = nullptr;
    ACameraActor* OverviewCameraActor = nullptr;
    ALevelSequenceActor* SequenceActor = nullptr;
    FGuid HandsBinding;
    FGuid CameraBinding;
    FString MapPackageName;
    FString SequencePackageName;
    FString SkinMaterialPackageName;
    FBox AnimatedGeometryBounds = FBox(ForceInit);
    int32 GeometryVerticesPerFrame = 0;
    FBox LeftLocalBounds = FBox(ForceInit), RightLocalBounds = FBox(ForceInit);
    int32 LeftGeometryVerticesPerFrame = 0, RightGeometryVerticesPerFrame = 0;
    TArray<int32> LeftMaterialIds, RightMaterialIds;
};

class CARDISTRYCAPTUREEDITOR_API FCardCapSequenceBuilder
{
public:
    // New packages only, under /Game/CardistryCapture/Generated/. Leaves a
    // rooted, isolated editor world alive for CaptureFrames; call ReleaseWorld.
    // Square-pixel perspective K is supported by the saved Sequencer camera.
    // Rejects Fx != Fy rather than silently changing the saved camera projection.
    static bool Build(USkeletalMesh* Mesh, UAnimSequence* Animation,
        const FCardCapSequenceBuildSettings& Settings,
        FCardCapSequenceBuildResult& Out, FString& OutError);

    // Real LevelSequencePlayer evaluation, followed by GPU SceneCapture2D.
    // Shared mode writes capture/overview images; local mode uses the same paths
    // for separately isolated left/right hand views, labeled in the report.
    // Run a separate editor commandlet with -AllowCommandletRendering
    // -RenderOffscreen -unattended and commandlet IsClient=true; never -NullRHI.
    // UE otherwise allocates a FNULLSceneInterface even with an initialized RHI.
    // No desktop capture required.
    static bool CaptureFrames(const FCardCapSequenceBuildResult& Built,
        const FCardCapSequenceBuildSettings& Settings, const FString& OutputDirectory,
        TSharedRef<FJsonObject> Report, FString& OutError);

    static void ReleaseWorld(FCardCapSequenceBuildResult& Built);
};
