#include "CardCapSequenceBuilder.h"

#include "Animation/AnimSequence.h"
#include "Animation/Skeleton.h"
#include "AssetCompilingManager.h"
#include "AssetRegistry/AssetRegistryModule.h"
#include "Camera/CameraTypes.h"
#include "Components/LightComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Dom/JsonObject.h"
#include "Engine/DirectionalLight.h"
#include "Engine/Engine.h"
#include "Engine/SceneCapture2D.h"
#include "Engine/SkeletalMesh.h"
#include "Engine/SkinnedAssetCommon.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "FileHelpers.h"
#include "HAL/FileManager.h"
#include "ImageUtils.h"
#include "LevelSequence.h"
#include "LevelSequenceActor.h"
#include "LevelSequencePlayer.h"
#include "Misc/App.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Materials/Material.h"
#include "MaterialDomain.h"
#include "MaterialShared.h"
#include "MaterialShaderPrecompileMode.h"
#include "Materials/MaterialExpressionConstant.h"
#include "Materials/MaterialExpressionConstant3Vector.h"
#include "Materials/MaterialExpressionMultiply.h"
#include "MovieScene.h"
#include "MovieSceneObjectBindingID.h"
#include "MovieSceneSequencePlayer.h"
#include "RenderingThread.h"
#include "Rendering/SkeletalMeshModel.h"
#include "RHI.h"
#include "RHIGlobals.h"
#include "SceneInterface.h"
#include "ShaderCompiler.h"
#include "Sections/MovieSceneCameraCutSection.h"
#include "Sections/MovieSceneSkeletalAnimationSection.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "TextureResource.h"
#include "Tracks/MovieSceneCameraCutTrack.h"
#include "Tracks/MovieSceneSkeletalAnimationTrack.h"
#include "UObject/Package.h"
#include "UObject/SavePackage.h"
#include "UObject/StrongObjectPtr.h"

namespace
{
    // Presentation lighting only. The single key light otherwise makes backlit
    // skin indistinguishable from the black background and suggests mesh gaps.
    constexpr float LocalInspectionEmissiveFraction = 0.12f;
    constexpr uint8 LocalSupportThreshold = 32;
    constexpr double MinimumLocalLitSupportFraction = 0.99;

    bool Fail(FString& Error, const FString& Message) { Error = Message; return false; }

    void SetFixedExposure(FPostProcessSettings& P)
    {
        P.bOverride_AutoExposureMethod = true;
        P.AutoExposureMethod = AEM_Manual;
        P.bOverride_AutoExposureApplyPhysicalCameraExposure = true;
        P.AutoExposureApplyPhysicalCameraExposure = false;
        P.bOverride_AutoExposureBias = true;
        P.AutoExposureBias = 0.0f;
        P.bOverride_MotionBlurAmount = true;
        P.MotionBlurAmount = 0.0f;
        P.bOverride_BloomIntensity = true;
        P.BloomIntensity = 0.0f;
    }

    FMatrix IntrinsicsProjection(const FCardCapSequenceBuildSettings& S)
    {
        // Engine camera view basis is right/up/forward before this matrix.
        // Infinite far plane, reversed Z. Projected pixel x=fx*X/Z+cx,
        // y=fy*Ydown/Z+cy; row-vector matrix convention used by UE.
        return FMatrix(
            FPlane(2.0 * S.Fx / S.Resolution.X, 0, 0, 0),
            FPlane(0, 2.0 * S.Fy / S.Resolution.Y, 0, 0),
            FPlane(2.0 * S.Cx / S.Resolution.X - 1.0,
                1.0 - 2.0 * S.Cy / S.Resolution.Y, 0, 1),
            FPlane(0, 0, S.NearClipCm, 0));
    }

    TArray<TSharedPtr<FJsonValue>> VectorValues(const FVector& V)
    {
        return {MakeShared<FJsonValueNumber>(V.X), MakeShared<FJsonValueNumber>(V.Y), MakeShared<FJsonValueNumber>(V.Z)};
    }

    // Framing only: evaluate LOD0's actual imported skin weights at every source
    // frame. Neither the mesh nor its actor is translated to produce this view.
    bool AnimatedGeometryBounds(USkeletalMesh* Mesh, UAnimSequence* Animation,
        const FCardCapSequenceBuildSettings& S, FBox& Box, int32& VertexCount, FString& Error,
        const TSet<int32>* IncludedSections = nullptr)
    {
        const FSkeletalMeshModel* Model = Mesh->GetImportedModel();
        const FReferenceSkeleton& Ref = Mesh->GetRefSkeleton();
        const FReferenceSkeleton& SkeletonRef = Animation->GetSkeleton()->GetReferenceSkeleton();
        const TArray<FMatrix44f>& InverseBind = Mesh->GetRefBasesInvMatrix();
        if (!Model || Model->LODModels.IsEmpty() || InverseBind.Num() != Ref.GetNum())
            return Fail(Error, TEXT("Overview framing requires imported LOD0 skin vertices and inverse bind matrices."));
        VertexCount = 0;
        for (int32 SectionIndex = 0; SectionIndex < Model->LODModels[0].Sections.Num(); ++SectionIndex)
            if (!IncludedSections || IncludedSections->Contains(SectionIndex))
                VertexCount += Model->LODModels[0].Sections[SectionIndex].SoftVertices.Num();
        if (VertexCount == 0) return Fail(Error, TEXT("Overview framing found no imported LOD0 vertices."));
        Box = FBox(ForceInit);
        for (int32 Frame = 0; Frame < S.FrameCount; ++Frame)
        {
            TArray<FTransform> Components;
            Components.SetNum(Ref.GetNum());
            for (int32 Bone = 0; Bone < Ref.GetNum(); ++Bone)
            {
                const int32 SkeletonBone = SkeletonRef.FindBoneIndex(Ref.GetBoneName(Bone));
                if (SkeletonBone == INDEX_NONE) return Fail(Error, TEXT("Overview mesh bone is absent from animation skeleton."));
                FTransform Local;
                Animation->GetBoneTransform(Local, FSkeletonPoseBoneIndex(SkeletonBone), Frame / S.DisplayRate.AsDecimal(), true);
                if (Local.ContainsNaN()) return Fail(Error, TEXT("Overview animation contains a non-finite bone transform."));
                const int32 Parent = Ref.GetParentIndex(Bone);
                Components[Bone] = Parent < 0 ? Local : Local * Components[Parent];
            }
            for (int32 SectionIndex = 0; SectionIndex < Model->LODModels[0].Sections.Num(); ++SectionIndex)
            {
                if (IncludedSections && !IncludedSections->Contains(SectionIndex)) continue;
                const FSkelMeshSection& Section = Model->LODModels[0].Sections[SectionIndex];
                for (const FSoftSkinVertex& Vertex : Section.SoftVertices)
                {
                    FVector Position = FVector::ZeroVector;
                    uint32 TotalWeight = 0;
                    for (int32 Influence = 0; Influence < MAX_TOTAL_INFLUENCES; ++Influence)
                    {
                        const uint16 Weight = Vertex.InfluenceWeights[Influence];
                        if (!Weight) continue;
                        const int32 SectionBone = Vertex.InfluenceBones[Influence];
                        if (!Section.BoneMap.IsValidIndex(SectionBone)) return Fail(Error, TEXT("Invalid skin section bone index during framing."));
                        const int32 Bone = Section.BoneMap[SectionBone];
                        if (!Components.IsValidIndex(Bone)) return Fail(Error, TEXT("Invalid mesh bone index during framing."));
                        const FVector BindLocal(InverseBind[Bone].TransformPosition(Vertex.Position));
                        Position += Components[Bone].TransformPosition(BindLocal) * Weight;
                        TotalWeight += Weight;
                    }
                    if (!TotalWeight) return Fail(Error, TEXT("Unweighted imported skin vertex during framing."));
                    Position /= TotalWeight;
                    if (Position.ContainsNaN()) return Fail(Error, TEXT("Non-finite skinned position during overview framing."));
                    Box += Position;
                }
            }
        }
        return Box.IsValid && Box.GetExtent().Size() > UE_SMALL_NUMBER;
    }

    // Derive the visible side from actual positive skin weights and the supplied
    // wrist mapping. Never assume that importer material slot 0 means left.
    bool ClassifyHandSections(USkeletalMesh* Mesh, FName LeftWrist, FName RightWrist,
        TArray<int32>& LeftMaterials, TArray<int32>& RightMaterials,
        TSet<int32>& LeftLOD0Sections, TSet<int32>& RightLOD0Sections, FString& Error)
    {
        const FSkeletalMeshModel* Model = Mesh->GetImportedModel();
        const FReferenceSkeleton& Ref = Mesh->GetRefSkeleton();
        const int32 Left = Ref.FindBoneIndex(LeftWrist), Right = Ref.FindBoneIndex(RightWrist);
        if (!Model || Model->LODModels.IsEmpty() || Left == INDEX_NONE || Right == INDEX_NONE || Left == Right)
            return Fail(Error, TEXT("Local hand preview requires actual left/right wrist mappings and imported skin sections."));
        TArray<int32> BoneSide; BoneSide.Init(INDEX_NONE, Ref.GetNum());
        for (int32 Bone = 0; Bone < Ref.GetNum(); ++Bone)
        {
            for (int32 Ancestor = Bone; Ancestor != INDEX_NONE; Ancestor = Ref.GetParentIndex(Ancestor))
            {
                if (Ancestor == Left) { BoneSide[Bone] = 0; break; }
                if (Ancestor == Right) { BoneSide[Bone] = 1; break; }
            }
        }
        TMap<int32, int32> MaterialOwners;
        for (int32 LOD = 0; LOD < Model->LODModels.Num(); ++LOD)
        {
            const FSkeletalMeshLODInfo* LODInfo = Mesh->GetLODInfo(LOD);
            for (int32 SectionIndex = 0; SectionIndex < Model->LODModels[LOD].Sections.Num(); ++SectionIndex)
            {
                const FSkelMeshSection& Section = Model->LODModels[LOD].Sections[SectionIndex];
                int32 Side = INDEX_NONE;
                for (const FSoftSkinVertex& Vertex : Section.SoftVertices)
                    for (int32 Influence = 0; Influence < MAX_TOTAL_INFLUENCES; ++Influence)
                    {
                        if (!Vertex.InfluenceWeights[Influence]) continue;
                        const int32 SectionBone = Vertex.InfluenceBones[Influence];
                        if (!Section.BoneMap.IsValidIndex(SectionBone) || !BoneSide.IsValidIndex(Section.BoneMap[SectionBone]))
                            return Fail(Error, TEXT("Invalid skin weight bone during local-hand section classification."));
                        const int32 Owner = BoneSide[Section.BoneMap[SectionBone]];
                        if (Owner == INDEX_NONE || (Side != INDEX_NONE && Side != Owner))
                            return Fail(Error, TEXT("A mesh section has mixed or unmapped hand influences; local visibility cannot safely separate hands."));
                        Side = Owner;
                    }
                if (Side == INDEX_NONE) continue;
                const int32 Material = LODInfo && LODInfo->LODMaterialMap.IsValidIndex(SectionIndex)
                    && LODInfo->LODMaterialMap[SectionIndex] != INDEX_NONE ? LODInfo->LODMaterialMap[SectionIndex] : Section.MaterialIndex;
                if (!Mesh->GetMaterials().IsValidIndex(Material)) return Fail(Error, TEXT("Local section references an invalid material slot."));
                const int32* Previous = MaterialOwners.Find(Material);
                if (Previous && *Previous != Side) return Fail(Error, TEXT("Both hand sides share a material slot; refusing to render a false combined local space."));
                MaterialOwners.Add(Material, Side);
                if (Side == 0) { LeftMaterials.AddUnique(Material); if (LOD == 0) LeftLOD0Sections.Add(SectionIndex); }
                else { RightMaterials.AddUnique(Material); if (LOD == 0) RightLOD0Sections.Add(SectionIndex); }
            }
        }
        return (!LeftLOD0Sections.IsEmpty() && !RightLOD0Sections.IsEmpty())
            || Fail(Error, TEXT("Both local hand sides need separately renderable skin sections."));
    }

    void FrameLocalHand(ACameraActor* Actor, const FBox& Bounds, FIntPoint Resolution)
    {
        auto* Camera = Actor->GetCameraComponent();
        // Display-only framing: ordinary 60 degree perspective, a per-hand
        // all-frame bounding sphere and presentation margin. No source K/Z.
        Camera->SetFieldOfView(60.0f);
        Camera->SetAspectRatio(double(Resolution.X) / Resolution.Y);
        Camera->bConstrainAspectRatio = true;
        const double HalfHorizontal = FMath::DegreesToRadians(30.0);
        const double HalfVertical = FMath::Atan(FMath::Tan(HalfHorizontal) / Camera->AspectRatio);
        const double Distance = Bounds.GetExtent().Size() / FMath::Sin(FMath::Min(HalfHorizontal, HalfVertical)) * 1.15;
        const FVector Center = Bounds.GetCenter();
        const FVector Location = Center + FVector(-1., -1., .65).GetSafeNormal() * Distance;
        Actor->SetActorLocation(Location);
        Actor->SetActorRotation(FRotationMatrix::MakeFromX(Center - Location).Rotator());
        SetFixedExposure(Camera->PostProcessSettings);
    }

    UMaterial* CreateSkinMaterial(const FString& PackageName, bool bLocalInspection)
    {
        UPackage* Package = CreatePackage(*PackageName);
        UMaterial* Material = NewObject<UMaterial>(Package, *FPackageName::GetLongPackageAssetName(PackageName), RF_Public | RF_Standalone);
        Material->MaterialDomain = MD_Surface;
        Material->BlendMode = BLEND_Opaque;
        Material->TwoSided = true;
        Material->SetShadingModel(MSM_DefaultLit);
        auto* Color = NewObject<UMaterialExpressionConstant3Vector>(Material);
        Color->Material = Material;
        Color->Constant = FLinearColor(0.55f, 0.32f, 0.20f);
        Material->GetExpressionCollection().AddExpression(Color);
        Material->GetEditorOnlyData()->BaseColor.Connect(0, Color);
        if (bLocalInspection)
        {
            auto* Fill = NewObject<UMaterialExpressionMultiply>(Material);
            Fill->Material = Material;
            Fill->A.Connect(0, Color);
            Fill->ConstB = LocalInspectionEmissiveFraction;
            Material->GetExpressionCollection().AddExpression(Fill);
            Material->GetEditorOnlyData()->EmissiveColor.Connect(0, Fill);
        }
        auto* Roughness = NewObject<UMaterialExpressionConstant>(Material);
        Roughness->Material = Material;
        Roughness->R = 0.8f;
        Material->GetExpressionCollection().AddExpression(Roughness);
        Material->GetEditorOnlyData()->Roughness.Connect(0, Roughness);
        auto* Specular = NewObject<UMaterialExpressionConstant>(Material);
        Specular->Material = Material;
        Specular->R = 0.2f;
        Material->GetExpressionCollection().AddExpression(Specular);
        Material->GetEditorOnlyData()->Specular.Connect(0, Specular);
        bool bNeedsRecompile = false;
        Material->SetMaterialUsage(bNeedsRecompile, MATUSAGE_SkeletalMesh);
        Material->PostEditChange();
        // UE 5.4 PostEditChange requests PrecompileMode::None. A batch renderer
        // must explicitly compile the new graph's render permutations instead
        // of accepting an existing but incomplete shader map and grey fallback.
        Material->ForceRecompileForRendering(EMaterialShaderPrecompileMode::Synchronous);
        FAssetRegistryModule::AssetCreated(Material);
        Package->MarkPackageDirty();
        return Material;
    }
}

void UCardCapResearchCameraComponent::GetCameraView(float DeltaTime, FMinimalViewInfo& DesiredView)
{
    Super::GetCameraView(DeltaTime, DesiredView);
    DesiredView.OffCenterProjectionOffset = PrincipalPointOffset;
}

ACardCapResearchCameraActor::ACardCapResearchCameraActor(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer.SetDefaultSubobjectClass<UCardCapResearchCameraComponent>(TEXT("CameraComponent")))
{}

ACardCapResearchHandsActor::ACardCapResearchHandsActor(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer.SetDefaultSubobjectClass<UCardCapResearchSkeletalMeshComponent>(TEXT("SkeletalMeshComponent0")))
{}

void UCardCapResearchSkeletalMeshComponent::SetLocalHandView(bool bRight)
{
    bShowRightLocalHand = bRight;
    if (!bSeparateLocalHands || !GetSkeletalMeshAsset()) return;
    const TArray<int32>& Visible = bRight ? LocalRightMaterialIds : LocalLeftMaterialIds;
    for (int32 LOD = 0; LOD < GetNumLODs(); ++LOD)
        for (int32 Material = 0; Material < GetSkeletalMeshAsset()->GetMaterials().Num(); ++Material)
            ShowMaterialSection(Material, INDEX_NONE, Visible.Contains(Material), LOD);
}

void UCardCapResearchSkeletalMeshComponent::OnRegister()
{
    Super::OnRegister();
    // Engine LODInfo/HiddenMaterials are transient. Persist our explicit local
    // side and reapply section visibility after map reload; no bone transforms.
    SetLocalHandView(bShowRightLocalHand);
}

#if WITH_EDITOR
void UCardCapResearchSkeletalMeshComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);
    SetLocalHandView(bShowRightLocalHand);
}
#endif

FBoxSphereBounds UCardCapResearchSkeletalMeshComponent::CalcBounds(const FTransform& LocalToWorld) const
{
    const TArray<FTransform>& Bones = GetComponentSpaceTransforms();
    if (Bones.IsEmpty()) return Super::CalcBounds(LocalToWorld);
    FBox Box(ForceInit);
    double MaxScale = 1.0;
    for (const FTransform& Bone : Bones)
    {
        Box += Bone.GetTranslation();
        MaxScale = FMath::Max(MaxScale, Bone.GetScale3D().GetAbsMax());
    }
    // Every normalized LBS vertex is a convex combination of transformed bind
    // vertices. This radius bounds each bind-vertex-to-joint distance. It is
    // conservative but follows BOTH wrists without modifying the imported mesh.
    return FBoxSphereBounds(Box.ExpandBy(BindVertexToBoneRadiusCm * MaxScale)).TransformBy(LocalToWorld);
}

bool FCardCapSequenceBuilder::Build(USkeletalMesh* Mesh, UAnimSequence* Animation,
    const FCardCapSequenceBuildSettings& S, FCardCapSequenceBuildResult& Out, FString& Error)
{
    Out = FCardCapSequenceBuildResult();
    Error.Reset();
    if (!IsRunningCommandlet()) return Fail(Error, TEXT("Sequence building is restricted to the separate batch commandlet."));
    if (!GIsClient) return Fail(Error, TEXT("The batch commandlet must set IsClient=true: UE AllocateScene otherwise creates FNULLSceneInterface even with -AllowCommandletRendering."));
    if (!Mesh || !Animation || Mesh->GetSkeleton() != Animation->GetSkeleton())
        return Fail(Error, TEXT("A real mesh and animation sharing the same skeleton are required."));
    if (!S.MeshTransform.Equals(FTransform::Identity))
        return Fail(Error, TEXT("Research hands must retain identity actor/component transforms; correct camera-space data upstream."));
    if (S.FrameCount < 1 || S.DisplayRate.Numerator <= 0 || S.DisplayRate.Denominator <= 0
        || S.Resolution.X < 1 || S.Resolution.Y < 1
        || (!S.bPerHandLocalPreview && (S.Fx <= 0 || S.Fy <= 0 || !FMath::IsFinite(S.Fx) || !FMath::IsFinite(S.Fy) || !FMath::IsFinite(S.Cx) || !FMath::IsFinite(S.Cy)))
        || S.NearClipCm <= 0)
        return Fail(Error, TEXT("Invalid frame range, resolution, or capture intrinsics."));
    if (!S.bPerHandLocalPreview && !FMath::IsNearlyEqual(S.Fx, S.Fy, FMath::Max(S.Fx, S.Fy) * 1.e-6))
        return Fail(Error, TEXT("The saved Sequencer camera requires square-pixel intrinsics (fx=fy). Non-square pixels require a projection extension, which is not implemented."));
    const double Duration = S.FrameCount / S.DisplayRate.AsDecimal();
    if (Animation->GetPlayLength() + 1.e-4 < Duration)
        return Fail(Error, TEXT("Animation is shorter than the requested sequence frame interval."));
    Out.MapPackageName = S.PackageDirectory / (S.AssetName + TEXT("_Map"));
    Out.SequencePackageName = S.PackageDirectory / (S.AssetName + TEXT("_Sequence"));
    Out.SkinMaterialPackageName = S.PackageDirectory / (S.AssetName + TEXT("_Skin"));
    for (const FString& Name : {Out.MapPackageName, Out.SequencePackageName, Out.SkinMaterialPackageName})
    {
        if (!Name.StartsWith(TEXT("/Game/CardistryCapture/Generated/")) || !FPackageName::IsValidLongPackageName(Name))
            return Fail(Error, TEXT("New sequence assets must use a valid /Game/CardistryCapture/Generated/ package path."));
        if (FPackageName::DoesPackageExist(Name) || FindPackage(nullptr, *Name))
            return Fail(Error, TEXT("Refusing to overwrite an existing sequence/map package: ") + Name);
    }

    FAssetCompilingManager::Get().FinishAllCompilation();
    if (S.bPerHandLocalPreview)
    {
        TSet<int32> LeftSections, RightSections;
        if (!ClassifyHandSections(Mesh, S.LeftWristBone, S.RightWristBone, Out.LeftMaterialIds, Out.RightMaterialIds, LeftSections, RightSections, Error) ||
            !AnimatedGeometryBounds(Mesh, Animation, S, Out.LeftLocalBounds, Out.LeftGeometryVerticesPerFrame, Error, &LeftSections) ||
            !AnimatedGeometryBounds(Mesh, Animation, S, Out.RightLocalBounds, Out.RightGeometryVerticesPerFrame, Error, &RightSections))
            return Fail(Error, Error.IsEmpty() ? TEXT("Invalid separate local-hand bounds.") : Error);
        Out.GeometryVerticesPerFrame = Out.LeftGeometryVerticesPerFrame + Out.RightGeometryVerticesPerFrame;
    }
    else if (!AnimatedGeometryBounds(Mesh, Animation, S, Out.AnimatedGeometryBounds, Out.GeometryVerticesPerFrame, Error))
        return Fail(Error, Error.IsEmpty() ? TEXT("Invalid animated geometry bounds.") : Error);
    UPackage* MapPackage = CreatePackage(*Out.MapPackageName);
    const UWorld::InitializationValues IV = UWorld::InitializationValues().AllowAudioPlayback(false)
        .CreatePhysicsScene(false).CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    Out.World = UWorld::CreateWorld(EWorldType::Editor, false, *FPackageName::GetLongPackageAssetName(Out.MapPackageName), MapPackage, true, ERHIFeatureLevel::Num, &IV);
    if (!Out.World) return Fail(Error, TEXT("Could not create isolated research world."));
    Out.World->SetFlags(RF_Public | RF_Standalone);
    GEngine->CreateNewWorldContext(EWorldType::Editor).SetCurrentWorld(Out.World);

    Out.HandsActor = Out.World->SpawnActor<ACardCapResearchHandsActor>();
    Out.HandsActor->SetActorLabel(S.bPerHandLocalPreview
        ? TEXT("Local hand pose viewer (one side at a time; relative hand space unknown)") : TEXT("Captured hands (MANO research asset)"));
    Out.HandsActor->SetActorTransform(S.MeshTransform);
    auto* MeshComponent = CastChecked<UCardCapResearchSkeletalMeshComponent>(Out.HandsActor->GetSkeletalMeshComponent());
    MeshComponent->SetMobility(EComponentMobility::Movable);
    MeshComponent->SetSkeletalMesh(Mesh);
    MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    MeshComponent->VisibilityBasedAnimTickOption = EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
    MeshComponent->bEnableUpdateRateOptimizations = false;
    MeshComponent->bComponentUseFixedSkelBounds = false;
    MeshComponent->SetCastShadow(false);
    if (!MeshComponent->GetRelativeTransform().Equals(FTransform::Identity) || MeshComponent->GetNumMaterials() < 1)
        return Fail(Error, TEXT("Research hand component requires identity transform and actual material slots."));
    UMaterial* SkinMaterial = CreateSkinMaterial(Out.SkinMaterialPackageName, S.bPerHandLocalPreview);
    for (int32 Slot = 0; Slot < MeshComponent->GetNumMaterials(); ++Slot)
        MeshComponent->SetMaterial(Slot, SkinMaterial);
    if (S.bPerHandLocalPreview)
    {
        MeshComponent->bSeparateLocalHands = true;
        MeshComponent->LocalLeftMaterialIds = Out.LeftMaterialIds;
        MeshComponent->LocalRightMaterialIds = Out.RightMaterialIds;
        MeshComponent->SetLocalHandView(false);
    }
    const FBoxSphereBounds BindBounds = Mesh->GetImportedBounds();
    TArray<FTransform> BindJoints;
    const FReferenceSkeleton& Ref = Mesh->GetRefSkeleton();
    for (int32 Index = 0; Index < Ref.GetNum(); ++Index)
    {
        const int32 Parent = Ref.GetParentIndex(Index);
        BindJoints.Add(Parent < 0 ? Ref.GetRefBonePose()[Index] : Ref.GetRefBonePose()[Index] * BindJoints[Parent]);
        MeshComponent->BindVertexToBoneRadiusCm = FMath::Max(MeshComponent->BindVertexToBoneRadiusCm,
            (BindJoints[Index].GetTranslation() - BindBounds.Origin).Length() + BindBounds.SphereRadius);
    }

    Out.CameraActor = Out.World->SpawnActor<ACardCapResearchCameraActor>();
    const FString SourceLabel = S.CameraSourceLabel.IsEmpty()
        ? (S.bCalibrated ? TEXT("calibrated") : TEXT("prior-based, uncalibrated")) : S.CameraSourceLabel;
    const FString CameraLabel = S.bPerHandLocalPreview ? TEXT("Left local viewer (display_only; no source camera)")
        : TEXT("Capture camera (") + SourceLabel + TEXT(")");
    Out.CameraActor->SetActorLabel(CameraLabel);
    Out.CameraActor->SetActorTransform(S.CameraTransform);
    auto* Camera = CastChecked<UCardCapResearchCameraComponent>(Out.CameraActor->GetCameraComponent());
    if (S.bPerHandLocalPreview) FrameLocalHand(Out.CameraActor, Out.LeftLocalBounds, S.Resolution);
    else
    {
        Camera->SetFieldOfView(FMath::RadiansToDegrees(2.0 * FMath::Atan(S.Resolution.X / (2.0 * S.Fx))));
        Camera->SetAspectRatio(double(S.Resolution.X) / S.Resolution.Y);
        Camera->bConstrainAspectRatio = true;
        Camera->PrincipalPointOffset = FVector2D(1.0 - 2.0 * S.Cx / S.Resolution.X, 2.0 * S.Cy / S.Resolution.Y - 1.0);
        SetFixedExposure(Camera->PostProcessSettings);
    }
    Out.OverviewCameraActor = Out.World->SpawnActor<ACameraActor>();
    Out.OverviewCameraActor->SetActorLabel(S.bPerHandLocalPreview
        ? TEXT("Right local viewer (display_only; no inter-hand spatial relation)") : TEXT("Overview camera (independent three-quarter perspective, geometry framing)"));
    UCameraComponent* Overview = Out.OverviewCameraActor->GetCameraComponent();
    Overview->SetFieldOfView(60.0f);
    Overview->SetAspectRatio(double(S.Resolution.X) / S.Resolution.Y);
    Overview->bConstrainAspectRatio = true;
    SetFixedExposure(Overview->PostProcessSettings);
    const FBox& ViewBounds = S.bPerHandLocalPreview ? Out.RightLocalBounds : Out.AnimatedGeometryBounds;
    const FVector Center = ViewBounds.GetCenter();
    const double Radius = ViewBounds.GetExtent().Size();
    const double HalfHorizontal = FMath::DegreesToRadians(30.0);
    const double HalfVertical = FMath::Atan(FMath::Tan(HalfHorizontal) / Overview->AspectRatio);
    const double Distance = Radius / FMath::Sin(FMath::Min(HalfHorizontal, HalfVertical)) * 1.15;
    const FVector OverviewLocation = Center + FVector(-1.0, -1.0, 0.65).GetSafeNormal() * Distance;
    Out.OverviewCameraActor->SetActorLocation(OverviewLocation);
    Out.OverviewCameraActor->SetActorRotation(FRotationMatrix::MakeFromX(Center - OverviewLocation).Rotator());
    auto* Light = Out.World->SpawnActor<ADirectionalLight>();
    Light->SetActorLabel(TEXT("Research key light (single directional)"));
    Light->SetActorRotation(FRotator(-35, -25, 0));
    Light->GetLightComponent()->SetMobility(EComponentMobility::Movable);
    Light->GetLightComponent()->SetIntensity(4.0f);

    UPackage* SequencePackage = CreatePackage(*Out.SequencePackageName);
    Out.Sequence = NewObject<ULevelSequence>(SequencePackage, *FPackageName::GetLongPackageAssetName(Out.SequencePackageName), RF_Public | RF_Standalone | RF_Transactional);
    Out.Sequence->Initialize();
    UMovieScene* Scene = Out.Sequence->GetMovieScene();
    // Integer display frames map exactly to 1000 ticks, including fractional fps.
    const FFrameRate TickRate(S.DisplayRate.Numerator * 1000, S.DisplayRate.Denominator);
    Scene->SetTickResolutionDirectly(TickRate);
    Scene->SetDisplayRate(S.DisplayRate);
    Scene->SetPlaybackRange(0, S.FrameCount * 1000);
    Out.HandsBinding = Scene->AddPossessable(S.bPerHandLocalPreview
        ? TEXT("Separate local hand poses (display_only; spatial relation unknown)") : TEXT("Captured dual hands"), Out.HandsActor->GetClass());
    Out.Sequence->BindPossessableObject(Out.HandsBinding, *Out.HandsActor, Out.World);
    auto* AnimTrack = Scene->AddTrack<UMovieSceneSkeletalAnimationTrack>(Out.HandsBinding);
    AnimTrack->bBlendFirstChildOfRoot = false;
    auto* Section = CastChecked<UMovieSceneSkeletalAnimationSection>(AnimTrack->AddNewAnimation(0, Animation));
    Section->Params.bForceCustomMode = true;
    Section->Params.bSkipAnimNotifiers = true;
    Section->SetRange(TRange<FFrameNumber>(0, S.FrameCount * 1000));
    Out.CameraBinding = Scene->AddPossessable(CameraLabel, Out.CameraActor->GetClass());
    Out.Sequence->BindPossessableObject(Out.CameraBinding, *Out.CameraActor, Out.World);
    auto* CutTrack = CastChecked<UMovieSceneCameraCutTrack>(Scene->AddCameraCutTrack(UMovieSceneCameraCutTrack::StaticClass()));
    auto* Cut = CutTrack->AddNewCameraCut(FMovieSceneObjectBindingID(UE::MovieScene::FRelativeObjectBindingID(Out.CameraBinding)), 0);
    Cut->SetRange(TRange<FFrameNumber>(0, S.FrameCount * 1000));
    Out.SequenceActor = Out.World->SpawnActor<ALevelSequenceActor>();
    Out.SequenceActor->SetActorLabel(S.bPerHandLocalPreview ? TEXT("Local pose sequence; no reconstructed two-hand space") : TEXT("Cardistry captured hand sequence"));
    Out.SequenceActor->SetSequence(Out.Sequence);
    Out.World->UpdateWorldComponents(true, false);
    FAssetRegistryModule::AssetCreated(Out.Sequence);
    FAssetRegistryModule::AssetCreated(Out.World);
    SequencePackage->MarkPackageDirty();
    MapPackage->MarkPackageDirty();
    const FString SequenceFilename = FPackageName::LongPackageNameToFilename(Out.SequencePackageName, FPackageName::GetAssetPackageExtension());
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(SequenceFilename), true);
    FSavePackageArgs SaveArgs;
    SaveArgs.TopLevelFlags = RF_Public | RF_Standalone;
    SaveArgs.SaveFlags = SAVE_NoError;
    FAssetCompilingManager::Get().FinishAllCompilation();
    // Shader compilation is not registered with FAssetCompilingManager in UE
    // 5.4. Wait for the new skin's shader map before saving/rendering, otherwise
    // SceneCapture can show the grey default material during async compilation.
    if (GShaderCompilingManager) GShaderCompilingManager->FinishAllCompilation();
    FMaterialResource* SkinResource = SkinMaterial->GetMaterialResource(Out.World->GetFeatureLevel());
    if (!SkinResource || !SkinResource->IsCompilationFinished() || !SkinResource->GetGameThreadShaderMap()
        || !SkinResource->IsGameThreadShaderMapComplete())
        return Fail(Error, TEXT("Shared skin material has no completed shader map; refusing default-material preview."));
    // The initial proxy may already have cached a fallback while the material
    // was compiling. Recreate it after shader completion, not just pose data.
    MeshComponent->MarkRenderStateDirty();
    Out.World->SendAllEndOfFrameUpdates();
    FlushRenderingCommands();
    const FString SkinFilename = FPackageName::LongPackageNameToFilename(Out.SkinMaterialPackageName, FPackageName::GetAssetPackageExtension());
    if (!UPackage::SavePackage(SkinMaterial->GetOutermost(), SkinMaterial, *SkinFilename, SaveArgs))
        return Fail(Error, TEXT("Could not save the shared skin material."));
    if (!UPackage::SavePackage(SequencePackage, Out.Sequence, *SequenceFilename, SaveArgs))
        return Fail(Error, TEXT("Could not save the Level Sequence asset."));
    const FString MapFilename = FPackageName::LongPackageNameToFilename(Out.MapPackageName, FPackageName::GetMapPackageExtension());
    if (!FEditorFileUtils::SaveMap(Out.World, MapFilename))
        return Fail(Error, TEXT("Could not save the research map asset."));
    return true;
}

bool FCardCapSequenceBuilder::CaptureFrames(const FCardCapSequenceBuildResult& Built,
    const FCardCapSequenceBuildSettings& S, const FString& OutputDirectory,
    TSharedRef<FJsonObject> Report, FString& Error)
{
    Error.Reset();
    if (!IsRunningCommandlet() || !FApp::CanEverRender() || GUsingNullRHI)
        return Fail(Error, TEXT("GPU capture requires a separate commandlet with -AllowCommandletRendering -RenderOffscreen and no -NullRHI."));
    if (!Built.World || !Built.Sequence || !Built.HandsActor || !Built.CameraActor || !Built.OverviewCameraActor)
        return Fail(Error, TEXT("Build must succeed before capture."));
    if (!GIsClient || !Built.World->Scene || !Built.World->Scene->GetRenderScene())
        return Fail(Error, TEXT("No real render scene: commandlet IsClient=true must be set before world creation (an RHI alone is insufficient)."));
    if (!IFileManager::Get().MakeDirectory(*OutputDirectory, true)) return Fail(Error, TEXT("Could not create frame output directory."));
    TArray<FString> FrameDirectories{FString(), TEXT("overview"), TEXT("camera_comparison")};
    if (S.bPerHandLocalPreview)
    {
        FrameDirectories.Add(TEXT("material_support/left"));
        FrameDirectories.Add(TEXT("material_support/right"));
    }
    for (const FString& Subdirectory : FrameDirectories)
    {
        if (!IFileManager::Get().MakeDirectory(*(OutputDirectory / Subdirectory), true))
            return Fail(Error, TEXT("Could not create view output directory."));
        for (int32 Frame = 0; Frame < S.FrameCount; ++Frame)
            if (FPaths::FileExists(OutputDirectory / Subdirectory / FString::Printf(TEXT("frame_%06d.png"), Frame)))
                return Fail(Error, TEXT("Refusing to overwrite rendered frames; use a fresh output directory."));
    }

    FMovieSceneSequencePlaybackSettings Playback;
    Playback.FinishCompletionStateOverride = EMovieSceneCompletionModeOverride::ForceKeepState;
    Playback.bDisableCameraCuts = false;
    ALevelSequenceActor* EvaluationActor = nullptr;
    ULevelSequencePlayer* Player = ULevelSequencePlayer::CreateLevelSequencePlayer(Built.World, Built.Sequence, Playback, EvaluationActor);
    if (!Player) return Fail(Error, TEXT("Could not initialize actual LevelSequencePlayer."));
    auto* CaptureActor = Built.World->SpawnActor<ASceneCapture2D>();
    auto* Capture = CaptureActor->GetCaptureComponent2D();
    Capture->bCaptureEveryFrame = false;
    Capture->bCaptureOnMovement = false;
    Capture->CaptureSource = SCS_FinalColorLDR;
    Capture->ShowFlags.SetMotionBlur(false);
    Capture->ShowFlags.SetTemporalAA(false);
    Capture->bUseCustomProjectionMatrix = !S.bPerHandLocalPreview;
    Capture->ProjectionType = ECameraProjectionMode::Perspective;
    Capture->bOverride_CustomNearClippingPlane = true;
    Capture->CustomNearClippingPlane = S.NearClipCm;
    Capture->CustomProjectionMatrix = S.bPerHandLocalPreview ? FMatrix::Identity : IntrinsicsProjection(S);
    Capture->FOVAngle = Built.CameraActor->GetCameraComponent()->FieldOfView;
    Capture->PostProcessBlendWeight = 1.0f;
    SetFixedExposure(Capture->PostProcessSettings);
    TStrongObjectPtr<UTextureRenderTarget2D> Target(NewObject<UTextureRenderTarget2D>(GetTransientPackage()));
    Target->ClearColor = FLinearColor::Black;
    Target->InitCustomFormat(S.Resolution.X, S.Resolution.Y, PF_B8G8R8A8, false);
    Target->UpdateResourceImmediate(true);
    Capture->TextureTarget = Target.Get();
    FAssetCompilingManager::Get().FinishAllCompilation();
    Built.World->UpdateWorldComponents(true, false);
    FlushRenderingCommands();

    TArray<TSharedPtr<FJsonValue>> Frames;
    auto* Mesh = CastChecked<UCardCapResearchSkeletalMeshComponent>(Built.HandsActor->GetSkeletalMeshComponent());
    bool bSuccess = true;
    auto SelectLocalSide = [&](bool bRight)
    {
        if (!S.bPerHandLocalPreview) return true;
        Mesh->SetLocalHandView(bRight);
        const TArray<int32>& Visible = bRight ? Built.RightMaterialIds : Built.LeftMaterialIds;
        for (int32 LOD = 0; LOD < Mesh->GetNumLODs(); ++LOD)
            for (int32 Material = 0; Material < Mesh->GetNumMaterials(); ++Material)
                if (Mesh->IsMaterialSectionShown(Material, LOD) != Visible.Contains(Material))
                    return Fail(Error, TEXT("Local section visibility did not isolate the requested hand; refusing combined local-origin rendering."));
        return true;
    };
    auto SavePNG = [&](const TArray<FColor>& Pixels, int32 Width, const FString& Filename)
    {
        TArray64<uint8> PNG;
        FImageUtils::PNGCompressImageArray(Width, S.Resolution.Y, TArrayView64<const FColor>(Pixels.GetData(), Pixels.Num()), PNG);
        return FFileHelper::SaveArrayToFile(PNG, *(OutputDirectory / Filename));
    };
    auto RenderView = [&](UCameraComponent* ViewCamera, bool bCaptureIntrinsics, int32 Frame, TArray<FColor>& Pixels, int32& NonBlack)
    {
        CaptureActor->SetActorTransform(ViewCamera->GetComponentTransform());
        Capture->bUseCustomProjectionMatrix = bCaptureIntrinsics;
        Capture->FOVAngle = ViewCamera->FieldOfView;
        Built.World->SendAllEndOfFrameUpdates();
        // Warm both actual viewpoints on the first frame after compilation.
        for (int32 Pass = 0; Pass < (Frame == 0 ? 3 : 1); ++Pass)
        {
            Capture->CaptureScene();
            FlushRenderingCommands();
        }
        FReadSurfaceDataFlags Flags(RCM_UNorm);
        Flags.SetLinearToGamma(false);
        if (!Target->GameThread_GetRenderTargetResource()->ReadPixels(Pixels, Flags)
            || Pixels.Num() != S.Resolution.X * S.Resolution.Y)
            return Fail(Error, TEXT("GPU render target pixel read failed."));
        NonBlack = 0;
        for (FColor& Pixel : Pixels)
        {
            if (Pixel.R > 8 || Pixel.G > 8 || Pixel.B > 8) ++NonBlack;
            Pixel.A = 255;
        }
        return true;
    };
    struct FLocalSupportEvidence
    {
        int32 Pixels = 0, InteriorPixels = 0, DarkInteriorPixels = 0;
        FIntRect Bounds;
        double LitFraction = 0.0;
        bool bPassed = false;
    };
    auto AboveSupportThreshold = [](const FColor& Color)
    {
        return Color.R > LocalSupportThreshold || Color.G > LocalSupportThreshold || Color.B > LocalSupportThreshold;
    };
    auto MeasureLocalSupport = [&](const TArray<FColor>& BaseColor, const TArray<FColor>& Lit)
    {
        FLocalSupportEvidence Evidence;
        const int32 Width = S.Resolution.X, Height = S.Resolution.Y;
        Evidence.Bounds = FIntRect(Width, Height, 0, 0);
        for (int32 Y = 0; Y < Height; ++Y)
            for (int32 X = 0; X < Width; ++X)
            {
                const int32 Pixel = Y * Width + X;
                if (!AboveSupportThreshold(BaseColor[Pixel])) continue;
                ++Evidence.Pixels;
                Evidence.Bounds.Min.X = FMath::Min(Evidence.Bounds.Min.X, X);
                Evidence.Bounds.Min.Y = FMath::Min(Evidence.Bounds.Min.Y, Y);
                Evidence.Bounds.Max.X = FMath::Max(Evidence.Bounds.Max.X, X + 1);
                Evidence.Bounds.Max.Y = FMath::Max(Evidence.Bounds.Max.Y, Y + 1);
                // Remove one 4-neighbour boundary pixel so edge antialiasing
                // cannot masquerade as a dark interior surface.
                if (X == 0 || Y == 0 || X + 1 == Width || Y + 1 == Height
                    || !AboveSupportThreshold(BaseColor[Pixel - 1]) || !AboveSupportThreshold(BaseColor[Pixel + 1])
                    || !AboveSupportThreshold(BaseColor[Pixel - Width]) || !AboveSupportThreshold(BaseColor[Pixel + Width])) continue;
                ++Evidence.InteriorPixels;
                if (!AboveSupportThreshold(Lit[Pixel])) ++Evidence.DarkInteriorPixels;
            }
        Evidence.LitFraction = Evidence.InteriorPixels > 0
            ? 1.0 - double(Evidence.DarkInteriorPixels) / Evidence.InteriorPixels : 0.0;
        Evidence.bPassed = Evidence.InteriorPixels >= 64 && Evidence.LitFraction >= MinimumLocalLitSupportFraction;
        return Evidence;
    };
    for (int32 Frame = 0; Frame < S.FrameCount; ++Frame)
    {
        // FrameTime uses the sequence display rate. Jump prevents accumulating
        // time, playing notifies or inventing intermediate motion.
        Player->SetPlaybackPosition(FMovieSceneSequencePlaybackParams(FFrameTime(Frame), EUpdatePositionMethod::Jump));
        const TArray<UObject*> BoundHands = Player->GetBoundObjects(
            FMovieSceneObjectBindingID(UE::MovieScene::FRelativeObjectBindingID(Built.HandsBinding)));
        if (!BoundHands.Contains(Built.HandsActor))
        {
            bSuccess = Fail(Error, TEXT("Sequencer did not resolve its animated hand actor binding."));
            break;
        }
        Mesh->TickAnimation(0.0f, false);
        Mesh->RefreshBoneTransforms();
        Mesh->UpdateComponentToWorld();
        Mesh->UpdateBounds();
        Mesh->MarkRenderTransformDirty();
        Mesh->MarkRenderDynamicDataDirty();
        if (!Built.HandsActor->GetActorTransform().Equals(FTransform::Identity)
            || !Mesh->GetRelativeTransform().Equals(FTransform::Identity))
        {
            bSuccess = Fail(Error, TEXT("Sequencer changed the identity hand actor/component transform."));
            break;
        }
        UCameraComponent* EvaluatedCamera = Player->GetActiveCameraComponent();
        if (!EvaluatedCamera || EvaluatedCamera->GetOwner() != Built.CameraActor)
        {
            bSuccess = Fail(Error, TEXT("Sequencer camera cut did not resolve to the saved capture camera."));
            break;
        }
        TArray<FColor> Pixels, OverviewPixels, LeftBaseColor, RightBaseColor;
        int32 NonBlack = 0, OverviewNonBlack = 0;
        bool bViewsRendered = SelectLocalSide(false) && RenderView(EvaluatedCamera, !S.bPerHandLocalPreview, Frame, Pixels, NonBlack);
        auto RenderMaterialSupport = [&](UCameraComponent* ViewCamera, TArray<FColor>& BaseColor)
        {
            int32 IgnoredNonBlack = 0;
            Capture->CaptureSource = SCS_BaseColor;
            const bool bRendered = RenderView(ViewCamera, false, Frame, BaseColor, IgnoredNonBlack);
            Capture->CaptureSource = SCS_FinalColorLDR;
            return bRendered;
        };
        if (S.bPerHandLocalPreview) bViewsRendered = bViewsRendered && RenderMaterialSupport(EvaluatedCamera, LeftBaseColor);
        bViewsRendered = bViewsRendered && SelectLocalSide(true) && RenderView(Built.OverviewCameraActor->GetCameraComponent(), false, Frame, OverviewPixels, OverviewNonBlack);
        if (S.bPerHandLocalPreview) bViewsRendered = bViewsRendered && RenderMaterialSupport(Built.OverviewCameraActor->GetCameraComponent(), RightBaseColor);
        bViewsRendered = SelectLocalSide(false) && bViewsRendered;
        if (!bViewsRendered)
        {
            bSuccess = false;
            break;
        }
        const FString Filename = FString::Printf(TEXT("frame_%06d.png"), Frame);
        TArray<FColor> Comparison;
        Comparison.SetNumUninitialized(Pixels.Num() * 2);
        for (int32 Row = 0; Row < S.Resolution.Y; ++Row)
        {
            FMemory::Memcpy(Comparison.GetData() + Row * S.Resolution.X * 2, Pixels.GetData() + Row * S.Resolution.X, S.Resolution.X * sizeof(FColor));
            FMemory::Memcpy(Comparison.GetData() + (Row * 2 + 1) * S.Resolution.X, OverviewPixels.GetData() + Row * S.Resolution.X, S.Resolution.X * sizeof(FColor));
        }
        if (!SavePNG(Pixels, S.Resolution.X, Filename)
            || !SavePNG(OverviewPixels, S.Resolution.X, TEXT("overview") / Filename)
            || !SavePNG(Comparison, S.Resolution.X * 2, TEXT("camera_comparison") / Filename)
            || (S.bPerHandLocalPreview && (!SavePNG(LeftBaseColor, S.Resolution.X, TEXT("material_support/left") / Filename)
                || !SavePNG(RightBaseColor, S.Resolution.X, TEXT("material_support/right") / Filename))))
        {
            bSuccess = Fail(Error, TEXT("Failed to save rendered PNG."));
            break;
        }
        if (Frame == 0 && S.bPerHandLocalPreview)
        {
            if (!SavePNG(LeftBaseColor, S.Resolution.X, TEXT("material_basecolor_frame_000000.png")))
            {
                bSuccess = Fail(Error, TEXT("Failed to save first-frame material base-color diagnostic."));
                break;
            }
        }
        else if (Frame == 0)
        {
            TArray<FColor> BaseColorPixels;
            int32 BaseColorNonBlack = 0;
            Capture->CaptureSource = SCS_BaseColor;
            const bool bBaseColorRendered = RenderView(EvaluatedCamera, !S.bPerHandLocalPreview, Frame, BaseColorPixels, BaseColorNonBlack);
            Capture->CaptureSource = SCS_FinalColorLDR;
            if (!bBaseColorRendered || !SavePNG(BaseColorPixels, S.Resolution.X, TEXT("material_basecolor_frame_000000.png")))
            {
                bSuccess = Fail(Error, TEXT("Failed to save first-frame material base-color diagnostic."));
                break;
            }
        }
        auto Item = MakeShared<FJsonObject>();
        Item->SetNumberField(TEXT("frame"), Frame);
        Item->SetNumberField(TEXT("time_seconds"), Frame / S.DisplayRate.AsDecimal());
        Item->SetStringField(TEXT("file"), Filename);
        Item->SetNumberField(TEXT("nonblack_pixels_above_8"), NonBlack);
        Item->SetStringField(TEXT("overview_file"), TEXT("overview") / Filename);
        Item->SetStringField(TEXT("comparison_file"), TEXT("camera_comparison") / Filename);
        Item->SetNumberField(TEXT("overview_nonblack_pixels_above_8"), OverviewNonBlack);
        bool bLocalReadabilityPassed = true;
        if (S.bPerHandLocalPreview)
        {
            auto PixelBounds = [&](const TArray<FColor>& Values)
            {
                int32 MinX = S.Resolution.X, MinY = S.Resolution.Y, MaxX = -1, MaxY = -1;
                for (int32 Pixel = 0; Pixel < Values.Num(); ++Pixel)
                {
                    const FColor& Color = Values[Pixel];
                    if (Color.R <= 8 && Color.G <= 8 && Color.B <= 8) continue;
                    const int32 X = Pixel % S.Resolution.X, Y = Pixel / S.Resolution.X;
                    MinX = FMath::Min(MinX, X); MinY = FMath::Min(MinY, Y);
                    MaxX = FMath::Max(MaxX, X); MaxY = FMath::Max(MaxY, Y);
                }
                return TArray<TSharedPtr<FJsonValue>>{MakeShared<FJsonValueNumber>(MinX), MakeShared<FJsonValueNumber>(MinY),
                    MakeShared<FJsonValueNumber>(MaxX + 1), MakeShared<FJsonValueNumber>(MaxY + 1)};
            };
            Item->SetArrayField(TEXT("left_hand_pixel_bounds_xyxy"), PixelBounds(Pixels));
            Item->SetArrayField(TEXT("right_hand_pixel_bounds_xyxy"), PixelBounds(OverviewPixels));
            Item->SetNumberField(TEXT("left_hand_nonblack_pixel_fraction"), double(NonBlack) / Pixels.Num());
            Item->SetNumberField(TEXT("right_hand_nonblack_pixel_fraction"), double(OverviewNonBlack) / OverviewPixels.Num());
            Item->SetStringField(TEXT("pixel_bounds_semantics"), TEXT("Rendered local hand support, background threshold >8 per color channel; presentation visibility only, not source observation coverage"));
            Item->SetBoolField(TEXT("separate_material_visibility_verified"), true);
            auto SupportObject = [&](const FLocalSupportEvidence& Evidence, const TCHAR* Side)
            {
                auto Object = MakeShared<FJsonObject>();
                Object->SetStringField(TEXT("basecolor_file"), FString(TEXT("material_support")) / Side / Filename);
                Object->SetNumberField(TEXT("basecolor_support_pixels"), Evidence.Pixels);
                Object->SetNumberField(TEXT("interior_support_pixels"), Evidence.InteriorPixels);
                Object->SetNumberField(TEXT("dark_interior_pixels"), Evidence.DarkInteriorPixels);
                Object->SetNumberField(TEXT("lit_interior_fraction"), Evidence.LitFraction);
                Object->SetBoolField(TEXT("passed"), Evidence.bPassed);
                Object->SetArrayField(TEXT("basecolor_bounds_xyxy"), TArray<TSharedPtr<FJsonValue>>{
                    MakeShared<FJsonValueNumber>(Evidence.Bounds.Min.X), MakeShared<FJsonValueNumber>(Evidence.Bounds.Min.Y),
                    MakeShared<FJsonValueNumber>(Evidence.Bounds.Max.X), MakeShared<FJsonValueNumber>(Evidence.Bounds.Max.Y)});
                return Object;
            };
            const FLocalSupportEvidence LeftSupport = MeasureLocalSupport(LeftBaseColor, Pixels);
            const FLocalSupportEvidence RightSupport = MeasureLocalSupport(RightBaseColor, OverviewPixels);
            Item->SetObjectField(TEXT("left_material_readability"), SupportObject(LeftSupport, TEXT("left")));
            Item->SetObjectField(TEXT("right_material_readability"), SupportObject(RightSupport, TEXT("right")));
            bLocalReadabilityPassed = LeftSupport.bPassed && RightSupport.bPassed;
            Item->SetBoolField(TEXT("local_material_readability_verified"), bLocalReadabilityPassed);
        }
        TArray<TSharedPtr<FJsonValue>> BonePositions;
        for (const FTransform& Bone : Mesh->GetComponentSpaceTransforms())
            BonePositions.Add(MakeShared<FJsonValueArray>(VectorValues(Bone.GetTranslation())));
        Item->SetArrayField(TEXT("evaluated_component_bone_positions_cm"), BonePositions);
        Frames.Add(MakeShared<FJsonValueObject>(Item));
        if (!bLocalReadabilityPassed)
        {
            bSuccess = Fail(Error, FString::Printf(TEXT("Frame %d has dark interior material support that looks like missing skin; lit and BaseColor PNGs retained."), Frame));
            break;
        }
        if (NonBlack == 0 || OverviewNonBlack == 0)
        {
            bSuccess = Fail(Error, FString::Printf(TEXT("Frame %d rendered black; PNG retained as failure evidence."), Frame));
            break;
        }
    }
    Report->SetStringField(TEXT("evaluation"), TEXT("ULevelSequencePlayer.SetPlaybackPosition(display frame), skeletal pose refresh, SceneCapture2D GPU readback"));
    Report->SetBoolField(TEXT("real_render_scene"), Built.World->Scene->GetRenderScene() != nullptr);
    Report->SetBoolField(TEXT("commandlet_client"), GIsClient);
    Report->SetStringField(TEXT("sequence"), Built.Sequence->GetPathName());
    Report->SetStringField(TEXT("map"), Built.World->GetPathName());
    Report->SetBoolField(TEXT("calibrated"), S.bCalibrated);
    Report->SetStringField(TEXT("view_mode"), S.bPerHandLocalPreview ? TEXT("per_hand_local") : TEXT("shared"));
    Report->SetBoolField(TEXT("display_only"), S.bPerHandLocalPreview);
    Report->SetStringField(TEXT("coordinate_frame"), S.bPerHandLocalPreview ? TEXT("per_hand_wrist_local") : TEXT("shared_camera"));
    Report->SetBoolField(TEXT("inter_hand_transform_known"), !S.bPerHandLocalPreview);
    Report->SetStringField(TEXT("capture_camera_source_label"), S.CameraSourceLabel.IsEmpty()
        ? (S.bCalibrated ? TEXT("calibrated") : TEXT("prior-based, uncalibrated")) : S.CameraSourceLabel);
    Report->SetStringField(TEXT("coordinate_units"), S.CoordinateUnits);
    if (!S.DistortionPolicy.IsEmpty()) Report->SetStringField(TEXT("distortion_policy"), S.DistortionPolicy);
    Report->SetStringField(TEXT("capture_intrinsics_policy"), S.bPerHandLocalPreview
        ? TEXT("Source camera intrinsics are null. Separate display_only local viewers are framed independently; no full-image depth or inter-hand spatial estimate is constructed.")
        : TEXT("fx/fy/cx/cy unchanged from capture; no camera substitution or hand actor alignment"));
    Report->SetStringField(TEXT("comparison_layout"), S.bPerHandLocalPreview
        ? TEXT("left: left hand wrist-local pose; right: right hand wrist-local pose. Independent display_only images, not one shared scene; relative hand position and depth unknown.")
        : TEXT("left: capture K; right: independent three-quarter perspective, 60 degree horizontal FOV"));
    if (S.bPerHandLocalPreview)
    {
        auto LocalView = [&](ACameraActor* Actor, const FBox& Bounds, int32 VertexCount, const TArray<int32>& Materials, const TCHAR* Side)
        {
            auto View = MakeShared<FJsonObject>();
            View->SetStringField(TEXT("role"), TEXT("display_only")); View->SetStringField(TEXT("side"), Side);
            View->SetStringField(TEXT("coordinate_frame"), TEXT("per_hand_wrist_local"));
            View->SetStringField(TEXT("projection"), TEXT("Ordinary perspective chosen solely for local pose inspection; never copied into source camera.intrinsics"));
            View->SetNumberField(TEXT("horizontal_fov_degrees"), Actor->GetCameraComponent()->FieldOfView);
            View->SetArrayField(TEXT("location_display_units"), VectorValues(Actor->GetActorLocation()));
            View->SetArrayField(TEXT("forward_direction"), VectorValues(Actor->GetActorForwardVector()));
            View->SetArrayField(TEXT("local_geometry_min_display_units"), VectorValues(Bounds.Min));
            View->SetArrayField(TEXT("local_geometry_max_display_units"), VectorValues(Bounds.Max));
            View->SetNumberField(TEXT("vertices_per_frame"), VertexCount);
            View->SetStringField(TEXT("framing"), TEXT("This hand only: actual animated imported LOD0 skin vertices across all source frames, bounding sphere plus 15 percent presentation margin"));
            TArray<TSharedPtr<FJsonValue>> MaterialValues;
            for (int32 Material : Materials) MaterialValues.Add(MakeShared<FJsonValueNumber>(Material));
            View->SetArrayField(TEXT("visible_material_ids_from_skin_weights"), MaterialValues);
            return View;
        };
        Report->SetObjectField(TEXT("left_local_view"), LocalView(Built.CameraActor, Built.LeftLocalBounds, Built.LeftGeometryVerticesPerFrame, Built.LeftMaterialIds, TEXT("left")));
        Report->SetObjectField(TEXT("right_local_view"), LocalView(Built.OverviewCameraActor, Built.RightLocalBounds, Built.RightGeometryVerticesPerFrame, Built.RightMaterialIds, TEXT("right")));
        Report->SetField(TEXT("overview_camera"), MakeShared<FJsonValueNull>());
        Report->SetField(TEXT("capture_camera_location_cm"), MakeShared<FJsonValueNull>());
        Report->SetField(TEXT("capture_camera_forward_direction"), MakeShared<FJsonValueNull>());
        Report->SetStringField(TEXT("visibility_policy"), TEXT("One combined asset, only one hand's independently classified material sections visible per image; no hidden-bone scaling, no actor or component alignment. Saved map defaults to left, with a local-side Details toggle."));
        auto Readability = MakeShared<FJsonObject>();
        Readability->SetStringField(TEXT("scope"), TEXT("display_only lighting readability, not geometry completeness, physical skin appearance or pose accuracy"));
        Readability->SetStringField(TEXT("method"), TEXT("Every side/frame: actual GPU BaseColor support versus final-lit pixels. RGB max >32, one 4-neighbour pixel erosion for interior, minimum 64 interior pixels and 99 percent visible interior support."));
        Readability->SetStringField(TEXT("legacy_pixel_bounds_warning"), TEXT("The legacy >8 nonblack bounding boxes include background noise; use per-side material-readability BaseColor bounds for rendered mesh support."));
        Readability->SetNumberField(TEXT("emissive_basecolor_fraction"), LocalInspectionEmissiveFraction);
        Readability->SetNumberField(TEXT("support_rgb_threshold"), LocalSupportThreshold);
        Readability->SetNumberField(TEXT("minimum_lit_interior_fraction"), MinimumLocalLitSupportFraction);
        Report->SetObjectField(TEXT("local_material_readability_policy"), Readability);
        Report->SetNumberField(TEXT("display_near_clip_units"), S.NearClipCm);
    }
    else
    {
        auto OverviewReport = MakeShared<FJsonObject>();
        OverviewReport->SetStringField(TEXT("projection"), TEXT("ordinary perspective; independent camera transform; no custom projection matrix"));
        OverviewReport->SetArrayField(TEXT("location_cm"), VectorValues(Built.OverviewCameraActor->GetActorLocation()));
        OverviewReport->SetArrayField(TEXT("forward_direction"), VectorValues(Built.OverviewCameraActor->GetActorForwardVector()));
        OverviewReport->SetNumberField(TEXT("horizontal_fov_degrees"), Built.OverviewCameraActor->GetCameraComponent()->FieldOfView);
        OverviewReport->SetStringField(TEXT("framing"), TEXT("all source frames, actual imported LOD0 linear blend skin vertices; enclosing sphere with 15 percent camera-distance margin; no outlier removal"));
        OverviewReport->SetNumberField(TEXT("vertices_per_frame"), Built.GeometryVerticesPerFrame);
        OverviewReport->SetArrayField(TEXT("animated_geometry_min_cm"), VectorValues(Built.AnimatedGeometryBounds.Min));
        OverviewReport->SetArrayField(TEXT("animated_geometry_max_cm"), VectorValues(Built.AnimatedGeometryBounds.Max));
        Report->SetObjectField(TEXT("overview_camera"), OverviewReport);
        Report->SetArrayField(TEXT("capture_camera_location_cm"), VectorValues(Built.CameraActor->GetActorLocation()));
        Report->SetArrayField(TEXT("capture_camera_forward_direction"), VectorValues(Built.CameraActor->GetActorForwardVector()));
    }
    Report->SetBoolField(TEXT("hand_actor_identity"), Built.HandsActor->GetActorTransform().Equals(FTransform::Identity));
    Report->SetBoolField(TEXT("hand_component_identity"), Mesh->GetRelativeTransform().Equals(FTransform::Identity));
    Report->SetBoolField(TEXT("mesh_proxy_fallback_to_default_material"), Mesh->ShouldRenderProxyFallbackToDefaultMaterial());
    Report->SetBoolField(TEXT("capture_show_materials"), Capture->ShowFlags.Materials);
    Report->SetBoolField(TEXT("capture_lighting_only_override"), Capture->ShowFlags.LightingOnlyOverride);
    Report->SetBoolField(TEXT("capture_diffuse_specular_override"), Capture->ShowFlags.OverrideDiffuseAndSpecular);
    TArray<TSharedPtr<FJsonValue>> MaterialSlots;
    for (int32 Slot = 0; Slot < Mesh->GetNumMaterials(); ++Slot)
    {
        auto Material = MakeShared<FJsonObject>();
        Material->SetNumberField(TEXT("slot"), Slot);
        Material->SetStringField(TEXT("material"), Mesh->GetMaterial(Slot) ? Mesh->GetMaterial(Slot)->GetPathName() : TEXT("null"));
        MaterialSlots.Add(MakeShared<FJsonValueObject>(Material));
    }
    Report->SetArrayField(TEXT("material_slots"), MaterialSlots);
    int32 DirectionalLightCount = 0;
    for (TActorIterator<ADirectionalLight> It(Built.World); It; ++It) ++DirectionalLightCount;
    Report->SetNumberField(TEXT("directional_light_count"), DirectionalLightCount);
    if (S.bPerHandLocalPreview)
        for (const TCHAR* Key : {TEXT("fx"), TEXT("fy"), TEXT("cx"), TEXT("cy")}) Report->SetField(Key, MakeShared<FJsonValueNull>());
    else
    {
        Report->SetNumberField(TEXT("fx"), S.Fx); Report->SetNumberField(TEXT("fy"), S.Fy);
        Report->SetNumberField(TEXT("cx"), S.Cx); Report->SetNumberField(TEXT("cy"), S.Cy);
    }
    Report->SetNumberField(TEXT("width"), S.Resolution.X);
    Report->SetNumberField(TEXT("height"), S.Resolution.Y);
    Report->SetNumberField(TEXT("fps"), S.DisplayRate.AsDecimal());
    Report->SetNumberField(TEXT("requested_frames"), S.FrameCount);
    Report->SetNumberField(TEXT("captured_frames"), Frames.Num());
    Report->SetBoolField(TEXT("success"), bSuccess);
    Report->SetStringField(TEXT("error"), Error);
    Report->SetArrayField(TEXT("frames"), Frames);
    FString JSON;
    FJsonSerializer::Serialize(Report, TJsonWriterFactory<>::Create(&JSON));
    if (!FFileHelper::SaveStringToFile(JSON, *(OutputDirectory / TEXT("sequence_render.json"))))
        bSuccess = Fail(Error, TEXT("Could not save sequence render evidence."));
    Player->Stop();
    Capture->TextureTarget = nullptr;
    Built.World->DestroyActor(CaptureActor);
    Built.World->DestroyActor(EvaluationActor);
    FlushRenderingCommands();
    return bSuccess;
}

void FCardCapSequenceBuilder::ReleaseWorld(FCardCapSequenceBuildResult& Built)
{
    if (Built.World)
    {
        if (GEngine) GEngine->DestroyWorldContext(Built.World);
        Built.World->DestroyWorld(false);
        Built.World->RemoveFromRoot();
    }
    Built = FCardCapSequenceBuildResult();
}
