#include "CardCapDisplaySpace.h"
#include "CardCapSequenceBuilder.h"

#if WITH_DEV_AUTOMATION_TESTS
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "LevelSequence.h"
#include "LevelSequenceActor.h"
#include "LevelSequencePlayer.h"
#include "Misc/AutomationTest.h"
#include "Misc/ScopeExit.h"
#include "MovieScene.h"
#include "Sections/MovieSceneFloatSection.h"
#include "Tracks/MovieSceneFloatTrack.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FCardCapDisplayFocalTest, "CardistryCapture.Editor.DisplaySpaceFocalControls",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FCardCapDisplayFocalTest::RunTest(const FString& Parameters)
{
    const UWorld::InitializationValues Init = UWorld::InitializationValues().AllowAudioPlayback(false)
        .CreatePhysicsScene(false).CreateNavigation(false).CreateAISystem(false).ShouldSimulatePhysics(false);
    UWorld* World = UWorld::CreateWorld(EWorldType::Editor, false, NAME_None, nullptr, true, ERHIFeatureLevel::Num, &Init);
    if (!TestNotNull(TEXT("Isolated display test world"), World)) return false;
    GEngine->CreateNewWorldContext(EWorldType::Editor).SetCurrentWorld(World);
    ON_SCOPE_EXIT { GEngine->DestroyWorldContext(World); World->DestroyWorld(false); World->RemoveFromRoot(); };
    auto* Display = World->SpawnActor<ACardCapDisplaySpaceActor>();
    Display->LeftHand = World->SpawnActor<ACardCapResearchHandsActor>();
    Display->RightHand = World->SpawnActor<ACardCapResearchHandsActor>();
    Display->DisplayCamera = World->SpawnActor<ACardCapResearchCameraActor>();
    Display->OverviewCamera = World->SpawnActor<ACameraActor>();
    Display->LeftBounds = FBox(FVector(-3), FVector(3));
    Display->RightBounds = Display->LeftBounds;
    FCardCapDisplaySpaceData Data;
    Data.AssumedFocalPx = 1234.567891234;
    Data.HandLength = 20;
    Data.Resolution = FIntPoint(720, 1280);
    Data.PrincipalPoint = FVector2D(360, 640);
    Data.LeftConstant = {FVector(0, -4, 2), FVector(0, -3, 2)};
    Data.RightConstant = {FVector(0, 4, 0), FVector(0, 6, 0)};
    Data.LeftPerFocal = {FVector(.03, 0, 0), FVector(.032, 0, 0)};
    Data.RightPerFocal = {FVector(.04, 0, 0), FVector(.041, 0, 0)};
    Display->Configure(Data);
    TestEqual(TEXT("The focal assumption retains double precision"), Display->DisplayAssumedFocalPx, Data.AssumedFocalPx);
    TestTrue(TEXT("Left display placement follows the sidecar equation"), Display->LeftHand->GetActorLocation().Equals(
        Data.LeftConstant[0] + Data.AssumedFocalPx * Data.LeftPerFocal[0], 1.e-9));
    const float FirstFov = Display->DisplayCamera->GetCameraComponent()->FieldOfView;
    const double FirstRatio = Display->WristDistanceOverHandLength;
    Display->DisplayAssumedFocalPx = Data.AssumedFocalPx * 1.25;
    Display->ApplyDisplay();
    TestTrue(TEXT("Editing focal length updates the display camera"), Display->DisplayCamera->GetCameraComponent()->FieldOfView < FirstFov);
    TestTrue(TEXT("Editing focal length updates wrist distance / hand length"), Display->WristDistanceOverHandLength > FirstRatio);
    TestTrue(TEXT("Both hand placements use the changed focal length"), Display->RightHand->GetActorLocation().Equals(
        Data.RightConstant[0] + Display->DisplayAssumedFocalPx * Data.RightPerFocal[0], 1.e-9));

    ULevelSequence* Sequence = NewObject<ULevelSequence>(World);
    Sequence->Initialize();
    UMovieScene* Scene = Sequence->GetMovieScene();
    Scene->SetDisplayRate(FFrameRate(30, 1));
    Scene->SetTickResolutionDirectly(FFrameRate(30000, 1));
    Scene->SetPlaybackRange(0, 2000);
    const FGuid Binding = Scene->AddPossessable(TEXT("Display assumptions"), Display->GetClass());
    Sequence->BindPossessableObject(Binding, *Display, World);
    auto* Track = Scene->AddTrack<UMovieSceneFloatTrack>(Binding);
    Track->SetPropertyNameAndPath(TEXT("DisplayFrame"), TEXT("DisplayFrame"));
    auto* Section = CastChecked<UMovieSceneFloatSection>(Track->CreateNewSection());
    Section->SetRange(TRange<FFrameNumber>(0, 2000));
    Section->GetChannel().AddLinearKey(0, 0.f);
    Section->GetChannel().AddLinearKey(2000, 2.f);
    Track->AddSection(*Section);
    FMovieSceneSequencePlaybackSettings Playback;
    Playback.FinishCompletionStateOverride = EMovieSceneCompletionModeOverride::ForceKeepState;
    ALevelSequenceActor* SequenceActor = nullptr;
    ULevelSequencePlayer* Player = ULevelSequencePlayer::CreateLevelSequencePlayer(World, Sequence, Playback, SequenceActor);
    if (!TestNotNull(TEXT("A real sequence player evaluates the display frame property"), Player)) return false;
    Player->SetPlaybackPosition(FMovieSceneSequencePlaybackParams(FFrameTime(1), EUpdatePositionMethod::Jump));
    TestEqual(TEXT("Sequencer calls the display frame setter"), Display->DisplayFrame, 1.f);
    TestTrue(TEXT("Scrubbing updates the display placement at the selected source frame"), Display->LeftHand->GetActorLocation().Equals(
        Data.LeftConstant[1] + Display->DisplayAssumedFocalPx * Data.LeftPerFocal[1], 1.e-9));
    const FVector CommonPosition = Display->RightHand->GetActorLocation();
    Display->bShowLocalHands = true;
    Display->bShowRightLocalHand = true;
    Display->ApplyDisplay();
    TestTrue(TEXT("Optional local view has no common-space placement"), Display->RightHand->GetActorLocation().IsNearlyZero());
    TestFalse(TEXT("Optional right local view hides the left copy"), Display->LeftHand->GetSkeletalMeshComponent()->IsVisible());
    TestTrue(TEXT("Optional right local view retains the right copy"), Display->RightHand->GetSkeletalMeshComponent()->IsVisible());
    Display->bShowLocalHands = false;
    Display->ApplyDisplay();
    TestTrue(TEXT("Returning to common view restores the same assumption at the same frame"), Display->RightHand->GetActorLocation().Equals(CommonPosition, 1.e-9));
    TestTrue(TEXT("Common view displays both copies"), Display->LeftHand->GetSkeletalMeshComponent()->IsVisible()
        && Display->RightHand->GetSkeletalMeshComponent()->IsVisible());
    Player->Stop();
    return true;
}
#endif
