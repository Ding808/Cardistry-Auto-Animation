#pragma once
#include "CoreMinimal.h"

// 1.0/1.1 retain their historical centimeter contract. 1.2 explicitly declares
// whether the same stored coordinates are metric or conditional display units.
// No coordinate conversion, metric anchoring or pose synthesis during parsing.
enum class ECardCapProvenance : uint8 { Observed, Calibrated, UserMeasured, Inferred, Unobservable };
enum class ECardCapSampleKind : uint8
{
    DetectedModelObservation, TrackedRoiModelObservation, Interpolated, EndpointHold, Missing
};

struct CARDISTRYCAPTURE_API FCardCapProvenance
{
    ECardCapProvenance CameraIntrinsics = ECardCapProvenance::Unobservable;
    ECardCapProvenance CameraDistortion = ECardCapProvenance::Unobservable;
    ECardCapProvenance MetricScale = ECardCapProvenance::Unobservable;
    ECardCapProvenance HandGeometry = ECardCapProvenance::Inferred;
    FString CoordinateUnits;
    FString CoordinateFrame; // 1.3: shared_camera or per_hand_wrist_local.
    TArray<FString> Assumptions;
};
struct CARDISTRYCAPTURE_API FCardCapMeta
{
    FString SourceVideo, ProcessedAt, PipelineVersion;
    double Fps = 0;
    int32 FrameCount = 0;
    FIntPoint Resolution = FIntPoint::ZeroValue;
};

struct CARDISTRYCAPTURE_API FCardCapCamera
{
    double Fx = 0, Fy = 0, Cx = 0, Cy = 0;
    TArray<double> Distortion;
    bool bCalibrated = false;
    // False means the associated storage is not a value. Never substitute zero.
    bool bHasIntrinsics = false, bHasDistortion = false;
};

struct CARDISTRYCAPTURE_API FCardCapScale
{
    double MetersPerUnit = 0, Confidence = 0;
    FString AnchorMethod;
    TArray<int32> AnchorFrames;
    bool bHasMetersPerUnit = false, bHasConfidence = false, bHasAnchorMethod = false;
};

struct CARDISTRYCAPTURE_API FCardCapHandFrame
{
    int32 Frame = 0;
    double Confidence = 0;
    FVector GlobalTransCm = FVector::ZeroVector;
    bool bHasGlobalTranslation = false; // 1.3 null is unknown, not a zero position.
    FQuat GlobalRotQuat = FQuat::Identity;
    TArray<FVector> JointPositionsCm;
    TMap<FName, FQuat> BoneRotations;
    TArray<int32> OccludedJoints;
    ECardCapSampleKind SampleKind = ECardCapSampleKind::DetectedModelObservation;
    bool bPoseValid = true; // 1.2 explicit mask; legacy poses keep legacy semantics.
};

struct CARDISTRYCAPTURE_API FCardCapHand
{
    FString Side;
    TArray<double> ManoShape;
    TArray<FCardCapHandFrame> Frames;
};

enum class ECardCapContactState : uint8 { Gripped, Free, Resting, Sliding, Unknown };

struct CARDISTRYCAPTURE_API FCardCapPacketFrame
{
    int32 Frame = 0;
    double Confidence = 0;
    TOptional<double> ReprojectionErrorPx;
    FVector PositionCm = FVector::ZeroVector;
    FQuat RotationQuat = FQuat::Identity;
    TOptional<FVector> LinearVelocityCmS;
    TOptional<FVector> AngularVelocityRadS;
    ECardCapContactState ContactState = ECardCapContactState::Unknown;
    TOptional<TArray<FName>> ContactBones; // Unset = unassessed; set empty = no listed contacts.
};

struct CARDISTRYCAPTURE_API FCardCapPacketDimensionsCm
{
    double Width = 0, Height = 0;
    bool bHasWidth = false, bHasHeight = false;
    TOptional<double> Thickness; // 1.1+ thickness; 1.2 also allows null width/height.
};

struct CARDISTRYCAPTURE_API FCardCapPacket
{
    FString Id;
    int32 BirthFrame = 0, DeathFrame = 0;
    TOptional<int32> CardCountEstimate;
    TOptional<double> CardCountConfidence; // Set/unset together with the estimate.
    FCardCapPacketDimensionsCm DimensionsCm;
    TArray<FCardCapPacketFrame> Frames;
};

struct CARDISTRYCAPTURE_API FCardCapSplitEvent
{
    int32 Frame = 0;
    FString Source;
    TArray<FString> Results;
};

struct CARDISTRYCAPTURE_API FCardCapMergeEvent
{
    int32 Frame = 0;
    TArray<FString> Sources;
    FString Result;
};

struct CARDISTRYCAPTURE_API FCardCapReleaseEvent
{
    int32 Frame = 0;
    FString Packet;
    FVector ReleaseVelocityCmS = FVector::ZeroVector;
    bool bHasReleaseVelocity = false;
};

struct CARDISTRYCAPTURE_API FCardCapEvents
{
    TArray<FCardCapSplitEvent> Splits;
    TArray<FCardCapMergeEvent> Merges;
    TArray<FCardCapReleaseEvent> Releases;
};

struct CARDISTRYCAPTURE_API FCardCapQuality
{
    double MeanHandConfidence = 0;
    TOptional<double> MeanReprojectionErrorPx; // Required JSON key; null = unmeasured.
    TArray<FIntPoint> LowConfidenceRanges; // Inclusive [first,last].
    TArray<FString> Warnings;
};

struct CARDISTRYCAPTURE_API FCardCapHandBoneMapping
{
    TArray<FName> BoneNames; // Wrist, then 15 MANO joints in configured order.
    FVector RestOffsetCameraM = FVector::ZeroVector;
};

struct CARDISTRYCAPTURE_API FCardCapBoneMapping
{
    FName RootBone;
    TArray<FString> ManoJointOrder;
    TArray<int32> ManoParents, LandmarkIndices;
    TMap<FString, FCardCapHandBoneMapping> Hands;
};

struct CARDISTRYCAPTURE_API FCardCapCapture
{
    FString FormatVersion;
    FCardCapMeta Meta;
    FCardCapCamera Camera;
    FCardCapScale Scale;
    TArray<FCardCapHand> Hands;
    TArray<FCardCapPacket> Packets;
    FCardCapEvents Events;
    FCardCapQuality Quality;
    FCardCapBoneMapping BoneMapping;
    FCardCapProvenance Provenance; // Mandatory for 1.2; not inferred for old files.
    bool bInterHandTransformKnown = true; // Explicit 1.3 mask; legacy semantics retained.
};
