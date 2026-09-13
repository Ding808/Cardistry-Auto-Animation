#pragma once
#include "CardCapData.h"

struct FReferenceSkeleton;

struct CARDISTRYCAPTURE_API FCardCapRetargetFrame
{
    int32 Frame = 0;
    TArray<FTransform> LocalTransforms; // Same complete order as target reference skeleton.
};

struct CARDISTRYCAPTURE_API FCardCapRetargetResult
{
    double Fps = 0;
    int32 FrameCount = 0;
    TArray<FName> BoneNames;
    TArray<FCardCapRetargetFrame> Frames;
};

class CARDISTRYCAPTURE_API FCardCapRetargeter
{
public:
    // Prepared full frames only: missing samples must be resolved and labeled upstream.
    // Unit reference scales are required. Finger lengths and all non-wrist translations
    // remain the reference values. Failure leaves OutResult unchanged.
    static bool BuildLocalPoseFrames(const FCardCapCapture& Capture,
        const FReferenceSkeleton& Target, FCardCapRetargetResult& OutResult, FString& OutError);
};
