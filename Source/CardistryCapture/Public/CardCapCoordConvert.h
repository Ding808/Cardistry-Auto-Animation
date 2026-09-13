#pragma once
#include "CoreMinimal.h"

class CARDISTRYCAPTURE_API FCardCapCoordConvert
{
public:
    // Convert the declared Y-up input from meters to centimeters.
    static FVector SpecManoPositionToUE(const FVector& PositionM);
    static FVector UEPositionToSpecMano(const FVector& PositionCm);
    // Actual WiLoR/OpenCV camera: X-right, Y-down, Z-forward, meters -> cm.
    static FVector CameraPositionToUE(const FVector& PositionM);
    static FVector UEPositionToCamera(const FVector& PositionCm);
    // Unit quaternions only. Each is the full basis change C R C^-1,
    // including the determinant correction for the camera reflection.
    static FQuat SpecManoRotationToUE(const FQuat& Rotation);
    static FQuat UERotationToSpecMano(const FQuat& Rotation);
    static FQuat CameraRotationToUE(const FQuat& Rotation);
    static FQuat UERotationToCamera(const FQuat& Rotation);
};
