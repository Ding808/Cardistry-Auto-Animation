#include "CardCapCoordConvert.h"

FVector FCardCapCoordConvert::SpecManoPositionToUE(const FVector& P)
{
    return FVector(P.Z, P.X, P.Y) * 100.0;
}

FVector FCardCapCoordConvert::UEPositionToSpecMano(const FVector& P)
{
    return FVector(P.Y, P.Z, P.X) / 100.0;
}

FVector FCardCapCoordConvert::CameraPositionToUE(const FVector& P)
{
    return FVector(P.Z, P.X, -P.Y) * 100.0;
}

FVector FCardCapCoordConvert::UEPositionToCamera(const FVector& P)
{
    return FVector(P.Y, -P.Z, P.X) / 100.0;
}

// For any orthogonal C, a quaternion's vector part transforms as det(C) C v,
// while w stays unchanged. This is C R(q) C^-1, also when det(C) = -1.
FQuat FCardCapCoordConvert::SpecManoRotationToUE(const FQuat& Q)
{
    return FQuat(Q.Z, Q.X, Q.Y, Q.W);
}

FQuat FCardCapCoordConvert::UERotationToSpecMano(const FQuat& Q)
{
    return FQuat(Q.Y, Q.Z, Q.X, Q.W);
}

FQuat FCardCapCoordConvert::CameraRotationToUE(const FQuat& Q)
{
    return FQuat(-Q.Z, -Q.X, Q.Y, Q.W);
}

FQuat FCardCapCoordConvert::UERotationToCamera(const FQuat& Q)
{
    return FQuat(-Q.Y, Q.Z, -Q.X, Q.W);
}
