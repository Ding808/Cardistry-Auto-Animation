#pragma once
#include "CoreMinimal.h"

struct FCardCapCapture;
class UAnimSequence;
class USkeletalMesh;
class FJsonObject;

class CARDISTRYCAPTUREEDITOR_API FCardCapAnimBaker
{
public:
    // Synchronous batch primitive. Invoke in a separate editor commandlet process.
    // Never overwrites an existing asset. The last sample is held for one frame.
    static UAnimSequence* Bake(const FCardCapCapture& Capture, USkeletalMesh* Mesh,
        const FString& PackageName, FString& OutError);

    // Evaluate the actual raw and compressed asset, including half-frame samples.
    static bool Validate(const FCardCapCapture& Capture, USkeletalMesh* Mesh,
        UAnimSequence* Animation, TSharedRef<FJsonObject> Report, FString& OutError);
};
