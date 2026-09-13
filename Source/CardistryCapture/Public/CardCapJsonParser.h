#pragma once
#include "CardCapData.h"

class CARDISTRYCAPTURE_API FCardCapJsonParser
{
public:
    // Failure leaves OutCapture untouched. Errors include a JSON field/index path.
    // An empty mapping path loads this plugin's Config/BoneMapping_UE5Mannequin.json.
    static bool ParseFile(const FString& Filename, FCardCapCapture& OutCapture,
        FString& OutError, const FString& BoneMappingPath = FString());
    static bool ParseString(const FString& Json, FCardCapCapture& OutCapture,
        FString& OutError, const FString& BoneMappingPath = FString());
    static bool LoadBoneMapping(const FString& Filename, FCardCapBoneMapping& OutMapping,
        FString& OutError);
};
