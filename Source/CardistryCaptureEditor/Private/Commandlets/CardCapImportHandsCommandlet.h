#pragma once

#include "CoreMinimal.h"
#include "Commandlets/Commandlet.h"
#include "CardCapImportHandsCommandlet.generated.h"

/** Isolated batch import for research hand rigs; never called by an editor tick. */
UCLASS()
class UCardCapImportHandsCommandlet : public UCommandlet
{
    GENERATED_BODY()
public:
    UCardCapImportHandsCommandlet();
    virtual int32 Main(const FString& Params) override;
};
