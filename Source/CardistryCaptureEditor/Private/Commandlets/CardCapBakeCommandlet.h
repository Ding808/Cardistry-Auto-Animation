#pragma once
#include "Commandlets/Commandlet.h"
#include "CardCapBakeCommandlet.generated.h"

UCLASS()
class UCardCapBakeCommandlet : public UCommandlet
{
    GENERATED_BODY()
public:
    UCardCapBakeCommandlet();
    virtual int32 Main(const FString& Params) override;
};
