#include "Modules/ModuleManager.h"
#include "Logging/LogMacros.h"

DEFINE_LOG_CATEGORY_STATIC(LogCardistryCapture, Log, All);

class FCardistryCaptureModule final : public IModuleInterface
{
public:
    virtual void StartupModule() override
    {
        UE_LOG(LogCardistryCapture, Display, TEXT("CardistryCapture runtime module loaded (0.0.2)."));
    }

    virtual void ShutdownModule() override
    {
        UE_LOG(LogCardistryCapture, Display, TEXT("CardistryCapture runtime module shutdown."));
    }
};

IMPLEMENT_MODULE(FCardistryCaptureModule, CardistryCapture)
