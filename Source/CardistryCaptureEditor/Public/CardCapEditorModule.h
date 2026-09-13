#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

class FCardCapPythonBridge;
class SCardCapPanel;
class SDockTab;
class FSpawnTabArgs;

class CARDISTRYCAPTUREEDITOR_API FCardCapEditorModule final : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    void OpenPanel();
    TSharedPtr<SCardCapPanel> GetPanel() const { return Panel.Pin(); }
    TSharedPtr<FCardCapPythonBridge> GetBridge() const { return Bridge; }
    static const FName TabName;

private:
    void RegisterMenus();
    TSharedRef<SDockTab> SpawnPanel(const FSpawnTabArgs& Args);
    TSharedPtr<FCardCapPythonBridge> Bridge;
    TWeakPtr<SCardCapPanel> Panel;
    TWeakPtr<SDockTab> PanelTab;
    bool bUiRegistered = false;
};
