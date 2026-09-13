#include "CardCapEditorModule.h"
#include "CardCapPythonBridge.h"
#include "SCardCapPanel.h"
#include "CoreGlobals.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/Commands/UIAction.h"
#include "Framework/Docking/TabManager.h"
#include "Modules/ModuleManager.h"
#include "Logging/LogMacros.h"
#include "Textures/SlateIcon.h"
#include "ToolMenus.h"
#include "Widgets/Docking/SDockTab.h"

DEFINE_LOG_CATEGORY_STATIC(LogCardistryCaptureEditor, Log, All);

#define LOCTEXT_NAMESPACE "CardCapEditorModule"

const FName FCardCapEditorModule::TabName(TEXT("CardistryCapture"));

void FCardCapEditorModule::StartupModule()
{
    if (IsRunningCommandlet() || !GIsEditor)
    {
        return;
    }
    Bridge = MakeShared<FCardCapPythonBridge>();
    FGlobalTabmanager::Get()->RegisterNomadTabSpawner(TabName,
        FOnSpawnTab::CreateRaw(this, &FCardCapEditorModule::SpawnPanel))
        .SetDisplayName(LOCTEXT("PanelTitle", "花切动作捕获"))
        .SetTooltipText(LOCTEXT("PanelTooltip", "从视频生成双手动作，并查看场景、动画和对照视频。"))
        .SetMenuType(ETabSpawnerMenuType::Hidden);
    UToolMenus::RegisterStartupCallback(FSimpleMulticastDelegate::FDelegate::CreateRaw(this, &FCardCapEditorModule::RegisterMenus));
    bUiRegistered = true;
    UE_LOG(LogCardistryCaptureEditor, Display, TEXT("CardistryCapture panel and persistent job service registered."));
}

void FCardCapEditorModule::ShutdownModule()
{
    // The service outlives individual tabs, but never the editor module.
    if (Bridge.IsValid())
    {
        Bridge->Shutdown();
    }
    if (bUiRegistered)
    {
        UToolMenus::UnRegisterStartupCallback(this);
        UToolMenus::UnregisterOwner(this);
        if (FSlateApplication::IsInitialized())
        {
            if (const TSharedPtr<SDockTab> Tab = PanelTab.Pin()) { Tab->RequestCloseTab(); }
            FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(TabName);
        }
    }
    Panel.Reset();
    PanelTab.Reset();
    Bridge.Reset();
    bUiRegistered = false;
    UE_LOG(LogCardistryCaptureEditor, Display, TEXT("CardistryCapture editor module shutdown."));
}

void FCardCapEditorModule::RegisterMenus()
{
    if (IsRunningCommandlet() || !Bridge.IsValid()) { return; }
    FToolMenuOwnerScoped OwnerScoped(this);
    for (const FName MenuName : {FName(TEXT("LevelEditor.MainMenu.Window")), FName(TEXT("LevelEditor.MainMenu.Tools"))})
    {
        UToolMenu* Menu = UToolMenus::Get()->ExtendMenu(MenuName);
        FToolMenuSection& Section = Menu->FindOrAddSection(TEXT("CardistryCapture"));
        Section.AddMenuEntry(TEXT("CardistryCapture.OpenPanel"),
            LOCTEXT("OpenPanel", "花切动作捕获"),
            LOCTEXT("OpenPanelHint", "Cardistry Capture：从视频生成并查看双手动作。"),
            FSlateIcon(), FUIAction(FExecuteAction::CreateRaw(this, &FCardCapEditorModule::OpenPanel)));
    }
}

void FCardCapEditorModule::OpenPanel()
{
    if (bUiRegistered && Bridge.IsValid() && FSlateApplication::IsInitialized())
    {
        FGlobalTabmanager::Get()->TryInvokeTab(FTabId(TabName));
    }
}

TSharedRef<SDockTab> FCardCapEditorModule::SpawnPanel(const FSpawnTabArgs&)
{
    TSharedRef<SCardCapPanel> Content = SNew(SCardCapPanel).Bridge(Bridge);
    TSharedRef<SDockTab> Tab = SNew(SDockTab).TabRole(ETabRole::NomadTab)[Content];
    Panel = Content;
    PanelTab = Tab;
    return Tab;
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FCardCapEditorModule, CardistryCaptureEditor)
