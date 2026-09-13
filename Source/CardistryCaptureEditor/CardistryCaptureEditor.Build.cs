using UnrealBuildTool;

public class CardistryCaptureEditor : ModuleRules
{
    public CardistryCaptureEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        AddEngineThirdPartyPrivateStaticDependencies(Target, "OpenSSL");
        PublicDependencyModuleNames.AddRange(new[] { "Core", "Slate", "SlateCore" });
        PrivateDependencyModuleNames.AddRange(new[] {
            "CoreUObject", "Engine", "CardistryCapture", "UnrealEd",
            "EditorStyle", "AnimationDataController", "DesktopPlatform", "ToolMenus", "ApplicationCore", "Projects",
            "AssetTools", "InterchangeEngine", "AssetRegistry", "Json", "JsonUtilities",
            "LevelSequence", "MovieScene", "MovieSceneTracks", "RenderCore", "RHI", "ImageWrapper",
            "Sequencer", "LevelSequenceEditor", "AutomationDriver", "InputCore"
        });
    }
}
