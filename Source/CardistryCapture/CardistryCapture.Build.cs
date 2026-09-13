using UnrealBuildTool;

public class CardistryCapture : ModuleRules
{
    public CardistryCapture(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] {
            "Core", "CoreUObject", "Engine", "Json", "JsonUtilities",
            "PhysicsCore", "Chaos", "AnimationCore"
        });
        PrivateDependencyModuleNames.Add("Projects");
    }
}
