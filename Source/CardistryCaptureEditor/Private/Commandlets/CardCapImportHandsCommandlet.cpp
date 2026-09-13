#include "Commandlets/CardCapImportHandsCommandlet.h"

#include "Animation/Skeleton.h"
#include "AssetCompilingManager.h"
#include "AssetImportTask.h"
#include "AssetToolsModule.h"
#include "Dom/JsonObject.h"
#include "Engine/SkeletalMesh.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/PackageName.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"
#include "UObject/Package.h"
#include "UObject/StrongObjectPtr.h"

DEFINE_LOG_CATEGORY_STATIC(LogCardCapImportHands, Log, All);

namespace
{
    TArray<TSharedPtr<FJsonValue>> VectorJson(const FVector& Value)
    {
        return {MakeShared<FJsonValueNumber>(Value.X), MakeShared<FJsonValueNumber>(Value.Y), MakeShared<FJsonValueNumber>(Value.Z)};
    }

    TArray<TSharedPtr<FJsonValue>> QuaternionJson(const FQuat& Value)
    {
        return {MakeShared<FJsonValueNumber>(Value.X), MakeShared<FJsonValueNumber>(Value.Y),
            MakeShared<FJsonValueNumber>(Value.Z), MakeShared<FJsonValueNumber>(Value.W)};
    }
}

UCardCapImportHandsCommandlet::UCardCapImportHandsCommandlet()
{
    IsClient = false;
    IsServer = false;
    IsEditor = true;
    LogToConsole = true;
    ShowErrorCount = true;
    HelpDescription = TEXT("Import a local research hand GLB and report its actual UE skeleton.");
    HelpUsage = TEXT("-run=CardCapImportHands -MeshSource=<glb> -Destination=<content folder> -Evidence=<json>");
}

int32 UCardCapImportHandsCommandlet::Main(const FString& Params)
{
    FString MeshSource, Destination, Evidence;
    if (!FParse::Value(*Params, TEXT("MeshSource="), MeshSource)
        || !FParse::Value(*Params, TEXT("Destination="), Destination)
        || !FParse::Value(*Params, TEXT("Evidence="), Evidence))
    {
        UE_LOG(LogCardCapImportHands, Error, TEXT("%s"), *HelpUsage);
        return 1;
    }
    MeshSource = FPaths::ConvertRelativePathToFull(MeshSource);
    Evidence = FPaths::ConvertRelativePathToFull(Evidence);
    if (!FPaths::FileExists(MeshSource) || !FPackageName::IsValidLongPackageName(Destination))
    {
        UE_LOG(LogCardCapImportHands, Error, TEXT("Source file or content destination is invalid."));
        return 2;
    }
    const bool bReplaceExisting = FParse::Param(*Params, TEXT("ReplaceExisting"));
    const FString DestinationDirectory = FPackageName::LongPackageNameToFilename(Destination);
    if (!bReplaceExisting && IFileManager::Get().DirectoryExists(*DestinationDirectory))
    {
        TArray<FString> ExistingFiles;
        IFileManager::Get().FindFilesRecursive(ExistingFiles, *DestinationDirectory, TEXT("*.uasset"), true, false);
        if (!ExistingFiles.IsEmpty())
        {
            UE_LOG(LogCardCapImportHands, Error, TEXT("Destination already contains assets; choose a new research folder or explicitly request replacement."));
            return 3;
        }
    }

    TStrongObjectPtr<UAssetImportTask> Task(NewObject<UAssetImportTask>());
    Task->Filename = MeshSource;
    Task->DestinationPath = Destination;
    Task->bAutomated = true;
    Task->bSave = true;
    Task->bAsync = false;
    Task->bReplaceExisting = bReplaceExisting;
    Task->bReplaceExistingSettings = bReplaceExisting;
    // This synchronous engine import runs only in this separate commandlet
    // process, never in the user's interactive editor or an editor UI callback.
    FAssetToolsModule::GetModule().Get().ImportAssetTasks({Task.Get()});
    FAssetCompilingManager::Get().FinishAllCompilation();

    TArray<TSharedPtr<FJsonValue>> Objects;
    TArray<TSharedPtr<FJsonValue>> Meshes;
    bool bAllSaved = true;
    for (UObject* Object : Task->GetObjects())
    {
        if (!Object)
        {
            continue;
        }
        const FString PackageFile = FPackageName::LongPackageNameToFilename(Object->GetOutermost()->GetName(), FPackageName::GetAssetPackageExtension());
        const bool bSaved = FPaths::FileExists(PackageFile);
        bAllSaved &= bSaved;
        TSharedRef<FJsonObject> Entry = MakeShared<FJsonObject>();
        Entry->SetStringField(TEXT("object_path"), Object->GetPathName());
        Entry->SetStringField(TEXT("class"), Object->GetClass()->GetName());
        Entry->SetStringField(TEXT("package_file"), PackageFile);
        Entry->SetBoolField(TEXT("saved_on_disk"), bSaved);
        Objects.Add(MakeShared<FJsonValueObject>(Entry));
        if (USkeletalMesh* Mesh = Cast<USkeletalMesh>(Object))
        {
            TSharedRef<FJsonObject> MeshEntry = MakeShared<FJsonObject>();
            MeshEntry->SetStringField(TEXT("mesh_path"), Mesh->GetPathName());
            MeshEntry->SetStringField(TEXT("skeleton_path"), Mesh->GetSkeleton() ? Mesh->GetSkeleton()->GetPathName() : FString());
            const FReferenceSkeleton& Reference = Mesh->GetRefSkeleton();
            TArray<TSharedPtr<FJsonValue>> Bones;
            for (int32 Index = 0; Index < Reference.GetNum(); ++Index)
            {
                const FTransform& Transform = Reference.GetRefBonePose()[Index];
                TSharedRef<FJsonObject> Bone = MakeShared<FJsonObject>();
                Bone->SetStringField(TEXT("name"), Reference.GetBoneName(Index).ToString());
                Bone->SetNumberField(TEXT("parent_index"), Reference.GetParentIndex(Index));
                Bone->SetArrayField(TEXT("translation_cm"), VectorJson(Transform.GetTranslation()));
                Bone->SetArrayField(TEXT("rotation_quat_xyzw"), QuaternionJson(Transform.GetRotation()));
                Bone->SetArrayField(TEXT("scale"), VectorJson(Transform.GetScale3D()));
                Bones.Add(MakeShared<FJsonValueObject>(Bone));
            }
            MeshEntry->SetNumberField(TEXT("bone_count"), Reference.GetNum());
            MeshEntry->SetArrayField(TEXT("bones"), Bones);
            Meshes.Add(MakeShared<FJsonValueObject>(MeshEntry));
        }
    }
    const bool bSuccess = !Meshes.IsEmpty() && bAllSaved;
    TSharedRef<FJsonObject> Report = MakeShared<FJsonObject>();
    Report->SetStringField(TEXT("status"), bSuccess ? TEXT("passed") : TEXT("failed"));
    Report->SetStringField(TEXT("scope"), TEXT("Actual UE research skeletal mesh import and saved reference pose; not animation or accuracy validation"));
    Report->SetStringField(TEXT("source"), MeshSource);
    Report->SetArrayField(TEXT("objects"), Objects);
    Report->SetArrayField(TEXT("skeletal_meshes"), Meshes);
    FString Json;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
    if (!FJsonSerializer::Serialize(Report, Writer)
        || !IFileManager::Get().MakeDirectory(*FPaths::GetPath(Evidence), true)
        || !FFileHelper::SaveStringToFile(Json, *Evidence, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    {
        UE_LOG(LogCardCapImportHands, Error, TEXT("Failed to write import evidence: %s"), *Evidence);
        return 4;
    }
    UE_LOG(LogCardCapImportHands, Display, TEXT("Research mesh import %s; %d skeletal mesh(es), evidence %s"), bSuccess ? TEXT("passed") : TEXT("failed"), Meshes.Num(), *Evidence);
    return bSuccess ? 0 : 5;
}
