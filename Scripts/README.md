# Setup and build scripts / 安装和编译脚本

[English](#english) | [简体中文](#简体中文)

## English

For normal setup, double-click `Setup.cmd`. The installer displays runtime terms separately from model terms, lets you choose the files, and creates a Python environment inside the plugin. It does not change your system Python. Applicable acceptances are recorded locally; checks do not ask for them again.

### Use a local runtime archive

Download `CardistryCapture-RuntimeDeps-v0.0.2.zip` from the [v0.0.2 release](https://github.com/Ding808/Cardistry-Auto-Animation/releases/tag/v0.0.2), then open PowerShell in the plugin directory:

```powershell
.\Scripts\Setup.ps1 -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.2.zip'
```

The ZIP contains five specific runtime libraries and their licenses. Other pinned dependencies still need an internet connection to PyPI or the official PyTorch index. Setup verifies the ZIP and each wheel's SHA-256. Do not bypass a failed check by changing the manifest.

### Specify all files

After obtaining the files and reading and accepting the applicable runtime, MANO, WiLoR, and SMPL-X terms, you can run:

```powershell
.\Scripts\Setup.ps1 -NonInteractive -AcceptedRuntimeLicenses -AcceptedModelLicenses `
  -Python 'C:\Python310\python.exe' `
  -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.2.zip' `
  -ManoArchive 'D:\Models\mano_v1_2.zip' `
  -WilorCheckpoint 'D:\Models\wilor_final.ckpt' `
  -WilorConfig 'D:\Models\model_config.yaml'
```

Replace these example paths with your own. `-Python` is optional and defaults to Python 3.10 through the Windows Python launcher. Already installed models do not need to be selected again. Read the [runtime distribution terms](../ThirdParty/MSVC/README.md) before using `-AcceptedRuntimeLicenses`. `-AcceptedModelLicenses` records your separate model-license acceptance. These flags do not register an account or accept a website agreement on your behalf.

| Option | Purpose |
| --- | --- |
| `-RuntimeOnly` | Prepare general dependencies without downloading restricted model sources or SMPL-X; hand reconstruction is not ready yet |
| `-CheckOnly` | Check the installed environment, models, and GPU without installing dependencies; this does not run video inference |
| `-CheckOnly -RuntimeOnly` | Check general dependencies without requiring restricted models |
| `-NonInteractive` | Disable questions and file dialogs; fail if a required argument is missing |

Restricted models stay in local `PythonPipeline/models/`. Runtime caches and setup reports use `PythonPipeline/.runtime-cache/`. Keep these files and `.venv` out of public repositories, and do not copy `.venv` between computers. Recreate the environment after moving the plugin.

### Build the plugin

Install Unreal Engine 5.4, Visual Studio C++ tools, and a Windows SDK, then run:

```powershell
.\Scripts\Build-Plugin.ps1 -EngineRoot 'C:\Program Files\Epic Games\UE_5.4' `
  -OutputDirectory 'D:\Builds\CardistryCapture'
```

The script uses Unreal Automation Tool's `BuildPlugin`. The output directory must be new or empty. Each user prepares video processing with `Setup.cmd` in the final installation directory. Validation uses UE 5.4.4.

## 简体中文

普通使用者双击 `Setup.cmd` 即可。安装器先展示运行库条款，再独立说明模型许可，之后打开文件选择窗口并创建插件自己的 Python 环境；不会修改系统 Python。适用许可的确认记录保存在本地，检查环境时不重复询问。

### 指定本地依赖包

从 [v0.0.2 Release](https://github.com/Ding808/Cardistry-Auto-Animation/releases/tag/v0.0.2) 下载 `CardistryCapture-RuntimeDeps-v0.0.2.zip` 后，在插件目录打开 PowerShell：

```powershell
.\Scripts\Setup.ps1 -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.2.zip'
```

这个 ZIP 只包含五个特定运行库和许可证。其余固定版本依赖仍需联网，从 PyPI 或 PyTorch 官方源获取。安装器校验 ZIP 和每个运行库的 SHA-256；校验失败时不要修改清单绕过检查。

### 指定所有文件

自行取得所需文件，并阅读、接受运行库及 MANO、WiLoR、SMPL-X 的适用许可后，可使用：

```powershell
.\Scripts\Setup.ps1 -NonInteractive -AcceptedRuntimeLicenses -AcceptedModelLicenses `
  -Python 'C:\Python310\python.exe' `
  -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.2.zip' `
  -ManoArchive 'D:\Models\mano_v1_2.zip' `
  -WilorCheckpoint 'D:\Models\wilor_final.ckpt' `
  -WilorConfig 'D:\Models\model_config.yaml'
```

以上均为示例路径，请替换成自己的文件位置。`-Python` 可省略，默认使用 Windows Python 启动器的 Python 3.10。已有本地模型时，不必反复指定其路径。先阅读 [运行库分发条款](../ThirdParty/MSVC/README.md)，再使用 `-AcceptedRuntimeLicenses`；模型许可另由 `-AcceptedModelLicenses` 确认。这些参数表示使用者自己的许可确认，不会替使用者注册或自动接受网站协议。

| 参数 | 用途 |
| --- | --- |
| `-RuntimeOnly` | 仅准备通用运行库，不下载受限模型源码或 SMPL-X；手部重建尚未就绪 |
| `-CheckOnly` | 只检查已安装环境、模型文件及 GPU，不安装依赖；未进行视频推理 |
| `-CheckOnly -RuntimeOnly` | 只检查通用运行库，无需受限模型 |
| `-NonInteractive` | 禁止交互输入和文件选择器，缺少必要参数即报错 |

受限模型只保存在本地 `PythonPipeline/models/`，运行库缓存和检查结果在 `PythonPipeline/.runtime-cache/`。不要把它们或 `.venv` 加入公共仓库，也不要直接复制 `.venv` 给其他电脑。移动插件后应在新位置重建环境。

### 编译插件

先安装 Unreal Engine 5.4、Visual Studio C++ 工具和 Windows SDK，然后运行：

```powershell
.\Scripts\Build-Plugin.ps1 -EngineRoot 'C:\Program Files\Epic Games\UE_5.4' `
  -OutputDirectory 'D:\Builds\CardistryCapture'
```

该脚本调用 Unreal Automation Tool 的 `BuildPlugin`，输出目录必须是新目录或空目录。每位使用者在最终安装目录运行 `Setup.cmd` 准备视频处理环境。验证环境为 UE 5.4.4。
