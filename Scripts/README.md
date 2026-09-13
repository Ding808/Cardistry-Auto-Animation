# 安装和编译脚本

普通使用者双击 `Setup.cmd` 即可。安装器先展示运行库条款，再独立说明模型许可，之后打开文件选择窗口并创建插件自己的 Python 环境；不会修改系统 Python。同一版本已接受的条款会记录在本地，检查环境时不重复询问。

## 指定本地依赖包

从同一个 `v0.0.1` Release 下载 `CardistryCapture-RuntimeDeps-v0.0.1.zip` 后，在插件目录打开 PowerShell：

```powershell
.\Scripts\Setup.ps1 -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.1.zip'
```

这个 ZIP 只包含五个特定运行库和许可证。其余固定版本依赖仍需联网，从 PyPI 或 PyTorch 官方源获取。安装器校验 ZIP 和每个运行库的 SHA-256；校验失败时不要修改清单绕过检查。

## 指定所有文件

自行阅读、取得并接受 MANO、WiLoR、SMPL-X 的适用许可后，可使用：

```powershell
.\Scripts\Setup.ps1 -NonInteractive -AcceptedRuntimeLicenses -AcceptedModelLicenses `
  -Python 'C:\Python310\python.exe' `
  -RuntimeArchive 'D:\Downloads\CardistryCapture-RuntimeDeps-v0.0.1.zip' `
  -ManoArchive 'D:\Models\mano_v1_2.zip' `
  -WilorCheckpoint 'D:\Models\wilor_final.ckpt' `
  -WilorConfig 'D:\Models\model_config.yaml'
```

`-Python` 可省略，默认使用 Windows Python 启动器的 Python 3.10。已有本地模型时，不必反复指定其路径。先阅读 [运行库分发条款](../ThirdParty/MSVC/README.md)，再使用 `-AcceptedRuntimeLicenses`；模型许可另由 `-AcceptedModelLicenses` 确认。这些参数表示使用者自己的许可确认，不会替使用者注册或自动接受网站协议。

| 参数 | 用途 |
| --- | --- |
| `-RuntimeOnly` | 仅准备通用运行库，不下载受限模型源码或 SMPL-X；手部重建尚未就绪 |
| `-CheckOnly` | 只检查已安装环境、模型文件及 GPU，不安装依赖；未进行视频推理 |
| `-CheckOnly -RuntimeOnly` | 只检查通用运行库，无需受限模型 |
| `-NonInteractive` | 禁止交互输入和文件选择器，缺少必要参数即报错 |

受限模型只保存在本地 `PythonPipeline/models/`，运行库缓存和检查结果在 `PythonPipeline/.runtime-cache/`。不要把它们或 `.venv` 加入公共仓库，也不要直接复制 `.venv` 给其他电脑。移动插件后建议在新位置重新准备环境。

## 编译可分发的插件

先安装对应版本的 Unreal Engine、Visual Studio C++ 工具和 Windows SDK，然后运行：

```powershell
.\Scripts\Build-Plugin.ps1 -EngineRoot 'C:\Program Files\Epic Games\UE_5.4' `
  -OutputDirectory 'D:\Builds\CardistryCapture'
```

该脚本调用 Unreal Automation Tool 的 `BuildPlugin`。首次处理环境由每位使用者在最终安装目录运行 `Setup.cmd` 准备。
