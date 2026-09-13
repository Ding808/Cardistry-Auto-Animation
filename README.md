# Cardistry Auto Animation

在 Unreal Engine 中选择一段视频，生成可编辑的手部动画和配套查看场景。

**v0.0.1 支持 Windows x64 / Unreal Engine 5.4，已在 5.4.4 验证。** 当前输出是动作初稿：缺少可靠相机参数时，左右手分别显示局部动作，不能据此测量两手距离或真实尺寸。扑克牌重建和物理模拟尚未提供。

## 快速开始

### 1. 安装插件

从 [Releases](https://github.com/Ding808/Cardistry-Auto-Animation/releases/tag/v0.0.1) 下载 `CardistryCapture-v0.0.1.zip`，解压得到 `CardistryCapture` 文件夹，将整个文件夹放入：

```text
你的工程/
  Plugins/
    CardistryCapture/
      CardistryCapture.uplugin
      Source/
      PythonPipeline/
      Scripts/
```

安装 Visual Studio 2022 的 **使用 C++ 的游戏开发** 工作负载和 Windows SDK。关闭编辑器，重新生成工程文件，编译工程的 **Development Editor / Win64**。纯蓝图工程可先添加一个空 C++ 类。

也可将插件放入引擎的 `Engine/Plugins/Marketplace/CardistryCapture`；首次准备环境时需要该目录的写入权限。日常生成的模型、动画和场景保存到当前工程，不写入引擎内容目录。新手建议使用工程内安装。

### 2. 首次准备处理环境和模型

先安装 **64 位 Python 3.10**，并准备支持 **CUDA 12.8** 的 NVIDIA 驱动。UE 自带的 Python 不用于视频处理。

本发行包包含可公开分发的检测资源和资源生成代码。完整手部重建还需要自行取得下列官方文件：

| 文件 | 获取位置 |
| --- | --- |
| MANO v1.2 官方 ZIP，包含左右手 | [MANO 官网](https://mano.is.tue.mpg.de/)：注册并按其许可下载 |
| `wilor_final.ckpt` 和 `model_config.yaml` | [WiLoR 官方项目](https://github.com/rolpotamias/WiLoR)：按模型下载说明获取 |

双击插件内 **`Scripts/Setup.cmd`**，按提示安装依赖、确认适用许可并选择上述文件。安装器会创建独立环境、校验文件并准备左右手；首次需要联网下载运行库。准备完成后无需反复安装。

MANO、WiLoR、SMPL-X 的许可与插件代码许可分开，当前重建组合面向符合这些许可的非商业研究用途。公开包不含受限权重、MANO 手模或其派生演示资产；它们由使用者在本机准备。详情见 [第三方资源说明](THIRD_PARTY_NOTICES.md)。

### 3. 处理视频

1. 打开工程，在 **窗口 / Window → 花切动作捕获** 打开面板。
2. 点击 **选择视频…**，然后点击 **开始处理**。
3. 完成后点击 **预览视频** 查看“原视频 / 左手 / 右手”，或点击 **打开场景** 播放、拖动动画时间线。

每次任务自动生成左右手网格、骨架、肤色材质、动画、查看场景、相机和照明，不需要另找场景模板或手模型素材。

黄色区段建议重点复核。检测、插值和保持的动作有不同色标；补出的动作不代表该帧被可靠识别。

## 查看和保存结果

| 入口 | 用途 |
| --- | --- |
| 打开场景 | 打开查看地图和时间线，定位第 0 帧；局部模式默认显示左手 |
| 预览视频 | 同时对照原视频与两只手各自的局部动作 |
| 结果文件夹 | 查看本次预览、交换数据和处理记录 |
| 查看日志 | 查看失败原因，修正后重新处理 |

局部场景中切换右手：选中手部 actor 的骨骼网格组件，在 **Local Hand Preview → Show right hand locally** 切换。此操作只改变显示，不确定双手空间关系。局部模式下“打开动画”按钮停用，避免把两个独立原点误看成两手重合；请用查看场景或预览视频。

- UE 资源：工程内容 `CardistryCapture/Generated/<任务编号>`。
- 处理文件：工程 `Saved/CardistryCapture/Runs/<任务编号>`。
- 可分享的本地预览：本次结果目录的 `Preview/review.mp4`。分享时仍须遵守素材和模型的适用许可。

每次处理保存独立结果。关闭面板不会中断任务；关闭整个编辑器会停止任务。

## 常见问题

**编译后没有菜单？** 在编辑 → 插件中启用 Cardistry Capture，并重新启动编辑器。该版本只验证了 UE 5.4.4；其他引擎版本可能需要适配源码。

**提示缺少 Python 或模型？** 在插件所在位置重新运行 `Scripts/Setup.cmd`，完成最后的就绪检查。不要复制其他电脑的 `.venv`；移动插件后也应重建环境。

**首次依赖下载较慢？** 可从同一 Release 手动下载 `CardistryCapture-RuntimeDeps-v0.0.1.zip`，安装时指定本地包，避免重复下载。详见 [安装参数](Scripts/README.md)。

**结果不贴合原片？** v0.0.1 尚未解决遮挡、左右手身份和所有手指姿态错误。优先使用清晰、光线均匀、双手持续入镜的视频，并对照来源色标检查补帧。

## 编译与许可

插件以源码形式提供。命令行编译可使用 `Scripts/Build-Plugin.ps1`；参数见 [脚本说明](Scripts/README.md)。编辑器模块依赖 UE 自带的 Level Sequence Editor 与 Interchange 导入组件。

插件自有代码见 [LICENSE](LICENSE)，第三方源码、检测资源和可选运行库保留各自许可。更新记录见 [CHANGELOG.md](CHANGELOG.md)。

