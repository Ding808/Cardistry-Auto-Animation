# Generated assets / 资源生成

[English](#english) | [简体中文](#简体中文)

## English

No prebuilt demo scene is required. After you prepare the licensed models and process a video, the plugin creates these resources for the current project:

- Left and right hand meshes, a skeleton, and materials.
- Editable hand animation.
- A review map, animation timeline, display cameras, and lighting.
- A video comparing the source with the separate local hand motions.

New assets use project content `CardistryCapture/Generated/<job-id>`. Processing records use the project's `Saved/CardistryCapture/Runs` folder. Without reliable camera parameters, these previews do not establish relative hand placement, depth, or real-world scale.

MANO models and derived hand meshes are not included in the public release. Follow the [quick start](../README.md#english) to prepare the official model files; the plugin generates scene resources locally. Models and outputs remain subject to their applicable licenses.

## 简体中文

本插件不依赖预先制作的演示场景。首次准备好获许可的模型并处理视频后，会为当前工程生成：

- 左右手网格、骨架和材质；
- 可编辑的手部动画；
- 查看地图、动画时间线、显示相机和照明；
- 原视频与两只手各自局部动作的对照预览。

新资源位于工程内容的 `CardistryCapture/Generated/<任务编号>`，处理记录位于工程的 `Saved/CardistryCapture/Runs`。缺少可靠相机参数时，这些预览不能确定双手相对位置、深度和真实尺度。

MANO 模型及其派生手部网格不随公开发行包提供。请按 [快速开始](../README.md#简体中文)完成官方模型准备；插件会在本机生成所需场景资源。模型和输出仍须遵守各自的适用许可。
