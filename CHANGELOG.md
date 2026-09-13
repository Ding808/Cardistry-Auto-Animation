# Changelog / 更新记录

[English](#english) | [简体中文](#简体中文)

## English

### v0.0.3

- Restore a common display of both hands as the default review, with a source comparison and independent overview.
- Choose a display focal assumption per clip from frame-by-frame hand-geometry checks, independently of image orientation; report failures rather than hide them.
- Keep camera intrinsics, distortion, metric scale, and global translation unknown when uncalibrated. Store display assumptions separately from capture data.
- Allow the assumed focal and local-hand view to be adjusted in the saved review scene; retain the optional separate-hand video layout.
- Preserve native crop-camera diagnostics and observation/interpolation labels.

### v0.0.2

- Use English by default for the capture panel, controls, and processing messages.
- Provide English-first documentation with complete Simplified Chinese sections and language jump links.
- Update source and runtime archive references to v0.0.2.
- Keep reconstruction algorithms, parameters, and output geometry unchanged. Hand identity, occlusion, pose errors, and unresolved shared space remain limitations.

### v0.0.1

- Select a video in an editor panel and generate a hand animation draft.
- Automatically create hand meshes, animation, materials, a review scene, lighting, and display cameras.
- Compare the source with separate local hand motions, including source labels and review ranges.
- Provide processing progress, cancellation, result opening, and log access.
- Install an isolated Python runtime and verify model files.
- Save all newly generated assets in the current project.
- Keep relative hand placement, depth, and metric scale unknown when reliable camera parameters are unavailable.

Validated environment: Windows x64, Unreal Engine 5.4.4, Python 3.10, and NVIDIA CUDA. Card reconstruction, physics simulation, and accurate shared-space reconstruction of both hands were not delivered in v0.0.1 and remain unavailable in v0.0.2.

## 简体中文

### v0.0.3

- 默认恢复双手共同显示，提供原视频对照与独立总览。
- 根据每段素材的逐帧手部几何检查选择显示焦距假设，不依赖画幅方向，并如实标出越界帧。
- 未标定时，相机内参、畸变、米制尺度与全局平移继续保持未知；显示假设单独保存。
- 在保存后的查看场景中调整假设焦距、切换单手查看，并保留可选的左右手分栏视频。
- 保留原生裁切相机诊断值与检测、插值等来源标注。

### v0.0.2

- 捕获面板、按钮和处理消息默认使用英文。
- 文档以英文为默认入口，并提供完整简体中文内容与语言跳转链接。
- 源码包和运行库包的引用更新为 v0.0.2。
- 重建算法、参数和输出几何保持不变。左右手身份、遮挡、姿态错误和未确定的共同空间仍是当前限制。

### v0.0.1

- 在编辑器面板中选择视频，生成手部动作初稿。
- 自动创建手部网格、动画、材质、查看场景、照明和显示相机。
- 提供原视频与两只手各自局部动作的对照、来源标签及待复核区段。
- 支持处理进度、取消、结果打开与日志查看。
- 提供隔离的 Python 处理环境安装流程和模型文件校验。
- 所有新生成资产归属于当前工程。
- 缺少可靠相机参数时，双手相对位置、深度与米制尺度保持未知。

验证环境：Windows x64、Unreal Engine 5.4.4、Python 3.10、NVIDIA CUDA。扑克牌重建、物理模拟与准确双手共同空间重建不属于 v0.0.1 已完成功能，v0.0.2 也尚未提供。
