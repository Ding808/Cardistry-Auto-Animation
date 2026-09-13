# MediaPipe Hand Landmarker

本插件包含 Google MediaPipe Hand Landmarker 的 **full / float16 / version 1** 检测模型。

- 文件：`PythonPipeline/models/hand_landmarker.task`
- 大小：7,819,105 字节
- SHA-256：`fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- [Google 原始模型下载](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
- [官方模型文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [官方模型卡](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf)
- 许可：[Apache License 2.0](LICENSE)，模型卡第 2 页明确标示该许可。

The MediaPipe Authors / Google。

模型保持原字节，用于定位手部和生成关键点。它不包含可导入 Unreal Engine 的 MANO 网格，也不证明视频中的真实手长、相机焦距或双手空间距离。遮挡、持物和运动模糊可能使检测不完整；处理结果需要使用者检查。
