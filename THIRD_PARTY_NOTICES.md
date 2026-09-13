# 第三方组件

仓库根目录的 MIT 许可仅覆盖本项目自行编写的代码。下列第三方组件、用户另行安装的模型及其输出，继续适用各自的许可。本项目未获第三方作者或厂商背书。

## 随源码包提供

| 组件 | 用途 | 许可与文件 |
| --- | --- | --- |
| Google MediaPipe Hand Landmarker，float16 full，version 1 | 视频中双手检测与关键点 | Apache-2.0；见 [模型说明](ThirdParty/MediaPipe/README.md) 和 [完整许可](ThirdParty/MediaPipe/LICENSE) |
| timm v0.6.12 的三个辅助源码 | 加载手部重建后端所需的层辅助函数 | Apache-2.0；见 `ThirdParty/timm/` 中的原始源码、完整许可与出处。其来源于 PyTorch 的函数同时保留 PyTorch BSD 通知 |

## 单独提供的 Windows Python 运行依赖

Release 中的 RuntimeDeps 安装包保留各 wheel 内的许可、版权及第三方通知；附加通知、固定来源和打包差异见该安装包中的 `RUNTIME_THIRD_PARTY_NOTICES.md`。这些文件不受本项目 MIT 许可重新授权。

| 组件 | 许可概要 | 本发行配置 |
| --- | --- | --- |
| NumPy 1.26.4 | BSD 及随包第三方许可 | MSVC 构建，无外部 BLAS/LAPACK |
| MediaPipe 0.10.31 | Apache-2.0 及随包通知 | 仅修正 Windows wheel 标签与包记录 |
| sounddevice 0.5.5 / PortAudio | MIT | 排除可选 ASIO DLL；保留普通 PortAudio DLL |
| Manifold3D 3.5.3 | Apache-2.0；第三方组件各自许可 | 原 Windows Python 3.10 wheel |
| Microsoft Visual C++ Runtime | Microsoft 专有可再分发代码条款 | Manifold wheel 所带 `msvcp140.dll`；[适用说明与全文](ThirdParty/MSVC/README.md) |
| SciPy 1.15.2 / LLVM runtime | BSD / Apache-2.0 WITH LLVM-exception 及第三方许可 | 原科学计算内核，添加本地 DLL 加载入口 |
| Intel oneMKL 2025.2 | Intel Simplified Software License（October 2022）及第三方许可 | 原字节 CPU DLL，sequential / LP64 |

安装器还会从其官方发布渠道安装 PyTorch、OpenCV 和其他 Python 依赖；各安装包保留自身许可。OpenCV wheel 携带的 FFmpeg 具有独立 LGPL 条款。此仓库不分发这些运行库二进制，也不分发 Unreal Engine 源码或引擎二进制。

## 用户另行提供或安装

- **MANO 左右手模型**：由用户从 [MANO 官网](https://mano.is.tue.mpg.de/) 自行取得并接受 [适用条款](https://mano.is.tue.mpg.de/license.html)。模型、其缓存和由其生成的手模不包含在公开仓库或 Release 中。MANO 条款限制用途和再分发。
- **WiLoR 源码、checkpoint、配置与 mean parameters**：适用 [官方仓库说明](https://github.com/rolpotamias/WiLoR/tree/fcb911312a38fa8badd30d9656a167485d61b8f9) 及 [模型许可](https://github.com/rolpotamias/WiLoR/blob/fcb911312a38fa8badd30d9656a167485d61b8f9/license.txt)。仓库模型通知为 CC-BY-NC-ND-4.0；checkpoint 中第三方模型内容还受其独立许可约束。本项目不包含这些文件。
- **SMPL-X 软件**：从其上游单独安装，适用 [SMPL-X 许可](https://github.com/vchoutas/smplx/blob/main/LICENSE)。本项目不分发其 wheel 或源码。

用户自备模型不等于取得商业授权。使用这些后端生成手模或动画前，使用者应确保用途符合各模型和软件的许可；本项目的 MIT 许可不扩大这些权利。安装器不会下载或使用 Ultralytics detector。

## 研究来源

- Romero, Tzionas, Black. *Embodied Hands: Modeling and Capturing Hands and Bodies Together*. ACM Transactions on Graphics, 2017.
- Potamias, Zhang, Deng, Zafeiriou. *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild*. CVPR, 2025.
- Pavlakos et al. *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image*. CVPR, 2019.

根据 MANO 许可，使用该模型的出版物须作适当引用；媒体项目还应按随模型条款保留所要求的署名。
