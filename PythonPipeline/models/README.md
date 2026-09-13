# Local models / 本地模型目录

[English](#english) | [简体中文](#简体中文)

## English

The public package includes only the Apache-2.0-licensed `hand_landmarker.task` detection resource.

Full reconstruction requires your official MANO v1.2 ZIP and the matching WiLoR `wilor_final.ckpt` and `model_config.yaml`. Run the plugin's `Scripts/Setup.cmd` and select those three files. Setup verifies them and prepares both hands and the required backend sources; no manual folder changes are needed.

After setup, this directory contains `MANO_LEFT.pkl`, `MANO_RIGHT.pkl`, their installation manifest, WiLoR files, and local caches. These files have separate license restrictions and are excluded from version control and public plugin packaging. Do not add them to a public repository.

See the [quick start](../../README.md#english) and [third-party notices](../../THIRD_PARTY_NOTICES.md). Preparing these models does not verify camera parameters or real-world hand scale.

## 简体中文

公开包仅提供 Apache-2.0 授权的 `hand_landmarker.task` 检测资源。

完整重建需要用户自行取得 MANO v1.2 官方 ZIP，以及匹配的 WiLoR `wilor_final.ckpt`、`model_config.yaml`。运行插件内 `Scripts/Setup.cmd`，选择这三个文件；安装器会校验文件、准备左右手模型及所需后端源码，不需要手动修改目录结构。

准备后，此目录会包含 `MANO_LEFT.pkl`、`MANO_RIGHT.pkl`、安装清单、WiLoR 文件及本地缓存。这些文件受各自许可限制，已从版本控制和插件公开打包规则中排除，不要把它们加入公共仓库。

使用方法见 [快速开始](../../README.md#简体中文)，许可说明见 [第三方组件](../../THIRD_PARTY_NOTICES.md)。准备好模型不代表相机参数或真实手部尺度已经验证。
