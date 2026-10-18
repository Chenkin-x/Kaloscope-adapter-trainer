# 来源与整理范围

本包是原 `kaloscope-dinov3` 工作区中风格适配器代码的可移植整理版，新增了公共数据格式和入口。
不是原日期目录的完整镜像，也不把新训练入口称作历史实验的逐位重放。

| 包中模块 | 原实现依据 | 整理内容 |
| --- | --- | --- |
| `adapter.py` | `krea2_ai0_lora_20261008/adapter.py` | 保留Z3状态键；移除LoRA分支；形状校验 |
| `backends/krea2.py` | `krea2_ai0_lora_20261008/backend.py` | 保留原生前向；去服务器路径；配置/设备化 |
| `backends/moss.py` | `moss_z3_demo_20261017/backend.py` | 保留checkpoint安全绑定；显式可选注入映射 |
| `backends/api.py` | Krea2缓存、Moss prepare/render | 分离文本/VAE/time/sampling接口 |
| `metrics.py` | `krea2_z3_multires_20261010/metrics_v2.py` | 保留CSD指纹、对称视图与画师聚类bootstrap |
| `kaloscope_dinov3/` | ComfyUI集成中的DINOv3推理子集 | 随包提供，不依赖原仓库导入路径 |
| `_vendor/moss.py` | 本地MOSS-Image-Base原生pipeline | 按原快照复制，Apache-2.0保留 |
| data/prepare/train/infer/evaluate | 上述训练算法及全量/诊断实现 | 新的相对路径、JSON清单、缓存和CLI |

源文件SHA与实验原记录位置见 [provenance.json](../evidence/provenance.json)。
证据JSON仅脱敏私人主机地址/机器绝对路径，数值保持原样；脱敏前后各有SHA。
`evidence/*_config.json`仅供追溯，带`<local-path>`等占位不能执行；可执行配置在 `configs/`。

本次整理不包含：原数据图片、全部图墙、巨型权重、私有主机凭证、远程队列/SSH控制器、原项目.git、
多分辨率在线Arrow数据生产器、patch-token/LoRA/结构loss等其他实验路线。
它们的负结果或对照如与Z3选择相关，会以指标和文字保留，不自动变为本包功能。

## 原生依赖与许可

- DINOv3推理子集来自原仓库/现有集成，保留Meta版权及DINOv3 License。
- Moss pipeline复制自本地模型快照，其模型卡与LICENSE标明Apache-2.0；
  代码来源摘要见 [Moss官方仓库](https://github.com/OpenMOSS/MOSS-Image)，模型资产单独准备。
- Krea2原生后端依赖外部Musubi。原生小模型测试基于本地源码快照，提供文件哈希，
  本地Git子模块元数据不完整，因此**没有捏造已验证的上游commit号**。
- CSD代码和权重按历史指纹在运行时读取，不把第三方模型权重打包进GitHub。
- 保留原仓库根许可，不为未知来源文件随意新设宽松许可。

## 验收边界

本次能验证：公共包可构建、独立导入、CLI、文件/链接和来源校验；两类真实原生小模型的数值测试；
小尺寸Moss训练与逐位恢复；两进程CPU非整除batch等价测试。

本次不能冒充已完成：新包在真实Krea2/Moss大权重上的CUDA训练/成图、完整真实缓存GPU路径、
Windows/低显存offload/量化组合、三seed重训或新的人工盲评。
本地环境没有可用CUDA。本包保留目标GPU的自动原生parity入口，使用前运行3步预检。
详细执行结果见 [VALIDATION.md](VALIDATION.md)。
