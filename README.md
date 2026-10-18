# Kaloscope Style Adapter · Z3

将 **Kaloscope 画师表征 → 风格 token → 冻结图像生成模型** 的训练流程整理为独立 Python 包。
默认示例使用 **Krea2 Raw**；另有可运行的 **MOSS-Image-Base** 后端，用于演示如何移植到其他模型。
不需要原研究仓库、ComfyUI、服务器脚本或原数据集目录。

本包包含源码、配置、数据格式、训练/恢复、单参考推理、评估、历史实验数据与测试。
**不附带模型权重、训练图片、画师名称原型库或原始大规模数据集。**
整理快照日期：2026-10-17。

## 从哪里开始

- [Krea2 安装、数据和训练](docs/TRAINING.md)
- [更换图像模型：以 Moss 为例](docs/MODEL_ADAPTATION.md)
- [主要可调参数与试验顺序](docs/PARAMETERS.md)
- [Z3 默认参数如何得出：实验数据与限制](docs/Z3_EVIDENCE.md)
- [评估方法、指标含义与复现](docs/EVALUATION.md)
- [来源、第三方代码与验证边界](docs/PROVENANCE.md)

## 核心结构

```text
参考图 → Kaloscope artist-v1 → 归一化 256D projector
                                  ↓
                         8 个 256D style token
                                  ↓
目标图 → 模型专用 VAE → 带噪 latent → 冻结 DiT → flow velocity
                                  ↑
           每个完整 block 后，cross-attention 只更新 image token
```

Krea2 默认 28 层、4 组共享 Q/out，每 7 层复用一组；宽度 256，4 个注意力头，
13,308,672 个可训练参数。主干、文本编码器、VAE 和 Kaloscope 均冻结，不训练 LoRA。
适配器零输出初始化，初始预测与原生模型一致。详见 [adapter.py](src/kaloscope_style/adapter.py)。

## 最短使用路径

先根据 [安装说明](docs/TRAINING.md) 准备 Python、PyTorch、Musubi 源码和本地权重。
修改 `configs/krea2_z3.json` 路径及 `examples/manifest.json` 后：

```bash
python -m kaloscope_style.prepare --config configs/krea2_z3.json --kind audit
python -m kaloscope_style.prepare --config configs/krea2_z3.json --kind all

# 单卡训练也保持 global_batch=16，通过梯度累积实现。
python -m kaloscope_style.train --config configs/krea2_z3.json --stop-after 3
python -m kaloscope_style.train --config configs/krea2_z3.json --resume

# 或在新 output 目录使用四卡：global_batch=16, micro_batch=1 → 每卡累积4次。
torchrun --standalone --nproc_per_node=4 -m kaloscope_style.train --config configs/krea2_z3.json
```

```bash
python -m kaloscope_style.infer \
  --config configs/krea2_z3.json \
  --checkpoint runs/krea2-z3/adapter_step00002048.pt \
  --reference data/reference.png --prompt "a person reading in a garden" \
  --output runs/example.png
```

本包的 `krea2_z3.json` 是 **Z3 结构与核心参数的从零训练示例**：2048 步、20 步 warmup。
它不是历史 Z3 step2024 的逐位复现。历史 Z3 是从 AI0 step1000 **继承权重和 AdamW** 再训练
1024 步；Z3C、全量 step12006 和后来的多分辨率检查点又是不同阶段。
`initial_checkpoint` 只加载适配器权重并新建 AdamW；`--resume` 才恢复本包完整训练状态。

## 验证和发布

```bash
python -m pytest -q
python scripts/verify_release.py
python -m build
```

这里的小模型测试不下载权重。Krea2 原生测试需要将 Musubi 的 `src` 加入 `PYTHONPATH`；
否则该项明确显示 skip。完整真实模型仍需先在目标 GPU 上执行 3 步预检和小规模成图。
当前发布验证见 [VALIDATION.md](docs/VALIDATION.md)。

上传 GitHub 时使用**本目录**作为仓库根目录；保留 `.gitignore`、许可证和来源说明。
不需要上传上一级 `research/`、已有权重、缓存或运行目录。
项目仓库：[222222222l/Kaloscope-adapter-trainer](https://github.com/222222222l/Kaloscope-adapter-trainer)。

## 许可

保留原仓库的 [DINOv3 License](LICENSE.md)，DINOv3 来源文件保留版权头。
随包 MOSS 原生实现保留 [Apache-2.0](licenses/MOSS-Apache-2.0.txt)。
权重、外部 Musubi/CSD 代码及数据按各自上游许可使用；没有将第三方材料统一改标为 MIT。
