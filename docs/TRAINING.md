# Krea2 训练指南

## 1. 环境与依赖

建议单独建立 Python 3.10–3.12 环境。先按 GPU/CUDA 安装匹配的 PyTorch，随后：

```bash
python -m pip install -e '.[krea2,dev]'
git clone https://github.com/kohya-ss/musubi-tuner.git vendor/musubi-tuner
export PYTHONPATH="$PWD/vendor/musubi-tuner/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_backends.py -k krea2
```

本包直接调用 Musubi 的 Krea2 原生模块，不调用它的 LoRA 训练入口。
`backend.source` 可以指向已有 Musubi 仓库或其 `src`；运行时自动加入模块路径。
不要直接安装旧 Musubi 全套锁定依赖覆盖 Moss 环境：旧副本约束的 diffusers/transformers 与 Moss 不同。
Krea2 和 Moss extras 的 Transformers 主版本要求不同，应分别建立环境。

本地原生源码副本的 Git 子模块元数据缺失，无法诚实提供已确认的 commit pin。
已提供 [Krea2 源码指纹](../evidence/krea2_source_hashes.json)，而不是猜测版本号。
上面 clone 会取上游当前版本，**应运行原生一致性测试并记录 `git rev-parse HEAD`**；
本包 GPU 预检也会再次核验关闭风格时的原生预测。上游接口变化时参考
[Musubi Krea2 文档](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)。
该官方文档确认 Krea2 使用 Qwen3-VL 文本编码器和 Qwen-Image VAE；本包选择历史实验所用 Raw 版。

本次整理环境的 CPU 验证使用 PyTorch 2.13.0+cu130、Transformers 5.13.0、Diffusers 0.39.0；
这不是最低版本覆盖测试，也不代表旧 Musubi 的完整依赖解析已验证。

## 2. 模型资产

```text
models/
  krea2/
    raw.safetensors
    vae.safetensors
    qwen3vl_4b_bf16.safetensors
    tokenizer/                         # 本地 Qwen3-VL tokenizer 快照
  z3_artist_v1/
    config.json
    model.safetensors
```

Kaloscope 应为训练匹配的 artist-v1 编码器，输出归一化的 **256 维 projector**，不能换成
1536 维 pooled backbone、patch token、分类 logits 或同维但语义不同的编码器。
支持两种格式：

1. 导出的目录：`config.json` 的 `model` 为 `{"name":"dinov3_vitb16","pooling":"cls_mean","projection_dim":256}`，
   `checkpoint` 为 `model.safetensors`。权重键含 `backbone.*`、`projector.0.*`、`projector.2.*`。
2. 原训练 `.pt`：含 `model_config` 和 `model`。只用 `torch.load(weights_only=True)` 读取；
   含自定义 Python 对象的旧快照应先在可信原环境导出为上述格式，不自动退回任意 pickle 加载。

模型和 tokenizer 由用户从其获准来源准备；不自动下载巨型权重。

## 3. 数据格式与隔离

[manifest.json](../examples/manifest.json) 是结构示例，其中图片路径为占位，需替换为自己数据。
所有配置路径相对于**配置文件目录**；图片路径相对于**manifest 所在目录**；CLI 的路径相对于当前目录。

- `assets`：每幅图有唯一 `id`、`path`、`artist`、全局唯一作品族 `family`、`split`。
- `train` / `dev`：每个目标有 `id`、`target`、`references` 列表、无画师名的内容 `prompt`。
- 同一配对：目标与参考同画师、不同作品族；默认随机选一张参考，100% 单参考训练。
- 同一作品及近重复、裁切、压缩、改色应归为同一作品族，族不能跨 train/dev/test。
- 优先跨角色、相近时期配对。历史 demo 使用日期差≤365天；本包不从文件名推断角色和日期，
  必须在构建 `references` 时筛选。本包会检查同画师、跨作品族和 split，不声称自动完成语义去重。
- dev 至少两位画师才能计算错误参考对照。若要测“适配器未见画师”，把这些画师的所有训练对排除，
  与仅“同画师新作品”分别报告。本包不会自动替你决定未见画师名单。
- test 可记录在 assets 中，但本包缓存/训练只读取 train/dev。真实的封存测试应在候选冻结后单独制作评估面板。

缓存前按 EXIF 转正、透明白底、中心裁方、PIL bicubic 缩放。训练默认拒绝放大不足尺寸的图片。
Kaloscope 参考始终512；训练目标由 `resolution` 控制。此版本入口采用固定方形分辨率；
历史的多阶段宽高比桶调度保留为研究证据，未冒充已经移植成通用训练入口。

## 4. 缓存

```bash
python -m kaloscope_style.prepare --config configs/krea2_z3.json --kind all
```

可依次使用 `--kind features` / `latents` / `text`，节省同时驻留显存。单进程串行准备，
不要向同一缓存目录并发写多个任务。文件存在则复用；每阶段完成后更新 `metadata.json`。
它记录 manifest、图片、模型和源码的身份信息。更换模型/编码器/文本长度/分辨率/数据应使用新缓存目录。
重复图片的字节哈希不能代替近重复检测。

Krea2 文本缓存保留多个选定隐藏层 `[B,L,K,D]`，用原生 `gather_valid_text` 去掉模板中间的无效 token，
再补右侧 padding；不能只把有效 token 数量当成前缀长度截断原始 Qwen 输出。

缓存换取训练时较低的编码显存开销，但全量数百万张图片的 latent/text 存储会很大。
本包面向清晰可改造的研究训练入口；原在线编码、Arrow 流水线和远端控制器没有原样塞入发布包。

## 5. 训练、权重分叉与恢复

```bash
python -m kaloscope_style.train --config configs/krea2_z3.json --stop-after 3
python -m kaloscope_style.train --config configs/krea2_z3.json --resume
```

`--stop-after` 是**绝对步数**，不缩短学习率曲线。例如 horizon=2048 时停止在3，再恢复，
学习率仍按2048步计算。首次零输出初始化；指定 `initial_checkpoint` 则严格加载已有适配器，
**重置**优化器；不把这个选项叫作“完整恢复”。

`latest.pt` 保存适配器、AdamW、各 rank RNG、步数、配置/代码/缓存身份。
`adapter_stepXXXXXXXX.pt` 是可移植权重包。`--resume` 不接受配置、代码、缓存或 world size 改变；
想改配方请新建 output，以 `initial_checkpoint` 分叉。
SIGINT/SIGTERM 会在完整 optimizer step 后保存。周期性文件不自动删除，由使用者控制磁盘保留策略。

DDP 使用 `torchrun`，global batch 按**原图**计数。每个 rank 按全局样本下标分配数据；
若 global batch 不能整除卡数（例如128/6），按实际样本数加权 loss 并补偿 DDP 的 world-size 平均。
不能把每卡 batch 平均后直接平均。micro batch 控制单次前后向，剩余样本自动累积。

训练目标：`x=(1−σ)z+σε`、`v*=ε−z`，FP32 均值平方误差；BF16 主干/自注意力前向，
FP32 adapter 权重和 AdamW。Krea2 的 logit-normal σ 做分辨率相关 shift。
训练只冻结参数；不能把第一处注入之后的主干前向放进 `no_grad`，否则早层 adapter 学不到。

## 6. 推理与控制组

[README](../README.md) 的命令是文生图，从纯噪声开始。
图生图增加 `--content data/content.png --noise 0.85`。
使用 `--native` 生成关闭风格的配对对照，固定同 prompt/seed/text CFG；
错误参考对照只替换 `--reference`，保持其他设置一致。

Krea2 默认28步 Euler、文本CFG2、风格guidance5.5、scale1。
`scale` 是每层 adapter 残差倍率；guidance 是模型速度预测差的系数，两者不是同一个旋钮。
本包采用FP32 Euler轨迹更新，其与历史某些BF16轨迹脚本可能存在舍入差异，
不承诺与原研究PNG逐位相同。运行的sigma列表、权重/参考/输出SHA写到图片同名JSON。

## 7. 导出到已有 ComfyUI Z3 插件

本包`.pt`的配置是分层结构，旧插件不能直接解释。标准Krea2 Z3架构可导出为插件已有的safetensors格式：

```bash
python -m kaloscope_style.export \
  runs/krea2-z3/adapter_step00002048.pt \
  runs/krea2_z3_step2048.safetensors
```

仅支持256D/width256/8 tokens/4 banks/全28层；Moss或其他结构会明确拒绝。
原插件使用的Kaloscope编码器和Krea2底模仍必须与训练一致。导出格式兼容不代表新权重已完成ComfyUI成图质量验证。
