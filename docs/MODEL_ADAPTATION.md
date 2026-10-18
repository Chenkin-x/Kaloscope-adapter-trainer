# 快速适配其他图像模型：MOSS 示例

可以复用的是 **Kaloscope 特征、style-token 映射结构、配对数据、flow loss 和评估协议**。
必须适配的是 **latent、文本条件、时间参数、主干 token 布局、注入位置、采样与解码**。
“改一个模型路径”不足以完成跨架构移植。

## 两个已实现后端的映射

| 接口 | Krea2 Raw | MOSS-Image-Base |
| --- | --- | --- |
| 主干隐藏宽度 / 层数 | 6144 / 28 | 3072 / 42 |
| style 输入 | artist-v1 256D、FP32 L2 归一化 | 同一编码器可复用 |
| adapter | width256、8 tokens、4 banks | 保持结构，Q/out 按3072重建 |
| 可训练参数 | 13,308,672 | 7,017,216 |
| bank 默认分配 | `floor(i*4/28)`，7/7/7/7层 | `floor(i*4/42)`，11/10/11/10层 |
| 图像 latent，512² | `[B,16,64,64]` | `[B,128,32,32]` |
| VAE | Qwen/Wan f8；DiT内2×2 patch | FLUX2 VAE posterior mode → 2×2 pack → BN标准化 |
| 文本 | Qwen3-VL多层 hidden `[B,L,K,D]` | Qwen3.5最后 hidden `[B,L,D]` |
| 时间传入主干 | σ，模型内进行时间嵌入缩放 | `1000*σ` |
| demo 训练σ | logit-normal，512 μ=0.58125的指数shift | logit-normal，shift=1 |
| 采样 | 28步Euler、随分辨率shift | 原生50步Euler、shift=5、训练网格吸附 |
| 默认推理引导 | text2 / style5.5 | text3 / style6（用户认可的探索设置） |
| 无条件分支 | 原生空文本无style | 空文本无style，保留原生skip layer22 |

结构依据来自随包的两个后端与本地原生代码快照；Moss公开模型入口见
[官方仓库](https://github.com/OpenMOSS/MOSS-Image)和[模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)。
随包的 MOSS native pipeline 位于 `_vendor/moss.py`，用于固定本轮移植依据。

## 直接运行 Moss

独立环境安装 CUDA 对应 PyTorch，然后：

```bash
python -m pip install -e '.[moss,dev]'
python -m pytest -q tests/test_backends.py -k moss
python -m kaloscope_style.prepare --config configs/moss_z3.json --kind audit
python -m kaloscope_style.prepare --config configs/moss_z3.json --kind all
python -m kaloscope_style.train --config configs/moss_z3.json --stop-after 3
python -m kaloscope_style.train --config configs/moss_z3.json --resume
```

配置的 `backend.model` 下需有 `transformer/`、`vae/`、`text_encoder/`、`tokenizer/`。
`initial_checkpoint=null`：Moss 从零输出初始化，**不能载入 Krea2 Z3 权重**。
形状不同会严格报错。缓存单独位于 `cache/moss`；不接受 Krea2 latent/text。
256D DINO 特征只有在编码器、crop、归一化都一致时才能直接迁移。

## 移植第三种模型的最小步骤

1. 先加载并冻结原生主干，用官方 forward 得到固定 `z/time/text` 的基准预测。
2. 在 [backends/api.py](../src/kaloscope_style/backends/api.py) 增加模型类型、加载器、VAE与文本codec、
   σ转换、采样日程；在 `read_config` 的枚举中注册。这里有明确分支，未宣称任意模型即插即用。
3. 复制一个 `Runner`，只适配模型自己的前向遍历。保留原始时间条件、AdaLN/scale/shift/gate、
   RoPE、attention mask、头部输出。**完整block后，仅对image token加旁路残差**。
4. 从模型config读取hidden width，按实际层数映射bank。文本在前/图像在后、双流DiT、UNet、
   图像参考ICL等布局必须显式划分图像片段，不能照搬 `x[:, :n]`。UNet还需定义空间展平和恢复。
5. 若采用 epsilon/v-prediction 或其他目标，连同噪声日程、loss target、输出到采样器的转换一起修改。
   当前公共训练器仅支持上述两个 rectified-flow 速度目标，不能强套到所有扩散模型。
6. 通过原生 parity、小批拟合、错误参考控制和成图评估后，才扩大数据/步数。

最常见移植错误：

- Moss重复乘时间1000，或Krea2提前乘1000；训练loss可能仍有数值，但时间语义错了。
- 忘记 FLUX2 pack/BN 的逆变换，或误用 Krea2 VAE 缩放常数。
- 把padding当有效文本；Qwen模板里的mask不一定是连续前缀。
- 用 `no_grad()` 包住整个冻结主干，切断adapter的梯度。
- checkpoint闭包晚绑定循环变量。Moss逐样本前向必须绑定当前样本的rope/modulation/active，
  否则反向重算可能用到下一张样本；随包测试覆盖batch>1和不等长文本。
- 隐藏宽度相同就直接转移权重。宽度相同不意味着模型内部坐标系和特征语义兼容。

## 上线训练前的最小核验

| 核验 | 预期 |
| --- | --- |
| 零初始化 / active=0 / scale=0 | 与原生输出一致 |
| adapter非零后 | 图像token有变化；文本token不被旁路直接改写 |
| 一步反向 | out投影非零有限梯度；主干grad全部None |
| checkpoint on/off | 同一batch的梯度一致 |
| padding扰动 | 只修改mask外文本，结果不变 |
| VAE round-trip | 编码/解码后图像正常，打包反解与BN一致 |
| 保存与重载 | 同输入预测一致；完整恢复能继续优化器 |
| 单卡/多卡 | global batch一致，非整除分配仍梯度等价 |
| 正确/错误/关闭参考 | 固定target/noise/time下比较FM，固定prompt/seed下比较成图 |

## Moss 的实际证据与边界

256步demo：32画师、2895目标、batch16，固定64例FM由0.40683651降到0.40651649；
参考图和名称原型Hit@1均0/8。说明路径接通，不能据此宣称风格迁移达标。

后续同一诊断协议下256→1024步：正确参考FM 0.42694739→0.42586944，
关闭参考仍为0.42746016；style CFG6下目标DINO余弦0.19017→0.26642，Hit@1仍0/16。
CFG3的未见组余弦从基模0.22678降到0.19678；CFG6的独立CSD对基模差值仅+0.00334。
这些数据不支持“稳定泛化已证明”，也不支持“主干完全不能响应风格”。
[机器可读诊断](../evidence/moss_diagnosis.json)。

之后用户明确接受CFG6的视觉效果并授权全量训练，原工程于2026-10-17启动六卡训练，
启动核验记录14个优化步。必须把这个**后来的主观选择与启动事实**和前面的诊断结论同时保留，
不能继续写“全量未启动”，也不能把启动当成质量验收。
[启动快照](../evidence/moss_launch.json)、[全量起点的独立评估](../evidence/moss_baseline.json)。
这些是本地保存的历史快照；本次整理没有连接远端查看或改变正在进行的训练。
