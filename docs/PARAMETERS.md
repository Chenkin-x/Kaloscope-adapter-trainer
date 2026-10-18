# 主要可调参数与试验顺序

默认不是穷举最优解。优先固定数据、参考、prompt、噪声、训练seed和样本曝光量，
一次只改一类变量。推理调参通常比重新训练便宜，但开发集反复挑选需要新的确认集。

## 参数地图

| 参数 / 配置位置 | 本包 Krea2 示例 | 可以测试 | 影响及比较要求 |
| --- | --- | --- | --- |
| `training.peak_lr` | 1e-4 | 1e-5 / 3e-5 / 1e-4 | Z3对Z2只改峰值LR，已有直接对照；别同时改初始化 |
| `training.min_lr` | 1e-5 | peak的1%/10% | 与horizon、warmup共同规定优化量 |
| `training.warmup_steps` | 20 | 0 / 20 / 100 | 本包从零示例；历史Z3续训没有新增warmup |
| `training.steps` | 2048 | 256预检→1024→2048 | 用固定验证和成图判断，不靠训练loss“看起来下降” |
| `training.text_dropout` | 0.5 | 0.1 / 0.3 / 0.5 | 内容文本和风格条件竞争；Z1→Z2是0.1→0.5 |
| `training.reference_dropout` | 0.1 | 0.05 / 0.1 / 0.2 | 训练无style分支；与推理guidance联动但需分开消融 |
| 单/多参考 | 100%单张 | 单张vs多图归一化均值 | 当前入口只实现单张；多图需改data，不是不存在的CLI开关 |
| `adapter.style_tokens` | 8 | 4 / 8 / 16 / 32 | 容量/带宽；必须重训，不兼容旧检查点 |
| `adapter.adapter_width` | 256 | 128 / 256 / 512 | 本实现head_dim固定64，因此heads=width/64 |
| `adapter.adapter_banks` | 4 | 4 / 7 / 28 | 共享程度与参数量同时改变，不能只归因于层数 |
| `adapter.injection_indices` | 省略=全部28层 | 末4层 / [3,11,19,27] / 全层 | 显式序列；配合`injection_slots`映射，长度必须一致 |
| `training.global_batch` | 16 | 16 / 32 / 128 | 记下样本曝光量和优化步数；放大batch并非无条件提质 |
| `training.micro_batch` | 1 | 1 / 2 / 4… | 改峰值内存和累积次数；尽量保持global batch不变 |
| `training.sampling` | artist_balanced | artist_balanced / shuffled_epochs | 画师均衡 vs 每目标一轮；改变画师长尾曝光 |
| `training.time_shift` | null=按Krea2分辨率计算 | 固定shift / 动态shift | 参数是正数shift，不是log-shift μ；Moss demo=1 |
| `resolution` | 512方图 | 512 / 768 / 1024 | 重新缓存latent；当前公共入口不实现宽高比分桶 |
| `text_max_length` | 128 | 128 / 256 | 重新缓存文本；截断内容提示会改变监督 |
| `weight_decay` / `clip_grad` | 0.01 / 1.0 | decay0/0.01，clip0.5/1 | 同时记录未裁剪norm和触发次数 |
| 数据覆盖 | 自备数据 | 32 / 128 / 更多画师 | 相同曝光量对照，独立未见画师确认 |
| 配对规则 | 同画师跨作品族 | 跨角色、时期窗口、去近重复 | 先改manifest；别让参考变成目标的裁切版本 |

改变tokens/width/banks等结构必须新建run；严格加载会拒绝形状不匹配。
本包没有训练可调scale随机化、LoRA、额外style loss或自动超参搜索；它们是可扩展研究方向，
不是通过配置写一个键就会生效的已有功能。

## 推理旋钮

| CLI | 默认 | 历史测试点 / 建议对照 | 解释 |
| --- | --- | --- | --- |
| `--text-cfg` | Krea2=2 | 2 / 3.5 / 5.5 | 控制内容文本相对空文本的速度差 |
| `--style-guidance` | Krea2=5.5 | 3 / 5.5 / 7.5 / 8 | 控制有/无参考之间的速度差 |
| `--scale` | 1 | 0.85 / 1 / 1.25 / 1.5 | adapter内部幅度；高值可能破坏结构 |
| `--noise` | 图生图0.85 | 0.65 / 0.80 / 0.85 | 越低越保留源图；文生图从1开始，不使用此值 |
| `--steps` | Krea2=28；Moss=50 | 固定模型后做步数扫描 | 与scheduler一起记录，不能只比较步数数字 |
| `--reference` | 单图 | 正确 / 错误 / 不同时期 | 同prompt/seed、同引导强度对照 |
| `--native` | false | 同一参数的无风格对照 | scale0/无参考不会修复其他模型或日程错误 |

三分支公式：`v = U + text_cfg*(T-U) + style_guidance*(S-T)`，
U为空文本无style，T为内容文本无style，S为内容文本有style。
两个guidance相等时退化为联合CFG。它与“两条文本分支都带style”的普通CFG不同。

## 推荐实验顺序

1. 先通过关闭风格一致性、单步梯度、保存恢复；小样本拟合确认条件可学习。
2. 固定权重扫描推理CFG/scale，不将不同面板的绝对余弦直接相减。
3. 固定AI0或从零初始化做LR、text dropout、单/多参考的独立消融。
4. 再测试注入位置/共享组/容量和数据覆盖；报告参数数、曝光次数及时间。
5. 冻结候选后用新画师、新内容、新seed和盲评确认；多seed重训才能评估训练随机性。
6. 全量和多分辨率升级后重新验证。旧512小面板的结论不能自动延伸到新模型/分辨率。

历史数字见 [Z3实验依据](Z3_EVIDENCE.md)。特别是scale4曾提高同源风格分却破坏图像，
说明“指标更高”不能代替质量判断。
