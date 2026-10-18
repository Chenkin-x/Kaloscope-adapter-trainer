# 本次整理的验证记录

日期：2026-10-17。最终CPU回归：**11项通过**（包括Krea2与Moss原生测试）。
当前环境无可用CUDA；未启动新的远端训练或生成。

| 项目 | 本次结果 | 证明范围 |
| --- | --- | --- |
| Krea2真实原生小模型 | 通过 | 零初始化/关闭适配器逐值一致，mask隔离，checkpoint梯度一致 |
| Moss真实原生小模型 | 通过 | 同上，含batch2及不等长文本 |
| 历史权重语义检查 | 通过 | 相同形状但不同注入层/bank映射会拒绝加载 |
| 公共CPU训练入口 | 通过 | 小Moss、真实FM反向、AdamW、保存/验证 |
| step2暂停→step4恢复 | 通过 | 与不间断训练的全部adapter张量逐值一致 |
| global batch5 / 两进程 | 通过 | 不均匀3/2样本分配与单进程一步更新一致（容差见测试） |
| reference dropout=1控制 | 通过 | 全部关闭风格时允许零梯度，不错误终止 |
| artist-v1实际导出权重 | 通过 | CPU512输入，重打包编码器与原加载器输出256D逐值一致、L2范数1 |
| ComfyUI格式导出 | 通过 | safetensors格式元数据与全部张量round-trip一致，未执行ComfyUI成图 |
| 源码构建与独立安装 | 通过 | wheel构建、脱离原目录导入和CLI帮助 |
| 发布审计 | 通过 | JSON、源码解析、本地文档链接、证据SHA、无原工程import/机器路径 |

本地双进程测试最初被沙箱的回环socket限制阻止；在允许本机通信的测试环境中重跑通过。
这不是GPU通信性能或NCCL多机测试。依赖产生的torch JIT/SWIG弃用警告不影响本次结果。

主要命令（Python指向装有相应依赖的独立环境）：

```bash
PYTHONPATH=src:vendor/musubi-tuner/src python -m pytest -q tests
ruff check src/kaloscope_style tests scripts --exclude _vendor
python scripts/verify_release.py
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/kaloscope-wheel .
```

实际CPU环境：Python3.12、PyTorch2.13.0+cu130、Transformers5.13.0、Diffusers0.39.0。
GitHub CI不下载Musubi/模型权重，Krea2原生测试会显示skip；本次本地验证额外提供了已记录SHA的Musubi源码，
因此Krea2小模型测试实际运行。CI尚未在GitHub远端执行，本次只交付工作流文件。

未验证项：新包的真实大模型CUDA前后向、缓存编码与成图全链、量化/低显存offload、Windows、
跨机器/不同软件版本逐位恢复、新数据的风格质量和所有声明最低依赖版本。
首次在目标GPU使用时先跑3步预检；历史的H100实验数据不当作新包本次GPU验收。

## 2026-10-18 上传前补充核验

修正发布文件筛选与Git忽略规则：权重目录`/models/`仅按根目录排除，
保留`src/kaloscope_dinov3/models/`的三份实现文件；同时排除Git元数据及本地环境。
更新后的ZIP共88份源文件，逐文件SHA验证通过；在临时目录解压后，
DINOv3 ViT、ConvNeXt与ArtistEncoder均可直接导入。此前wheel包含模型源码，
此次修复针对源码ZIP及Git上传规则，不改变模型训练算法。
