# 工作记录

## 2026-10-01

### 确定 Part 3 的三个方向

- 范围：Part 3 做两到三个方向，不按作业的最低要求只做一条。
- 方向与顺序：背景抑制，然后多尺度特征，然后提示与视觉距离融合。
- 先不做：合成缺陷、记忆库压缩、选择性预测。
- 记录：`docs/adr/0001-part3-exploration-directions.md`
- 仓库：在 `Project_2` 单独建 git。作业 PDF 留在本地，不进仓库。

### 确定跨数据集的位置

- 主跨数据集：VisA。
- MVTec AD 2：只用公开测试集 TESTpub，作为背景抑制和多尺度之后的压力测试。不提交私有榜。
- 记录：`docs/adr/0002-visa-main-cross-dataset.md`

### 确定 MVTec AD 的类别范围

- 老师的 PDF 只要求至少 5 个物体类和 5 个纹理类，没有点名。
- 决定：15 类全做，包括 5 个纹理类和 10 个物体类，以便和 AnomalyDINO 的全类均值对照。
- 不单独立 ADR：之后若要改成子集，按类保存的指标可以直接重算均值。
- 算力核对：本机 8 张 RTX 4090（各 24GB）和约 504GB 可用磁盘。15 类全做放得下，主实验按单卡规划。

### 确定自采检验集的物体

- 物体：鼠标。照片之后由作者拍摄提供。
- 受控拍摄：机位、姿态、中性光照、干净背景固定。
- 偏移拍摄：同一姿态下只改光照，或只改背景。
- 三种人为缺陷：细划痕、可洗色笔印、贴胶带或拿掉脚垫。
- 术语：`CONTEXT.md`。记录：`docs/adr/0003-wild-set-mouse.md`

### 确定自采检验集的数量和掩码

- 受控无缺陷 30 张：20 张参考池，10 张阈值验证图。
- 偏移无缺陷 20 张：只改光照 10 张，只改背景 10 张。只做测试，不进参考池。
- 受控缺陷 30 张：细划痕、可洗色笔印、贴胶带或拿掉脚垫，每种 10 张。
- 每张缺陷图配同尺寸二值掩码，白色为缺陷，黑色为其余区域。正常图无掩码。
- 术语：`CONTEXT.md`。记录：`docs/adr/0004-wild-set-capture-quota.md`

### 确定工作点

- 图像异常分数取热力图最大像素值。
- 工作点取该分数在阈值验证图上的均值加 3 倍标准差。
- 每个方法、每个类别分别计算，不共用一个数值。
- 图像判定和二值热力图共用这个分界。测试集上的误报率照实报告。
- 术语：`CONTEXT.md`。记录：`docs/adr/0005-operating-point-mean-plus-3std.md`

### 开始实现

- 截止日期不作为排期约束，按已定方案立即开工，能做完的部分先做完。
- 自采鼠标照片尚未提供，先做环境、官方代码、MVTec AD、VisA 和 Part 1。
- Python 环境使用 `/home/zhangyuming/CV_project/env_cv`。该环境原先没有 pip，已用 ensurepip 装上 pip，并升级到 26.2.1。
- 上游检出：AnomalyDINO `b9d1c264`，AnomalyCLIP `3911738c`。记录在 `third_party/README.md`，目录本身不进 git。
- 开始下载 MVTec AD（约 5.3GB）和 VisA（约 1.9GB）到 `data/archives/`。
- MVTec 官方 mydrive 只有几十 KB/s，已改从 Hugging Face 上同一大小的归档下载。
- MVTec 正常图划分写成 `docs/adr/0006-shared-normal-split.md`：每类从 `train/good` 抽 10 张做阈值验证，种子 0。
- Part 1 代码：配准差分在 `src/part1/difference.py`，自编码器在 `src/part1/autoencoder.py`。工作分辨率先用 256，和后面的 DINOv2 分辨率分开记录。

### 对照论文复核已定方案

- 对照了 AnomalyDINO（WACV 2025）、其官方代码、AnomalyCLIP 的 `test.sh`，以及 MVTec AD 2 的阈值段落。
- 15 类 MVTec、VisA 做主跨数据集、AD 2 只做公开集压力测试，与 AnomalyDINO 的实验设置一致。
- 已按论文改正：检测分数用最高 1% ；工作点用验证图像素的均值加 3 倍标准差，图像判定再用最大像素。MVTec 阈值验证图改到文件名排序的尾部，官方少样本切片留在前缀。背景抑制只做 PCA 掩码失败的情况。
- 记录：`CONTEXT.md`，`docs/adr/0001-part3-exploration-directions.md`，`docs/adr/0005-operating-point-mean-plus-3std.md`，`docs/adr/0006-shared-normal-split.md`。

### 下载并整理 VisA

- `VisA_20220922.tar` 已下完，大小 1929840640 字节，与官方对象一致。
- 解压到 `data/visa_raw`。`src/inspection/prepare_visa.py` 按 `split_csv/1cls.csv` 链到 `data/VisA_pytorch/1cls`，异常类目录用 `test/bad`，掩码用 `<stem>.png`，与 AnomalyDINO 的 VisA 读取方式一致。

### 设定完成目标

- 目标是做完整个 Project 2，不只做 Part 1。
- MVTec 归档已到齐，大小 5264982680 字节。正在解压。
- 训练和长实验至少隔 15 分钟才查看，完成与否看进程自己打印的结束标记，例如 `PART1_DONE`。
- 解压后的类别直接落在 `data/` 下，已移到 `data/mvtec_anomaly_detection/`。15 类划分已写入 `outputs/splits/mvtec/`：每类验证图 10 张，官方前缀 48 张，无重叠。toothbrush 的参考池是 50 张。

### 装好环境并启动 Part 1

- PyTorch 2.11.0+cu128 安装成功，CUDA 可用，看到 8 张 RTX 4090。
- 协议测试和差分冒烟测试 5 项通过。
- 接着用 `scripts/run_part1.sh` 跑全部 15 类。完成标记是日志里的 `PART1_DONE`。
- 在 GPU 1 上启动 AnomalyDINO 官方复现：dinov2_vits14、448、agnostic、1/2/4/8-shot、3 个种子、`--faiss_on_cpu`、同时算分割。完成标记是 `ANOMALYDINO_MVTEC_DONE`。Part 1 的自编码器仍用 GPU 0。

### 评测和单张图 demo

- `src/inspection/metrics.py` 提供图像 AUROC、AP、工作点上的误报率，以及积分上限 0.3 的 AUPRO。合成例子测试已通过。
- `src/demo/inspect.py` 对一张图跑 Part 1 差分，写出叠加热力图。若该类的 `scores.json` 已有工作点，同时给出 OK/NG。
- AnomalyCLIP 仓库没有附带 `epoch_15.pth`。提示权重要在空闲 GPU 上按官方设定训练，不占用正在跑的 Part 1 和 AnomalyDINO。
- 已为 AnomalyCLIP 生成 `data/mvtec_anomaly_detection/meta.json` 和 `data/visa_raw/meta.json`。训练脚本用 GPU 2，先在 VisA 上训提示，再在 MVTec 上训提示。完成标记是 `ANOMALYCLIP_TRAIN_DONE`。
