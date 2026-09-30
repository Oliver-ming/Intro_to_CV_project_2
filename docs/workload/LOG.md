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
- 第一次启动因缺少 `thop` 退出。已装上 `thop` 和 `tabulate`，并重新启动。
- 第二次启动时 CLIP 权重被写到作者机器上的 `/remote-home/...`。已改回 `~/.cache/clip` 后再启动。

### Part 3 的可单测部分

- `src/inspection/evaluate_run.py` 能在 `scores.json` 和热力图写完后计算图像 AUROC、AP、误报率，以及像素 AUROC、AP、AUPRO。
- `src/part3/background.py` 用中心矩形初始化的 GrabCut 得到前景，并把背景上的异常分数清零。这是 PCA 掩码失败时的替代，不是再做一遍第一主成分。
- `src/part3/fusion.py` 先用各自验证像素的均值和标准差把两张热力图标准化，再取平均。
- 这三项的单元测试已通过。长任务的中间进度这轮没有查看。

### 多尺度融合和少样本汇总

- `src/part3/multiscale.py` 把粗分辨率热力图放大到细分辨率网格，再按验证像素做标准化后平均。细图上的尖峰在粗图平坦时仍然保留。
- `src/inspection/aggregate_seeds.py` 读取 AnomalyDINO 每个种子的均值指标，汇总成均值和样本标准差，供 1/2/4/8-shot 的三次重复使用。
- 相关单元测试已通过。

### 自采划分和 VisA 复现

- `src/inspection/wild.py` 按约定把受控正常图分成 20 张参考池和 10 张阈值验证图。光照和背景偏移只进测试集。单元测试已通过。
- 在 GPU 3 上启动 VisA 的 AnomalyDINO，协议与 MVTec 相同。完成标记是 `ANOMALYDINO_VISA_DONE`。

### 多尺度所需的 672 分辨率和 AD 2 数据

- 在 GPU 4 上按同一套 1/2/4/8-shot、3 个种子跑 672 分辨率。完成标记是 `ANOMALYDINO_MVTEC_672_DONE`。和 448 的结果目录按分辨率分开。
- 开始下载 MVTec AD 2 全量归档，约 32.7GB，放到 `data/archives/mvtec_ad_2.tar`。压力测试只用公开测试集，不提交私有榜。
- 官方 mydrive 只有约 8KB/s，预计要四十多天，下载已中止，127KB 的残片已删除。Hugging Face 上的 FiftyOne 转写不是官方 TESTpub 划分，不拿来代替。压力测试数据等有更快的官方来源再下。

### 效率和 AnomalyCLIP 测试入口

- `src/inspection/efficiency.py` 汇总 AnomalyDINO 测量表里的记忆库构建时间和逐图打分时间。单元测试已通过。
- `scripts/run_anomalyclip_test.sh` 在两份 `epoch_15.pth` 都存在后，用 VisA 上学到的提示测 MVTec，再用 MVTec 上学到的提示测 VisA。完成标记是 `ANOMALYCLIP_TEST_DONE`。权重还没写完，所以这轮没有启动测试。
- 记忆库规模按 AnomalyDINO 的 8 个旋转角和 patch 14 估算。1-shot、448 分辨率的上界是 8192 个 patch 向量。demo 叠图上会写检测分数，有工作点时同时写 OK/NG。
- `python -m src.inspection.evaluate_run` 在分数文件写完后输出指标 JSON。`python -m src.part3.apply_multiscale` 把 448 和 672 的同名热力图融合成一张。目录融合的单元测试已通过。
- `python -m src.part3.apply_background` 用 GrabCut 前景把已保存热力图的背景分数清零。边角被清掉、物体中心保留的测试已通过。Part 1、两路 AnomalyDINO 和 AnomalyCLIP 训练进程仍在，这轮没有读取它们的进度。

### 15 分钟核对

- 只查结束标记和进程是否还在。没有 `*_DONE`。Part 1 的差分已完成 bottle，正在做 cable 的测试图，大约每张 7 到 12 秒，没有卡住。AnomalyDINO 的 448、672、VisA 以及 AnomalyCLIP 训练进程都还在。
- 提示分数和视觉检测分数的融合写在 `src/part3/fuse_scores.py`：两边都用正常验证分数做标准化后再平均。
- `python -m src.part3.apply_score_fusion` 读两份分数文件，写出融合后的检测分数和 OK/NG。共享图像上的单元测试已通过。这轮没有再读长任务日志。
- demo 叠图会写成带分数的 PNG，单元测试确认文件非空。`python -m src.inspection.completion` 只根据日志里的 `*_DONE` 判断任务是否结束。
- 错误判定按原因归类：正常图的边角高响应算背景或边界，漏检且缺陷像素很少算细缺陷。`tests/test_failures.py` 已通过。这轮没有读长任务日志。
- bottle 差分的图像级结果：AUROC 0.910，AP 0.970，83 张测试图。工作点下正常测试图的误报率是 1.0，最大像素很容易超过像素均值加 3 倍标准差。像素级 AUROC 0.922，AP 0.453，AUPRO 0.801（积分上限 0.3）。数字写在 `outputs/part1/difference/bottle/metrics.json`。demo 叠图写到 `outputs/demo/bottle_broken_large.png`。
- AUPRO 改为和 MVTec / AnomalyDINO 相同的做法：所有像素按分数排序后一次累加，不再抽 256 个分位点。bottle 的 AUPRO 从近似值 0.8009 变成精确值 0.8010。`python -m src.inspection.collect_tables` 会把已经写好的 Part 1 指标和 AnomalyDINO 的 `metrics_seed` 收成 `outputs/tables/comparison.json`，缺的类别会跳过。当前表里只有 bottle 差分。MVTec AD 2 的公开集仍没有可用的快速镜像，官方归档约 30GB，不重新开始那次过慢的下载。
- 官方 AnomalyDINO 只给 `test/` 打分。工作点要用的正常验证尾图由 `scripts/score_anomalydino_heldout.py` 按同样的参考图切片补打分，完成标记是 `ANOMALYDINO_HELDOUT_DONE`。单张 bottle 验证图已在 GPU 6 上跑通，参考图是 `000.png`。AnomalyCLIP 的 `test.py` 现在会把未四舍五入的指标写到 `metrics.json`，`collect_tables` 会读它。
- 自编码器改成和差分并行：`scripts/run_autoencoder.sh` 用 GPU 0，结束标记 `AUTOENCODER_DONE`。已经写过 `scores.json` 的类别会跳过，文件锁避免和 Part 1 脚本里随后那次训练叠在一起。672 的验证图打分在 GPU 7，结束标记 `ANOMALYDINO_HELDOUT_672_DONE`。这轮只查了结束标记，都还没有。
- 提示分数和视觉距离要对齐到同一条相对路径才能融合。AnomalyDINO 的测量表把测试图记成 `类别/test/缺陷/文件名`。AnomalyCLIP 测试会额外写 `image_scores.json`，每张测试图一个检测分数。验证尾图的分数仍来自补打分。`src/part2/score_files.py` 把两边收成融合脚本能读的分数文件。
- bottle 差分的错误判定有 20 条，全部是正常图被判成 NG，原因都是物体内部的正常起伏，不是边角。缺陷图没有漏检。清单在 `outputs/part1/difference/bottle/failures.json`。VisA 和 MVTec 的 `meta.json` 测试路径能对上磁盘上的图像。
- `python -m src.part2.build_fusion` 把 AnomalyDINO 的测试分数、两边的验证尾图分数和 AnomalyCLIP 的逐图分数合成一份融合结果。方向相反的标准化分数会互相抵消，单元测试已通过。`scripts/run_anomalyclip_heldout.sh` 在 VisA 上学到的 `epoch_15.pth` 出现后，给 MVTec 验证尾图打提示分数。权重还没写出来，所以这轮没有启动。
- GrabCut 只用于 PCA 掩码失败的 cable、transistor 和自采鼠标。改过的热力图用 `python -m src.part3.rescore` 在验证图上重拟合工作点。cable 差分的图像级结果：AUROC 0.607，AP 0.723，150 张测试图，正常图误报率 0.931。GrabCut 后 AUROC 0.636，AP 0.734，误报率 1.0。像素级 AUROC 从 0.815 升到 0.896，AP 从 0.096 升到 0.147，AUPRO 从 0.484 升到 0.614。定位变好了，工作点下正常图仍全部被判成 NG。数字在 `outputs/part3/grabcut/cable/metrics.json`。GrabCut 的 58 条错误判定全是正常图内部起伏，没有漏检。这轮结束标记都还没有。MVTec AD 2 只索引 `test_public`，坏图掩码是 `ground_truth/bad/<stem>_mask.png`，私有划分不读。
- 已保存的热力图可以直接叠到原图上。cable 弯曲导线那张 GrabCut demo 在 `outputs/demo/cable_grabcut.png`，判定 NG，检测分数 0.737。汇总表现在也会读 `outputs/part3/grabcut`。
- 自编码器已经写完 bottle、cable、capsule、carpet、grid 的分数。图像级 AUROC / AP / 正常图误报率：bottle 0.761 / 0.928 / 1.0，cable 0.507 / 0.659 / 1.0，capsule 0.576 / 0.875 / 1.0，carpet 0.476 / 0.765 / 1.0，grid 0.972 / 0.991 / 0.952。除了 grid，排序接近或低于差分；工作点下正常图几乎全部被判成 NG。grid 的缺陷图平均检测分数是 0.104，正常图是 0.069。carpet 两边都是 0.103，所以分不开。demo 在 `outputs/demo/grid_autoencoder.png` 和 `outputs/demo/carpet_autoencoder.png`。
- AnomalyDINO 在 MVTec、448、1-shot、种子 0 上跑完 15 类。图像级 AUROC 0.970，AP 0.985。像素级 AUROC 0.965，AUPRO 0.918。transistor 的 AUPRO 只有 0.641，是这组里最低的，和 PCA 掩码在该类失败一致。这只是三个种子里的第一个，还不能写均值和标准差。1-shot 的记忆库构建平均 0.58 秒/类，逐图打分平均 0.44 秒，记忆库上界 8192 个 patch。1725 张测试图。
- 自编码器 hazelnut 的图像级 AUROC 0.950，AP 0.973，110 张测试图，正常图误报率 1.0。leather 是 0.750 / 0.921 / 1.0，124 张。正在把 GrabCut 用到 AnomalyDINO 1-shot 种子 0 的 cable 和 transistor patch 上，比较抑制前后的检测分数。这轮结束标记都还没有。
- 448 和 672 的 patch 距离在 1-shot 种子 0 上做了融合，只用两边都写完的 8 类：bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut。标准化用的是同一种子的正常验证图。图像级 AUROC / AP：448 为 0.980 / 0.992，672 为 0.981 / 0.993，融合后 0.983 / 0.994。cable 从 0.932 升到 0.941。这还不是 15 类，也还没有像素级。数字在 `outputs/part3/multiscale/1shot_seed0.json`。
- pill 后来也齐了。448 / 672 / 融合的图像级 AUROC 是 0.915 / 0.927 / 0.954。加上 pill 后这 9 类的平均 AUROC 是 0.973 / 0.975 / 0.980。自编码器 metal_nut 的图像级 AUROC 只有 0.356：正常图平均分 0.089，缺陷图 0.081，排序是反的。115 张测试图，误报率 1.0。
- 这 9 类在 672 的 patch 网格上又算了像素指标，掩码缩小到该网格，不是官方全分辨率。平均像素 AUROC / AUPRO：把 448 放大到这个网格是 0.976 / 0.944，原生 672 是 0.968 / 0.918，融合是 0.977 / 0.945。融合的图像级 AUROC 仍是 0.980。数字在 `outputs/part3/multiscale/1shot_seed0_pixels.json`。
- 自编码器 pill 的图像级 AUROC 0.745，AP 0.942，167 张，误报率 1.0。差分 capsule 的图像级 AUROC 0.613，AP 0.868，132 张，误报率 1.0。
- 自编码器又写完 screw 和 tile。screw 图像级 AUROC 0.580，AP 0.815，160 张，误报率 1.0。tile 是 0.897 / 0.962 / 1.0，117 张。toothbrush 图像级 AUROC 0.564。这 12 类在 256 的热力图上算了像素指标，掩码缩小到热力图，不是全分辨率。平均图像级 AUROC 0.678，AP 0.861，像素 AUROC 0.831，像素 AP 0.231，AUPRO 0.667，误报率 0.954。AnomalyCLIP 的 `epoch_15.pth` 还没有。
- AnomalyDINO 1-shot 种子 0 的 cable、transistor 用 GrabCut 清了背景，照片先缩到 256，分数仍是最高 1% patch。像素指标在 patch 网格上，不是官方全分辨率。cable 图像级 AUROC 从 0.932 升到 0.942，像素 AUROC 从 0.906 升到 0.921，AUPRO 从 0.830 降到 0.770。transistor 图像级 AUROC 从 0.937 降到 0.902，像素 AUROC 从 0.809 降到 0.740，AUPRO 从 0.532 降到 0.474。数字在 `outputs/part3/grabcut/anomalydino_1shot_seed0.json`。
