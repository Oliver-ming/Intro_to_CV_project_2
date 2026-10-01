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
- 自编码器 transistor 图像级 AUROC 0.401，AP 0.439，像素 AUROC 0.616，AUPRO 0.420，100 张，误报率 1.0。加上它之后 13 类平均是图像级 AUROC 0.657，AP 0.829，像素 AUROC 0.814，AUPRO 0.648。screw 的 448 / 672 / 融合图像级 AUROC 是 0.889 / 0.931 / 0.923；融合后的 patch 网格 AUPRO 是 0.909，高于单独的 0.894 和 0.877。各长任务的结束标记都还没有。
- 448、1-shot 的种子 1 已经写完 bottle、cable、capsule、carpet、grid、hazelnut。和种子 0 比，这 6 类平均图像级 AUROC 是 0.974 和 0.953，两次的均值 0.963，样本标准差 0.015。capsule 从 0.939 掉到 0.820，标准差 0.084，是这 6 类里最不稳的。差分 capsule 在 256 热力图上的像素 AUROC 0.907，AUPRO 0.814，图像级 AUROC 仍是 0.613。
- 种子 1 的 leather 也齐了，两次图像级 AUROC 都是 1.0。加上 leather 后这 7 类两次平均是 0.977 和 0.959，均值 0.968，样本标准差 0.013。
- 这 7 类在 patch 网格上的像素指标也齐了，不是官方全分辨率。两次平均像素 AUROC 是 0.968 和 0.967，均值 0.967，标准差 0.001。AUPRO 是 0.913 和 0.909，均值 0.911，标准差 0.003。capsule 的图像级分数掉得厉害，AUPRO 只从 0.894 降到 0.886。
- 种子 1 的 metal_nut 图像级 AUROC 也是 1.0，AUPRO 两次是 0.832 和 0.832。加上它，这 8 类两次的图像级 AUROC 均值是 0.972，标准差 0.011；像素 AUROC 均值 0.964，标准差 0.001；AUPRO 均值 0.901，标准差 0.003。仍是 patch 网格，不是全分辨率。
- 种子 1 的 pill 也齐了。图像级 AUROC 两次是 0.915 和 0.944，AUPRO 是 0.877 和 0.871。加上 pill，这 9 类、两个种子的图像级 AUROC 均值是 0.967，标准差 0.007；像素 AUROC 均值 0.962，标准差 0.001；AUPRO 均值 0.898，标准差 0.003。
- 种子 1 的 screw 也齐了。图像级 AUROC 两次是 0.889 和 0.881，AUPRO 是 0.822 和 0.823。加上 screw，这 10 类的图像级 AUROC 均值是 0.959，标准差 0.007；像素 AUROC 均值 0.961，标准差 0.001；AUPRO 均值 0.891，标准差 0.002。仍是 patch 网格。
- 种子 1 的 tile 也齐了。图像级 AUROC 两次是 0.999 和 1.000，AUPRO 是 0.920 和 0.922。加上 tile，这 11 类的图像级 AUROC 均值是 0.963，标准差 0.007；像素 AUROC 均值 0.960，标准差 0.001；AUPRO 均值 0.893，标准差 0.002。
- 种子 1 的 transistor 也齐了。图像级 AUROC 从 0.937 降到 0.895，AUPRO 从 0.532 升到 0.565。加上它，这 13 类两个种子的图像级 AUROC 均值是 0.962，标准差 0.008；像素 AUROC 均值 0.949，标准差 0.001；AUPRO 均值 0.866，标准差 0.000。还差 wood 和 zipper。仍是 patch 网格。
- 种子 1 的 wood 也齐了。图像级 AUROC 两次都是 0.991，AUPRO 是 0.907 和 0.904。加上它，这 14 类两个种子的图像级 AUROC 均值是 0.964，标准差 0.007；像素 AUROC 均值 0.948，标准差 0.001；AUPRO 均值 0.868，两次几乎一样。1-shot 的第二个种子和 672 融合都还差 zipper。
- zipper 的 672 也齐了，15 类 1-shot 种子 0 的多尺度融合完整了。zipper 图像级 AUROC 是 448 的 0.988、672 的 0.984、融合 0.988；patch 网格 AUPRO 是 0.816、0.816、0.849。15 类均值：图像级 AUROC 0.970 / 0.972 / 0.978，像素 AUROC 0.961 / 0.953 / 0.964，AUPRO 0.909 / 0.886 / 0.914。数字在 `outputs/part3/multiscale/1shot_seed0_all15.json`。定位仍是 672 的 patch 网格，比官方全分辨率大约低 0.06。
- 种子 1 的 zipper 也齐了，448、1-shot 的 15 类两个种子可以对照。zipper 图像级 AUROC 是 0.988 和 0.987，AUPRO 是 0.757 和 0.740。15 类均值：图像级 AUROC 0.970 / 0.960，均值 0.965，标准差 0.007；AP 0.985 / 0.979，均值 0.982，标准差 0.004；像素 AUROC 均值 0.943，标准差 0.002；AUPRO 均值 0.860，标准差 0.001。图像级波动主要来自 capsule（0.939 到 0.820）和 transistor（0.937 到 0.895）。还缺种子 2。数字在 `outputs/part2/anomalydino_partial/448_1shot_seed1_all15.json`，定位是 patch 网格。
- tile 的 448 / 672 / 融合图像级 AUROC 是 0.999 / 1.000 / 1.000。加上先前的 10 类，这 11 类平均图像级 AUROC 是 0.967 / 0.973 / 0.977，patch 网格 AUPRO 是 0.939 / 0.915 / 0.941。
- toothbrush 的 448 / 672 / 融合图像级 AUROC 是 0.994 / 0.981 / 0.997。加上它，这 12 类平均图像级 AUROC 是 0.970 / 0.974 / 0.979，patch 网格 AUPRO 是 0.939 / 0.913 / 0.939。还差 transistor、wood、zipper。
- transistor 的 448 / 672 / 融合图像级 AUROC 是 0.937 / 0.924 / 0.945，patch 网格 AUPRO 是 0.627 / 0.598 / 0.645。多尺度融合把这一类的检测和定位都抬高了，和 GrabCut 在该类上变差相反。加上它，这 13 类平均图像级 AUROC 是 0.967 / 0.970 / 0.976，AUPRO 是 0.915 / 0.889 / 0.917。还差 wood、zipper。
- wood 的 448 / 672 / 融合图像级 AUROC 是 0.991 / 0.982 / 0.990，AUPRO 是 0.931 / 0.921 / 0.937。加上它，这 14 类平均图像级 AUROC 是 0.969 / 0.971 / 0.977，AUPRO 是 0.916 / 0.891 / 0.918。种子 1 的 toothbrush 和种子 0 一样，图像级 AUROC 都是 0.994。加上它，12 类两个种子的图像级 AUROC 均值是 0.965，标准差 0.006；AUPRO 均值 0.892，标准差 0.002。1-shot 的 672 还差 zipper。
- GrabCut 在 patch 网格上切掉的真缺陷：cable 平均 8.6%，中位数 0，92 张里只有 5 张丢掉大半缺陷。transistor 平均 42.8%，中位数 47.4%，40 张里有 18 张丢掉大半缺陷。所以 cable 的检测分数略升，transistor 的检测和定位都下降。数字在 `outputs/part3/grabcut/defect_removed.json`。
- 种子 0 的 15 类里，patch 网格上的图像级 AUROC 和官方全分辨率完全一致。像素 AUROC 平均低 0.021，AUPRO 平均低 0.057：patch 网格 0.861，官方 0.918。所以前面用 patch 网格报的定位数字整体偏保守，图像级数字就是官方分数。对照写在 `outputs/part2/anomalydino_partial/448_1shot_seed0_all15_vs_official.json`。
- VisA、448、1-shot、种子 0 的 12 类 patch 图已经齐，官方 `metrics_seed` 还没写，所以按同样的最高 1% patch 距离先算。平均图像级 AUROC 0.864，AP 0.880。patch 网格上的像素 AUROC 0.953，AUPRO 0.851。macaroni2 的图像级 AUROC 只有 0.602，像素 AUROC 仍有 0.935。这只是一个种子。数字在 `outputs/part2/anomalydino_partial/visa_448_1shot_seed0.json`。同一套算法收在 `src/part2/score_visa_maps.py`。AnomalyCLIP 正在训 VisA 提示，已经写到 `epoch_4.pth`，15 轮还没完。
- 自编码器 15 类从 03:29 到 04:24，墙钟时间约 55 分钟。AnomalyDINO 448、1-shot、种子 0 的测试分数和验证尾图分数已收成 15 份融合输入，在 `outputs/part2/fusion_inputs/anomalydino_448_1shot_seed0/`。提示分数还要等 AnomalyCLIP。
- 差分的 bottle 和 cable 也在 256 热力图上重算了像素指标，和全分辨率几乎一样：bottle 像素 AUROC 0.922、AUPRO 0.802，cable 是 0.813、0.485。同一分辨率下自编码器更低，bottle 是 0.785、0.485，cable 是 0.501、0.313。
- 差分 carpet 的图像级 AUROC 只有 0.419，AP 0.698，像素 AUROC 0.629，AUPRO 0.175，117 张，误报率 0.786。同一分辨率下自编码器的图像级 AUROC 是 0.476，像素 AUROC 0.757，AUPRO 0.528。纹理上配准差分比自编码器更差。
- 差分 carpet 的 52 次判错里，30 次漏掉缺陷，14 次是边缘误报，8 次是正常纹理变化。正常图检测分数均值 0.785，缺陷图 0.779，排序反了。色差缺陷 `color/000` 被判 OK，分数 0.749；正常图 `good/001` 被判 NG，分数 0.819。热力图跟着编织纹走，缺陷没有单独亮出来。目录在 `outputs/part1/difference/carpet/failures.json`，示意在 `outputs/demo/carpet_difference_miss.png` 和 `carpet_difference_border.png`。
- 差分 grid 的图像级 AUROC 是 0.429，AP 0.649，78 张，误报率 0.714。正常图检测分数均值 0.515，缺陷图 0.488，排序也反了。像素 AUROC 仍有 0.821，AUPRO 0.475。同一分辨率下自编码器是图像级 AUROC 0.972、像素 AUROC 0.951、AUPRO 0.886。网格纹理上整图分数不可用，定位也明显不如自编码器。
- 差分已经齐了 5 类，都在 256 热力图上和自编码器比。这 5 类差分的均值是图像级 AUROC 0.596、像素 AUROC 0.819、AUPRO 0.550；自编码器是 0.659、0.780、0.599。拆开看：bottle、cable、capsule 上差分更好，图像级 AUROC 0.710 对 0.615，AUPRO 0.700 对 0.527。carpet 和 grid 上自编码器更好，图像级 AUROC 0.724 对 0.424，AUPRO 0.707 对 0.325。还差 10 类。
- 自编码器 15 类全部训练并打分完成，结束标记 `AUTOENCODER_DONE`。wood 图像级 AUROC 0.989，AP 0.997，79 张，误报率 0.421，是目前唯一明显低于 1 的工作点。zipper 图像级 AUROC 0.584，AP 0.881，151 张，误报率 1.0。15 类在 256 热力图上的平均是图像级 AUROC 0.674，AP 0.843，像素 AUROC 0.819，AUPRO 0.646，误报率 0.925。种子 1 的 metal_nut 图像级 AUROC 是 1.0。cable 的 1-shot demo 在 `outputs/demo/cable_anomalydino_1shot.png`，判定 NG，检测分数 0.379。
- AnomalyDINO 1-shot 种子 0 的 cable、transistor 用 GrabCut 清了背景，照片先缩到 256，分数仍是最高 1% patch。像素指标在 patch 网格上，不是官方全分辨率。cable 图像级 AUROC 从 0.932 升到 0.942，像素 AUROC 从 0.906 升到 0.921，AUPRO 从 0.830 降到 0.770。transistor 图像级 AUROC 从 0.937 降到 0.902，像素 AUROC 从 0.809 降到 0.740，AUPRO 从 0.532 降到 0.474。数字在 `outputs/part3/grabcut/anomalydino_1shot_seed0.json`。
- 1-shot 的测量表已经写完四份：MVTec 448 种子 0 和种子 1 各 1725 张，672 种子 0 也是 1725 张，VisA 448 种子 0 是 2162 张。FAISS 在 CPU 上。448 种子 0 前几类大约 0.25 秒/张，记忆库大约 0.27 秒/类；跑到 wood 和 zipper 时升到大约 1.1 到 1.4 秒/张。整表平均是 0.440 秒/张，记忆库按类平均 0.579 秒。种子 1 在同一张 GPU 上已经是 1.715 秒/张、记忆库 4.00 秒/类，因为当时差分、VisA、672 和验证尾图都在占 CPU。672 的 3.208 秒/张和记忆库 9.06 秒/类也是在这个负载下量的，不能当成分辨率本身慢了 7 倍。patch 向量上界是 448 的 8192 对 672 的 18432。VisA 种子 0 是 1.609 秒/张，记忆库按类 3.19 秒。这几路长任务都还在跑。
- pill 的印字缺陷 `faulty_imprint/018` 做成了 1-shot 种子 0 的三张热力图。448 和 672 的原始最高 1% 均值是 0.607 和 0.624，热点都落在药片左侧的印字上。融合后的分数是验证图标准化以后的 z 分数，最高 1% 均值 15.20，平滑后最大像素 16.22，验证工作点 2.92，判定 NG。26 张正常药片里有 25 张也超过这个工作点，所以排序 AUROC 0.954 并没有变成可用的 OK/NG。图在 `outputs/demo/pill_448.png`、`pill_672.png`、`pill_fused.png`。
- AnomalyCLIP 官方 `Dataset` 默认读 `meta.json` 的 test。这是对的：VisA 的 train 有 8659 张且没有缺陷，test 有 2162 张，其中 1200 张有缺陷掩码，提示学习的损失需要这些掩码。MVTec 也一样，train 3629 张全是正常图，test 1725 张里有 1258 张缺陷。两边的 test 图像路径都存在。用 VisA 的 test 学提示再测 MVTec，仍然没有看过目标类别。正在跑的训练不用改。
- 官方 AnomalyDINO 的 classification F1 和 segmentation F1 是测试集精确率-召回率曲线上的最大值，用了测试标签选阈值。汇总表可以留着这个数，但不能把它写成我们的 OK/NG。提示和视觉分数融合后没有像素图，工作点用融合验证检测分数的均值加 3 倍总体标准差。这两句写进了 `docs/adr/0005-operating-point-mean-plus-3std.md`。
- 448、1-shot、15 类、两个种子的像素 AP 也齐了。种子 0 是 0.451，种子 1 是 0.442，均值 0.447，标准差 0.006。这是 patch 网格，官方表没有像素 AP。像素 AUROC 有 0.943 时 AP 仍然只有 0.45，因为缺陷像素占比很小。
- 15 类 1-shot 种子 0 的多尺度融合补上图像 AP 和像素 AP。448 / 672 / 融合的图像 AP 是 0.985 / 0.985 / 0.988。像素 AP 是 0.532 / 0.495 / 0.539。像素数都在 672 的 patch 网格上，448 是放大过去再算的，所以高于原生 448 网格上的 0.451。融合的像素 AP 高于单独的两个分辨率。
- VisA、448、1-shot、种子 0 的 12 类像素 AP 均值是 0.257，图像 AP 0.880，像素 AUROC 0.953，AUPRO 0.851。macaroni2 的像素 AP 只有 0.022，像素 AUROC 仍是 0.935。和 MVTec 同一分辨率的像素 AP 0.451 比，跨数据集的定位精度掉得比 AUROC 更多。仍是一个种子、patch 网格。
- 自编码器 15 类在 256 热力图上的像素 AP 均值是 0.224。最低是 cable 0.038、screw 0.047、transistor 0.100，最高是 hazelnut 0.607、grid 0.421、wood 0.353。像素 AUROC 均值 0.819 时，AP 仍然低，因为缺陷像素少。
- `collect_tables` 现在会读 15 类多尺度融合。`outputs/tables/comparison.json` 里差分 5 类、自编码器 15 类、GrabCut 1 类、AnomalyDINO 官方表 2 份、融合 15 类。
- VisA、448、1-shot、种子 0 的官方全分辨率结果写出来了。图像级 AUROC 0.864，AP 0.880，像素 AUROC 0.975，AUPRO 0.907。MVTec 同设置的图像级 AUROC 是 0.970，像素 AUROC 0.965，AUPRO 0.918。跨数据集主要掉在整图检测。macaroni2 图像级 AUROC 只有 0.602，像素 AUROC 却有 0.981，AUPRO 0.897。官方分割 F1 是测试曲线上的最大值，VisA 0.425，MVTec 0.589，这不是我们的工作点。
- macaroni2 的 `bad/055.JPG` 热力图打在右侧两块通心粉的缺口上，检测分数 0.242。这一类正常图分数均值 0.197，范围 0.151 到 0.285；缺陷图均值 0.204，范围 0.164 到 0.272。0.242 已经高于缺陷图均值，但仍落在正常图的范围里。图在 `outputs/demo/visa_macaroni2_055.png`。
- 差分如果发现该类已经有 `scores.json` 就跳过，并在新跑完的类别上直接写 `metrics.json`。正在跑的那次进程还是旧代码，不会中途改行为。`tests/test_difference_smoke.py` 确认已完成的类别不会重算。
- VisA 种子 0 的 patch 网格和官方全分辨率对过了。图像级 AUROC 完全一致。像素 AUROC 平均低 0.021，AUPRO 平均低 0.056：patch 网格 0.953 / 0.851，官方 0.975 / 0.907。和 MVTec 上低 0.021、0.057 的差距一样。这几路长任务都还在。
- 提示和视觉分数的批量融合写在 `fuse_ready_categories`。`scripts/run_prompt_fusion.sh` 在 AnomalyCLIP 的 MVTec 测试分数和验证尾图分数都齐之前退出。齐了之后按 15 类写出融合文件，并打印 `PROMPT_FUSION_DONE`。现在测试分数还没有，脚本按预期退出。已经写好的类别不会重算。
- 448、1-shot、两个种子、15 类的图像级 AUROC，按类算的标准差中位数是 0.002。明显高于这个的只有 capsule，0.939 对 0.820，标准差 0.084；其次是 transistor，0.937 对 0.895，标准差 0.029。pill 大约 0.020。其余类别两个种子几乎重合。还缺种子 2，定位仍是 patch 网格。
- 差分原先每张测试图都重新读取整个参考池。hazelnut 有 391 张参考图，读一遍大约 17 秒，三次配准大约 2 秒。现在每一类只把参考图载入一次。已完成的 5 类有 `scores.json`，重跑会跳过。旧进程已停，退出码 143。
- 448 的验证尾图打分已经写完 1/2/4/8-shot、3 个种子、15 类，共 180 份 `scores.json`。Python 正常结束，外层脚本在打印结束标记前因引号解析失败退出，退出码 2。分数没有缺。结束标记已补进 `outputs/part2/anomalydino_heldout.log`。672 那一路还在跑。
- 差分 hazelnut：图像级 AUROC 0.848，AP 0.919，像素 AUROC 0.961，像素 AP 0.509，AUPRO 0.883，110 张，误报率 1.0。自编码器是 0.950、0.973、0.966、0.607、0.928，误报率也是 1.0。定位接近，整图分数自编码器更高。
- 448、1-shot 的官方全分辨率两个种子：图像级 AUROC 0.970 / 0.960，均值 0.965，标准差 0.007；AP 0.985 / 0.979，均值 0.982，标准差 0.004；像素 AUROC 均值 0.963，标准差 0.002；AUPRO 均值 0.917，标准差 0.001。图像级数字和 patch 网格完全一致。capsule 的图像级 AUROC 仍是 0.939 对 0.820，但官方 AUPRO 是 0.970 对 0.961。还缺种子 2。
- 672、1-shot、种子 0 的官方全分辨率：图像级 AUROC 0.972，AP 0.985，像素 AUROC 0.969，AUPRO 0.927。同一种子 448 的 AUPRO 是 0.918。所以全分辨率上 672 的定位更好。先前 patch 网格上 672 的 AUPRO 更低，那个比较偏保守。这只是一个种子。
- 差分已经齐了 6 类。均值是图像级 AUROC 0.638、AP 0.805、像素 AUROC 0.842、像素 AP 0.214、AUPRO 0.605，误报率 0.905。同一 6 类自编码器是 0.707、0.865、0.811、0.251、0.654，误报率 0.992。
- 448、1-shot、种子 2 的 bottle 热力图齐了。图像级 AUROC 0.990，三个种子是 0.998、0.996、0.990。patch 网格 AUPRO 是 0.907、0.899、0.905。这一类对参考图切片不敏感。种子 2 的其余类别还没齐。
- VisA 种子 1 先齐了 candle、capsules、cashew。图像级 AUROC 相对种子 0 是 candle 0.877 到 0.861，capsules 0.983 到 0.989，cashew 0.785 到 0.864。cashew 换了一张参考图之后整图检测明显回升。还差 9 类，不能算 12 类均值。汇总表已收入现有的官方结果：448 两个种子、672 一个种子、VisA 一个种子。
- 官方全分辨率上，1-shot 种子 0 的 672 比 448 在 15 类里有 11 类 AUPRO 更高。提升最大的是 zipper 0.831 到 0.890，screw 0.909 到 0.936，transistor 0.641 到 0.654。下降的四类都不到 0.006，最大的是 tile。图像级 AUROC 的平均差只有 0.002。672 主要改善定位，不改善整图检测。screw 是例外，图像级 AUROC 从 0.889 升到 0.931。
- 差分 leather：图像级 AUROC 0.474，AP 0.759，像素 AUROC 0.791，像素 AP 0.063，AUPRO 0.562，124 张，误报率 1.0。自编码器是 0.750、0.921、0.858、0.319、0.737，误报率也是 1.0。皮革上自编码器的检测和定位都更好。加上它，已齐的 7 类差分均值是图像级 AUROC 0.614、像素 AUROC 0.835、AUPRO 0.599；自编码器是 0.713、0.818、0.666。
- 672、1-shot、种子 1 的 bottle 热力图齐了。图像级 AUROC 是 0.997。种子 0 的官方数是 1.000，448 种子 1 的官方数是 0.996。两个分辨率、两个种子在 bottle 上都饱和。672 的种子 1 还只有这一类。
- 差分 metal_nut：图像级 AUROC 0.537，像素 AUROC 0.902，AUPRO 0.661，115 张，误报率 1.0。自编码器是 0.356、0.729、0.381，误报率也是 1.0。这一类自编码器的整图排序是反的，差分的定位明显更好。加上它，已齐的 8 类差分均值是图像级 AUROC 0.605、像素 AUROC 0.843、AUPRO 0.607；自编码器是 0.669、0.807、0.630。
- 448、1-shot、cable 的三个种子图像级 AUROC 是 0.932、0.914、0.902。第三个种子最低，但落差小于 capsule。种子 2 的 AUPRO 0.824 仍是 patch 网格。
- 差分 pill：图像级 AUROC 0.606，AP 0.898，像素 AUROC 0.904，像素 AP 0.226，AUPRO 0.824，167 张，误报率 1.0。自编码器是 0.745、0.942、0.910、0.284、0.866，误报率也是 1.0。整图分数自编码器更高，定位两边接近。加上它，已齐的 9 类差分均值是图像级 AUROC 0.605、像素 AUROC 0.850、AUPRO 0.631；自编码器是 0.677、0.818、0.656。
- 已齐的 9 类按物体和纹理分开。6 个物体上差分更好：图像级 AUROC 0.687 对 0.649，像素 AUROC 0.902 对 0.800，AUPRO 0.744 对 0.626。3 个纹理上自编码器更好：图像级 AUROC 0.733 对 0.441，像素 AUROC 0.855 对 0.747，AUPRO 0.717 对 0.404。hazelnut 和 pill 拉近了物体上的整图差距，定位差距还在。tile 和 wood 还没进这组。
- capsule 的 1-shot 三个种子图像级 AUROC 是 0.939、0.820、0.941，均值 0.900，标准差 0.069。种子 1 是低的那次，种子 2 回到了种子 0。官方 AUPRO 前两个种子是 0.970 和 0.961，种子 2 的 patch 网格 AUPRO 是 0.896，和前两次的 patch 网格一样，没有跟着整图分数掉。
- 缓存参考图之后，差分从 05:14 重启到 05:23 写完 hazelnut、leather、metal_nut、pill。hazelnut 用了约 2.5 分钟，后面三类大约每分钟一类。旧进程每类要十几分钟，主要花在重复读参考图。
- 差分 screw：图像级 AUROC 0.788，AP 0.890，像素 AUROC 0.989，像素 AP 0.367，AUPRO 0.957，160 张，误报率 1.0。自编码器是 0.580、0.815、0.941、0.047、0.810，误报率也是 1.0。螺丝上差分的检测和定位都更好。7 个物体的均值因此拉开：差分图像级 AUROC 0.701、像素 AUROC 0.914、AUPRO 0.775，自编码器是 0.639、0.820、0.652。已齐的 10 类总均值仍是自编码器整图更高，0.667 对 0.623，因为三个纹理把差分拖下去了。
- VisA 种子 1 的 chewinggum 图像级 AUROC 是 0.977，种子 0 是 0.983。加上先前三类，这 4 类的图像级 AUROC 均值从 0.907 升到 0.923，主要是 cashew 回升。还差 8 类。672 的验证尾图已有 132 份，180 份还没齐。
- 差分 tile：图像级 AUROC 0.406，AP 0.677，像素 AUROC 0.515，AUPRO 0.141，117 张。正常图检测分数均值 0.838，缺陷图 0.819，排序反了。工作点 0.941 比两边的均值都高，所以误报率只有 0.242，不是因为分得开。自编码器图像级 AUROC 0.897，像素 AUROC 0.748，AUPRO 0.576，误报率 1.0。四个纹理上差分的图像级 AUROC 均值降到 0.432，自编码器是 0.774。
- 差分 toothbrush：图像级 AUROC 0.936，AP 0.975，像素 AUROC 0.977，AUPRO 0.902，42 张，误报率 1.0。自编码器是 0.564、0.786、0.916、0.706，误报率 0.500。牙刷上差分的整图检测高很多。8 个物体的均值变成差分图像级 AUROC 0.731、像素 AUROC 0.922、AUPRO 0.791，自编码器是 0.630、0.832、0.659。样本只有 42 张。
- carpet 的 1-shot 三个种子图像级 AUROC 都是 1.000，标准差是 0。官方 AUPRO 前两个种子是 0.979 和 0.980。种子 2 的 patch 网格 AUPRO 是 0.953，和分辨率差一致，不是种子差。
- 差分 transistor：图像级 AUROC 0.793，AP 0.738，像素 AUROC 0.880，AUPRO 0.731，100 张，误报率 1.0。自编码器是 0.401、0.439、0.616、0.420，误报率也是 1.0。这一类自编码器几乎分不开，差分的检测和定位都更好。9 个物体的均值是差分图像级 AUROC 0.737、像素 AUROC 0.917、AUPRO 0.784，自编码器是 0.604、0.808、0.632。
- 已齐的 13 类总均值接近了。差分是图像级 AUROC 0.643、像素 AUROC 0.847、AUPRO 0.647、误报率 0.898。自编码器是 0.657、0.814、0.648、0.958。整图分数仍略偏自编码器，像素 AUROC 偏差分，AUPRO 几乎一样。还差 wood 和 zipper。
- 672、1-shot、cable 的种子 1 图像级 AUROC 是 0.928，种子 0 的官方数是 0.935。448 的三个种子是 0.932、0.914、0.902。到目前为止 672 在这一类上更高，也更稳。还只有两个种子。
- grid 的 1-shot 三个种子图像级 AUROC 是 0.991、0.993、0.992，均值 0.992，标准差 0.001。种子 2 的 patch 网格 AUPRO 是 0.928，和种子 0 的 patch 网格一样。
- 差分 wood：图像级 AUROC 0.659，AP 0.871，像素 AUROC 0.646，AUPRO 0.351，79 张，误报率 0。正常图均值 0.653，缺陷图 0.670，工作点 0.829 比两边都高，所以没有误报，也没有把缺陷分开。自编码器是图像级 AUROC 0.989，像素 AUROC 0.833，AUPRO 0.672，误报率 0.421。五个纹理齐了：差分图像级 AUROC 0.477、像素 AUROC 0.680、AUPRO 0.341，自编码器是 0.817、0.829、0.680。差分还差 zipper。
- VisA 种子 1 的 fryum 图像级 AUROC 是 0.958，种子 0 是 0.955。加上先前四类，这 5 类的均值从 0.917 升到 0.930。还差 7 类。
- 差分 15 类全部写完，`PART1_DONE` 在 05:36。zipper：差分图像级 AUROC 0.742、像素 AUROC 0.738、AUPRO 0.441、误报率 0.156；自编码器是 0.584、0.870、0.600、误报率 1.0。整图分数差分更高，定位自编码器更好。15 类均值：差分图像级 AUROC 0.651、AP 0.827、像素 AUROC 0.826、像素 AP 0.230、AUPRO 0.613、误报率 0.789；自编码器是 0.674、0.843、0.819、0.224、0.646、0.925。10 个物体上差分仍更高，图像级 AUROC 0.738 对 0.602，AUPRO 0.750 对 0.629。
- 448、1-shot 里已经有三个种子的 5 类是 bottle、cable、capsule、carpet、grid。三类种子的图像级 AUROC 均值是 0.972、0.945、0.965，总均值 0.961，标准差 0.014。种子 1 偏低是因为 capsule。这还不是 15 类。
- zipper 差分的示意：`combined/001` 判 NG，分数 0.699，热点在拉链下端断开的齿上；`good/008` 判 OK，分数 0.410。图在 `outputs/demo/zipper_difference_ng.png` 和 `zipper_difference_ok.png`。151 张里有 80 张缺陷被判 NG，27 张正常图被判 OK。
- hazelnut 的 1-shot 三个种子图像级 AUROC 是 0.982、0.993、0.966，均值 0.980，标准差 0.013。加上它，已经有三个种子的 6 类均值是 0.974、0.953、0.965，总均值 0.964，标准差 0.011。还差 9 类。
- 这 6 类在种子 0 上，448 和 672 的官方图像级 AUROC 是 0.974 对 0.975，AUPRO 是 0.955 对 0.960。分辨率差几乎不在这 6 类里。672 的定位提升主要在还没有第三个种子的 zipper 和 screw。448、672、VisA、AnomalyCLIP 训练和 672 验证尾图都还在跑。
- 单元测试 51 个里有 1 个失败：AnomalyCLIP 的逐图分数钩子已经写进 `test.py`，测试仍指望第一次插入返回真。测试现在会先剥掉已有钩子再检查插入。该测试已通过。
- 还没有第三个种子的 9 类里，前两个官方种子的图像级 AUROC 差距超过 0.02 的只有 transistor（0.937 对 0.895）和 pill（0.915 对 0.944）。leather、toothbrush、wood、tile、zipper、metal_nut 的差距都不超过 0.003。第三个种子会改总均值的，主要是 transistor 和 pill。
- 验证尾图打分在 Python 结束时也会打印完成标记，不再只靠外层脚本。448 那次就是外层脚本在分数写完后解析失败。正在跑的 672 进程还是旧代码，这行要等下次启动才生效。外层脚本如果能跑到最后，仍会再打一次标记。
- leather 的 1-shot 三个种子图像级 AUROC 都是 1.000，标准差是 0。加上它，已有三个种子的 7 类均值是 0.977、0.959、0.970，总均值 0.969，标准差 0.009。
- 448、1-shot、种子 0 的工作点用在验证尾图自己身上：15 类平均有 94.7% 的正常验证图被判成 NG。bottle、cable、capsule、carpet、grid、pill、screw、transistor、wood 是 10/10。最轻的 hazelnut 也有 7/10。最大像素对像素均值加 3 倍标准差这条规则，在 AnomalyDINO 上同样几乎总是响。图像级 AUROC 用的是另一套最高 1% 的平均，不受这个工作点影响。
- 同一套工作点在种子 0 上不随参考图变多而放松。1/2/4/8-shot 的正常验证图被判 NG 的比例是 0.947、0.947、0.960、0.967。8-shot 比 1-shot 还略高。参考图从 1 张加到 8 张，OK/NG 仍然几乎总是响。
- 1-shot 的三个种子都是这样。正常验证图被判 NG 的比例是 0.947、0.933、0.960。不是某一个参考图切片造成的。
- 种子 0 上，1-shot 到 8-shot 时，15 类平均阈值从 0.150 降到 0.107，验证图最大像素从 0.222 降到 0.159。两边一起降，最大像素仍高于阈值，所以判 NG 的比例没有下降。
- 8-shot 在三个种子上都比 1-shot 更常把正常验证图判成 NG：1-shot 是 0.947、0.933、0.960，8-shot 是 0.967、0.960、0.973。参考图变多没有让 OK/NG 变松，三个种子一致。
- AnomalyCLIP 的 VisA 提示目前写到 `epoch_8.pth`，文件时间是 05:26。15 轮还没完，测 MVTec 和提示-视觉融合都还要等这轮，以及后面用 MVTec 再训的那一轮。
- 448、1-shot、两个官方种子的 AUPRO，15 类里最大差距是 cable 的 0.016（0.887 对 0.871），中位数是 0.004。capsule 的图像级 AUROC 差了 0.119，AUPRO 只差 0.010。定位对参考图切片很稳，整图检测不是。
- 像素 AUROC 更稳。两个官方种子的最大差距是 wood 的 0.010（0.954 对 0.944），中位数是 0.001。capsule 的像素 AUROC 是 0.985 对 0.980。
- 图像 AP 的最大差距是 transistor 的 0.063（0.906 对 0.844），其次是 capsule 的 0.029（0.985 对 0.955）。中位数是 0.001。capsule 的图像级 AUROC 掉了 0.119，AP 只掉 0.029，因为高分样本仍然靠前。
- AnomalyCLIP 测试不再等两份 `epoch_15.pth` 都齐。VisA 上学到的提示一出现就测 MVTec，MVTec 上学到的提示出现后再测 VisA。两边的 `metrics.json` 都在才打印 `ANOMALYCLIP_TEST_DONE`。现在 VisA 的第 15 轮还没有，脚本会退出。
- AnomalyCLIP 的 VisA 提示已经写到 `epoch_9.pth`，时间是 05:57。训练进程还在，这一轮大约 30 分钟。后面还有 6 轮 VisA，然后才用 MVTec 再训。
- 672、1-shot、carpet 的种子 1 图像级 AUROC 是 1.000，种子 0 的官方数也是 1.000。448 的三个种子同样是 1.000。这一类在两个分辨率、现有种子上都是满分。
- pill 的 1-shot 三个种子图像级 AUROC 是 0.915、0.944、0.911，均值 0.923，标准差 0.018。种子 2 靠近种子 0，种子 1 是高的那次。加上它，已有三个种子的 9 类均值是 0.973、0.962、0.967，总均值 0.967，标准差 0.005。还差 6 类，其中 transistor 仍是最可能再拉动均值的一类。
- 672、1-shot、grid 的种子 1 图像级 AUROC 是 0.993，种子 0 的官方数是 0.992。448 的三个种子是 0.991、0.993、0.992。两个分辨率都停在 0.99。
- metal_nut 的 1-shot 三个种子图像级 AUROC 是 0.997、1.000、1.000，均值 0.999，标准差 0.002。加上它，已有三个种子的 8 类均值是 0.980、0.964、0.974，总均值 0.973，标准差 0.008。还差 7 类。
- VisA 种子 1 的 macaroni1 图像级 AUROC 是 0.786，种子 0 是 0.854。加上先前 5 类，这 6 类的均值变成 0.906 和 0.906。cashew 的回升被 macaroni1 的下降抵消了。还差 6 类。
- VisA 种子 1 的 macaroni2 图像级 AUROC 是 0.562，种子 0 是 0.602。这是目前最弱的一类，第二个种子更低。加上它，已有两个种子的 7 类均值是 0.863 和 0.857。还差 pcb1、pcb2、pcb3、pcb4、pipe_fryum。
- screw 的 1-shot 三个种子图像级 AUROC 是 0.889、0.881、0.903，均值 0.891，标准差 0.011。种子 2 是高的那次。加上它，已有三个种子的 10 类均值是 0.964、0.954、0.961，总均值 0.960，标准差 0.005。还差 tile、toothbrush、transistor、wood、zipper。
- 448、1-shot、种子 2 的 15 类图像级 AUROC 都已从 patch 网格算出，和官方图像级 AUROC 用的是同一个最高 1% 平均。三个种子的 15 类均值是 0.970、0.960、0.965，总均值 0.965，种子间标准差 0.005。像素级还要等 `metrics_seed=2.json`。
- 种子 2 新齐的 5 类：tile 是 0.999、1.000、1.000，均值 1.000。toothbrush 是 0.994、0.994、1.000，均值 0.996。wood 是 0.991、0.991、0.982，均值 0.988。zipper 是 0.988、0.987、0.990，均值 0.989。transistor 是 0.937、0.895、0.895，均值 0.909，标准差 0.024。种子 2 停在种子 1 的低位，没有像 capsule 那样回到种子 0。
- 672、1-shot、种子 1：hazelnut 图像级 AUROC 是 1.000，种子 0 的官方数是 0.987。leather 两个种子都是 1.000。metal_nut 是 1.000，种子 0 是 0.998。
- VisA 种子 1：pcb1 是 0.887，种子 0 是 0.881。pcb2 是 0.826，种子 0 是 0.852。已有两个种子的 9 类均值是 0.864 和 0.857。还差 pcb3、pcb4、pipe_fryum。
- 效率只收录已经写完 `metrics_seed` 的种子，记忆库时间每个类别只计一次。448 种子 0：逐图 0.440 秒，记忆库 0.579 秒（0.241 到 2.306）。种子 1：逐图 1.715 秒，记忆库 4.004 秒，这是 CPU 已经被别的任务占住之后量到的。672 种子 0：逐图 3.208 秒，记忆库 9.064 秒，也是在这个负载下量的，不能当成分辨率本身慢 7 倍。VisA 种子 0：2162 张，逐图 1.609 秒，记忆库 3.189 秒。整类图齐了才打分的检查每 15 分钟看一次完成文件。
- 448、1-shot、三个种子、15 类的图像级 AP 均值是 0.985、0.979、0.981，总均值 0.981，种子间标准差 0.003。transistor 的 AP 是 0.906、0.844、0.862，均值 0.871，标准差 0.032。capsule 的 AP 在种子 2 回到 0.987，和它的图像级 AUROC 一样，种子 1 是低的那次。
- 672、1-shot、种子 1 又齐了 4 类。pill 是 0.949，种子 0 官方是 0.927。screw 是 0.936 对 0.931。tile 是 1.000 对 1.000。toothbrush 是 0.972 对 0.981。两边都有的 12 类，图像级 AUROC 是 0.974 对 0.967，AP 是 0.991 对 0.990。还差 transistor、wood、zipper。
- VisA 种子 1 的 12 类图像分数都齐了。pcb3 是 0.800，种子 0 是 0.847。pcb4 是 0.599，种子 0 是 0.861，AP 从 0.848 掉到 0.539。pipe_fryum 是 0.919，种子 0 是 0.887。12 类均值：种子 0 图像级 AUROC 0.864、AP 0.880，种子 1 是 0.836 和 0.848。两个种子的总均值是 0.850 和 0.864，标准差 0.020 和 0.023。种子 1 被 pcb4 拉低。
- AnomalyCLIP 的 VisA 提示已经写到 `epoch_12.pth`，时间是 07:23。训练进程还在。第 10 到第 12 轮大约每 30 分钟一轮，后面还有 3 轮 VisA，然后才用 MVTec 再训。第 15 轮权重一出现，检查脚本会在 GPU 5 上测 MVTec，接着打验证尾图并做提示-视觉融合。两次查看间隔 15 分钟。
- VisA 的 15 轮在 08:28 写完，`epoch_15.pth` 与前面各轮一样大。外层脚本在接着训 MVTec 时退出码 127，`--save_path` 被当成了命令，MVTec 那一轮没有开始。已有的第 15 轮权重会跳过，只补训 MVTec。GPU 5 上有别的任务，MVTec 测试和验证尾图改到 GPU 6。
- 448、1-shot、三个种子的官方全分辨率结果齐了。15 类均值：图像 AUROC 是 0.970、0.960、0.965，总均值 0.965，标准差 0.005。图像 AP 是 0.985、0.979、0.981，总均值 0.981，标准差 0.003。像素 AUROC 是 0.965、0.962、0.961，总均值 0.963，标准差 0.002。AUPRO 是 0.918、0.917、0.917，总均值 0.917，标准差 0.001。图像级和 patch 网格上的数一致。定位几乎不随种子变。
- capsule 的图像 AUROC 是 0.939、0.820、0.941，AUPRO 是 0.970、0.961、0.967。种子 1 只把整图检测拉低。transistor 的图像 AUROC 是 0.937、0.895、0.895，AUPRO 三次都在 0.64 附近。
- 672、1-shot、两个官方种子的图像 AUROC 是 0.972 和 0.963，均值 0.968，标准差 0.006。AUPRO 是 0.927 和 0.929，均值 0.928。比同一种子的 448 高 0.009 和 0.012。capsule 种子 1 仍是 0.825，AUPRO 仍是 0.967。zipper 的 AUPRO 是 0.890 和 0.908，screw 是 0.936 和 0.939。
- 448、2-shot、种子 0 先齐了 bottle、cable、capsule、carpet、grid、hazelnut、leather。这 7 类的图像 AUROC 均值，1-shot 是 0.977，2-shot 是 0.978。还差 8 类和另外两个种子。
- 同一批测量里，448 种子 2 逐图 2.632 秒、记忆库 7.203 秒，672 种子 1 逐图 4.976 秒、记忆库 12.446 秒。都比各自的种子 0 更慢，当时 CPU 更挤。
- 1-shot 种子 1 的 448 和 672 patch 网格也融合完了，15 类。图像级 AUROC：448 是 0.960，672 是 0.963，融合是 0.968。图像 AP 融合是 0.984。patch 网格上的 AUPRO：448 是 0.907，672 是 0.884，融合是 0.912。这里的 672 定位仍低于 448，和官方全分辨率相反，所以这组 AUPRO 只能当 patch 网格上的数。
- 种子 1 里融合把图像级抬过两个分辨率的有 transistor（0.895、0.908、0.932）和 pill（0.944、0.949、0.965）。screw 的融合 0.915 落在 0.881 和 0.936 之间。capsule 三个数都在 0.82 附近，融合补不回这张参考图。
- 两个种子、15 类的融合图像级 AUROC 均值是 0.973，标准差 0.007。patch 网格 AUPRO 均值是 0.913，标准差 0.001。
- VisA、448、1-shot 的种子 1 官方全分辨率结果也齐了。12 类图像 AUROC 是 0.864 对 0.836，均值 0.850，标准差 0.020。图像 AP 是 0.880 对 0.848，均值 0.864，标准差 0.023。像素 AUROC 是 0.975 对 0.972，均值 0.973，标准差 0.002。AUPRO 是 0.907 对 0.900，均值 0.904，标准差 0.005。定位整体很稳。pcb4 是例外：图像 AUROC 0.861 掉到 0.599，AUPRO 0.806 掉到 0.708。macaroni2 的图像级仍弱（0.602、0.562），像素 AUROC 仍在 0.98。
- pcb4 种子 1 的正常图均值是 0.224（0.166 到 0.388），缺陷图均值是 0.228（0.165 到 0.345）。两组几乎叠在一起。种子 0 是 0.197 对 0.248，还有间隔。`bad/023` 的热力落在左侧接口，和掩码位置一致，分数 0.229；一张正常板 `good/0214` 分数 0.231，热力在芯片和接口上。VisA 没有阈值验证划分，这两张只标了检测分数，没有 OK/NG。图在 `outputs/demo/visa_pcb4_seed1_bad.png` 和 `visa_pcb4_seed1_good.png`。
- 448、2-shot、种子 0 又齐了 metal_nut 和 pill，一共 9 类。这 9 类的图像 AUROC 均值，1-shot 种子 0 是 0.973，2-shot 是 0.978。变化超过 0.01 的只有 pill（0.915 到 0.959）和 cable（0.932 到 0.918）。还差 screw、tile、toothbrush、transistor、wood、zipper。
- 672、1-shot、bottle 的三个种子图像 AUROC 是 1.000、0.997、0.995。这一类在两个分辨率上都停在 0.995 以上。
- AnomalyCLIP 的 MVTec 测试在 08:41 开始，用的是 VisA 上训完的 `epoch_15.pth`，GPU 6。MVTec 提示还在 GPU 2 上训练，VisA 测试要等那份权重。
- Part 1 的对比表按 10 个物体和 5 个纹理分开均值。差分在物体上更好：图像 AUROC 0.738 对自编码器 0.602，像素 AUROC 0.899 对 0.814，AUPRO 0.750 对 0.629。纹理上相反：图像 AUROC 0.477 对 0.817，像素 AP 0.050 对 0.305，AUPRO 0.341 对 0.680。纹理差分的误报率 0.548 低于物体的 0.909，是阈值抬得高，不是分开了正常和缺陷。
- AnomalyDINO 的 448、1-shot、三个官方种子也按这两组分开。物体图像 AUROC 是 0.957、0.942、0.950，均值 0.950，标准差 0.007。纹理是 0.996、0.997、0.995，均值 0.996，标准差 0.001。AUPRO 物体 0.901，纹理 0.949。672 的两个种子同样：物体图像 0.954，纹理 0.995。Part 1 最弱的纹理，在 AnomalyDINO 上几乎是满分。
- transistor 的官方 AUPRO 停在 0.64，不是因为最热点完全离开缺陷。448、1-shot、种子 0 的 patch 网格上，缺陷像素比背景平均只热 0.083（中位数 0.046）。cable 是 0.176，capsule 是 0.249，carpet 是 0.370。整图分数仍能把有缺陷的图排到前面，缺陷区域本身不够热，所以区域重叠上得慢。GrabCut 再削掉缺陷 patch，定位会更差。
- 多尺度融合没有把 transistor 的缺陷热度抬过较细的那张图。用同一套验证像素做标准化之后，缺陷减背景：种子 0 是 448 的 2.50、672 的 2.81、融合的 2.65；种子 1 是 3.02、3.51、3.27。融合值落在两者中间。种子 1 的图像 AUROC 仍从 0.895 和 0.908 升到 0.932，所以增益来自两套检测分数互补，不是缺陷区域变得更热。
- 对比表会在 AnomalyCLIP 的 `metrics.json` 写出来之后，把零样本结果和 AnomalyDINO 的 448、1-shot 种子均值并排。比较图像 AUROC、图像 AP 和像素 AUROC。AUPRO 不并排，AnomalyCLIP 用 200 个阈值积分，AnomalyDINO 用官方的排序积分。MVTec 测试还在跑，这张对照现在是空的。
- 448、2-shot、种子 0 现在有 12 类。和 1-shot 种子 0 比，这 12 类的图像 AUROC 均值是 0.970 对 0.976。变化超过 0.01 的是 pill（0.915 到 0.959）、screw（0.889 到 0.909）和 cable（0.932 到 0.918）。还差 transistor、wood、zipper。
- VisA 种子 2 先齐了 3 类。candle 三个种子是 0.877、0.861、0.880，均值 0.873，标准差 0.010。capsules 是 0.983、0.989、0.991，均值 0.987，标准差 0.005。cashew 是 0.785、0.864、0.904，均值 0.851，标准差 0.061，一个种子比一个种子高。
- 672、1-shot、cable 的三个种子图像 AUROC 是 0.935、0.928、0.916，均值 0.926，标准差 0.010。第三个种子最低，和 448 上 cable 的走势一样。
- 这 12 类里，2-shot 多出来的图像 AUROC 几乎都在物体上。已齐的 8 个物体，1-shot 种子 0 是 0.956，2-shot 是 0.965。已齐的 4 个纹理是 0.998 和 0.998。纹理本来就满，加一张参考图不动。还缺的 transistor 和 zipper 都是物体，wood 是纹理。
- 提示和视觉分数融合后的文件会带上图像 AUROC 和 AP。路径里含 `good` 的测试图算正常，其余算缺陷。两类都有才写这两个数。AnomalyCLIP 的测试还没写完，融合还没开始。
- 448 的验证尾图三个种子都齐了。正常图被判 NG 的比例，1/2/4/8-shot 的均值是 0.947、0.953、0.960、0.967，标准差 0.013、0.018、0.013、0.007。参考图变多，这个比例不降。672 的 1/2/4-shot 三个种子是 0.958、0.962、0.960。8-shot 还缺种子 2，已有两个种子的均值是 0.963。换到 672 也没有把正常图放回去。
- 同一批 12 类的图像 AP 几乎不动：1-shot 种子 0 是 0.989，2-shot 是 0.991。8 个物体是 0.984 对 0.987，4 个纹理是 0.999 对 0.999。pill 的 AP 从 0.981 到 0.992，AUROC 却从 0.915 到 0.959。screw 的 AP 从 0.957 到 0.967，AUROC 从 0.889 到 0.909。多一张参考图主要拉开中段排序，高分段本来就比较准。
- 这三张图的像素指标是在 448 的 patch 网格上比的，不是官方全分辨率。pill 的像素 AUROC 从 0.951 到 0.953，AUPRO 从 0.877 到 0.882，像素 AP 从 0.505 到 0.543。图像 AUROC 升了 0.044，定位几乎没动。screw 的像素 AUROC 从 0.953 到 0.958，AUPRO 从 0.822 到 0.842。cable 的图像 AUROC 下降，AUPRO 仍是 0.830。
- transistor 的 2-shot 种子 0 图也齐了。图像 AUROC 是 0.937 对 1-shot 的 0.938，AP 是 0.904 对 0.906，整图检测没变。同一套 patch 网格上 AUPRO 从 0.532 升到 0.579。和 pill 相反：pill 是整图排序变好、定位几乎不动，transistor 是定位变好、整图分数不动。
- 算上 transistor，已齐的 13 类图像 AUROC 均值是 0.967 对 0.973。这 13 类的 patch 网格 AUPRO 均值是 0.866 对 0.873。还差 wood 和 zipper。
- 记忆库的 patch 上限按 8 个旋转、掩码之前计算。448 的 1/2/4/8-shot 是 8192、16384、32768、65536。672 是 18432、36864、73728、147456。8-shot 672 是 1-shot 448 的 18 倍。纹理不做前景掩码，会接近这个上限。前面测到的逐图时间受 CPU 拥挤影响，不能当成这个倍数。
- 672、1-shot、capsule 的种子 1 图像级 AUROC 是 0.825，种子 0 的官方数是 0.937。448 种子 1 是 0.820。同一张参考图在两个分辨率上都把整图检测拉低，换分辨率补不回来。
