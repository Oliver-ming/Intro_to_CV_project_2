# 少样本工业外观异常检测

只用无缺陷参考图判断测试图是否异常，并给出像素级热力图。课程要求见本地 PDF，不进 git。已经定下的术语在 `CONTEXT.md`，决定在 `docs/adr/`。

## 环境

Python 使用仓库外的 `env_cv`：

```bash
/home/zhangyuming/CV_project/env_cv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
/home/zhangyuming/CV_project/env_cv/bin/python -m pip install -r requirements.txt
```

不要把 PyTorch 降到 AnomalyCLIP 钉死的 2.0.0。上游代码的提交号在 `third_party/README.md`。

## 数据

下载与目录说明在 `docs/data.md`。

```bash
# 压缩包下载完成后
mkdir -p data
tar -xJf data/archives/mvtec_anomaly_detection.tar.xz -C data
```

VisA 解压后还要整理成 AnomalyDINO 使用的 `data/VisA_pytorch/1cls/`。

## Part 1

正常图划分对所有方法共用，规则见 `docs/adr/0006-shared-normal-split.md`。检测分数和工作点不是同一个数，规则见 `docs/adr/0005-operating-point-mean-plus-3std.md`。

```bash
PY=/home/zhangyuming/CV_project/env_cv/bin/python
$PY -m src.inspection.build_splits --data-root data/mvtec_anomaly_detection
$PY -m src.part1.run_difference --categories bottle
$PY -m src.part1.train_ae --categories bottle
$PY -m unittest tests.test_protocol
```

Part 1 的差分基线把测试图配准到最接近的正常参考，再用 RGB 绝对差和 SSIM 残差的平均作为热力图。自编码器只在参考池上训练。

单张图 demo，在该类的 `scores.json` 已经写好时会同时标出 OK/NG：

```bash
$PY -m src.demo.inspect \
  --image data/mvtec_anomaly_detection/bottle/test/broken_large/000.png \
  --split outputs/splits/mvtec/bottle.json \
  --scores outputs/part1/difference/bottle/scores.json \
  --out outputs/demo/bottle.png
```

## Part 2

AnomalyDINO 的 MVTec、VisA 和 672 分辨率复现分别由 `scripts/run_anomalydino_mvtec.sh`、`scripts/run_anomalydino_visa.sh`、`scripts/run_anomalydino_mvtec_672.sh` 启动。AnomalyCLIP 的提示训练和测试是 `scripts/run_anomalyclip_train.sh` 与 `scripts/run_anomalyclip_test.sh`。长任务结束时会在各自日志里打印 `*_DONE`。
