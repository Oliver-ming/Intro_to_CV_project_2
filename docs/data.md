# 数据放哪里

压缩包和图像都在 `data/`，这个目录不进 git。

## MVTec AD

压缩包与官方文件大小一致（5264982680 字节）。官方 mydrive 在这台机器上只有几十 KB/s，改从 Hugging Face 的同一份归档下载：

```bash
wget -O data/archives/mvtec_anomaly_detection.tar.xz \
  "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz"
mkdir -p data
tar -xJf data/archives/mvtec_anomaly_detection.tar.xz -C data
```

解压后应是：

```text
data/mvtec_anomaly_detection/<category>/{train,test,ground_truth}/
```

15 个类别都会用到。协议见 `docs/adr/0006-shared-normal-split.md`。

## VisA

```bash
wget -O data/archives/VisA_20220922.tar \
  "https://huggingface.co/datasets/XiranZhao/iad/resolve/main/VisA_20220922.tar"
```

官方 S3 在这台机器上只有几 KB/s。上面这份与官方归档同为 1.93GB。官方 1-class 划分放到：

```text
data/VisA_pytorch/1cls/<category>/{train,test,ground_truth}/
```

这是 AnomalyDINO 的默认路径。VisA 是主跨数据集。

## MVTec AD 2

只在 Part 3 的压力测试里使用公开测试集，不进主表，也不提交私有榜。放到 `data/mvtec_ad_2/`。

## 自采检验集

照片放到 `data/wild_mouse/`。文件名随意，程序按文件名排序后，用种子 0 从受控正常图里抽 10 张做阈值验证图，其余做参考池。

```text
data/wild_mouse/
  controlled/good/                 # 30 张无缺陷
  shift_light/good/                # 10 张，只改光照
  shift_background/good/           # 10 张，只改背景
  controlled/scratch/              # 10 张
  controlled/ink/                  # 10 张
  controlled/tape_or_missing/      # 10 张
  masks/scratch/                   # 与缺陷图同名的二值掩码
  masks/ink/
  masks/tape_or_missing/
```

掩码为同尺寸 PNG，白色是缺陷，黑色是其余区域。正常图不要掩码。三种缺陷是细划痕、可洗色笔印、贴胶带或拿掉脚垫。
