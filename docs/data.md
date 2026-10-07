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

这是 AnomalyDINO 的默认路径。VisA 是课程唯一必做数据集。缺陷掩码是 `ground_truth/bad/<stem>.png`。Part 1 的划分在 `outputs/splits/visa`，尾部 10 张做阈值验证，前缀留给 1/2/4/8-shot。解压到 `data/visa_raw` 后运行：

```bash
/home/zhangyuming/CV_project/env_cv/bin/python -m src.inspection.prepare_visa
```

## MVTec AD 2

2026-10-01 的作业说明不再要求下载它。它只出现在阅读列表里。本地没有 `data/mvtec_ad_2/`，也不再补下官方归档。下面的目录布局留给阅读，当前实验不读这些路径。

公开测试图的官方布局是：

```text
data/mvtec_ad_2/<category>/test_public/good/
data/mvtec_ad_2/<category>/test_public/bad/
data/mvtec_ad_2/<category>/test_public/ground_truth/bad/<stem>_mask.png
```

`test_private` 和 `test_private_mixed` 不读，也不上传到 benchmark.mvtec.com。八个类别是 can、fabric、fruit_jelly、rice、sheet_metal、vial、wallplugs、walnuts。

## 自采检验集

照片在 `CV资料/Project2/`，用符号链接放进 `data/wild_mouse/`。程序按文件名排序后，用种子 0 从受控正常图里抽 10 张做阈值验证图，其余做参考池。

实际拍到的和原来的 80 张清单不一样：

| 来源文件夹 | 链接到 | 张数 | 说明 |
| --- | --- | --- | --- |
| 完好受控条件 | `controlled/good/` | 40 | 8 次快门，每次 5 张，同一机位 |
| 背景改变 | `shift_background/good/` | 10 | 只换了桌面 |
| 白色印记 | `controlled/ink/` | 15 | 白色印记，当作色笔这一类 |
| 胶带 | `controlled/tape_or_missing/` | 10 | 一小块胶带 |
| 划痕 | `controlled/scratch/` | 9 | 另一台相机，分辨率和机位都不同 |

没有只改光照的一组，所以 `shift_light/good/` 是空的。缺陷图没有掩码，像素级指标先不算。

```text
data/wild_mouse/
  controlled/good/                 # 40 张，链到完好受控条件
  shift_background/good/           # 10 张，链到背景改变
  controlled/scratch/              # 9 张，链到划痕
  controlled/ink/                  # 15 张，链到白色印记
  controlled/tape_or_missing/      # 10 张，链到胶带
```

掩码仍按同尺寸 PNG 来：白色是缺陷，黑色是其余区域。正常图不要掩码。这批照片里还没有掩码。
