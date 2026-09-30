# 上游代码

这两个仓库只在本地检出，不进本项目的 git 历史。版本以检出时的提交号为准。

| 仓库 | 用途 | 提交 |
| --- | --- | --- |
| [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) | Part 2 少样本复现 | `b9d1c2648e3a5247437d4d953d907a8f3d994457` |
| [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) | Part 2 零样本对照 | `3911738c0867544f545a076ad78f3f11d9ecbfdf` |

重新检出：

```bash
git clone https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
git -C third_party/AnomalyDINO checkout b9d1c2648e3a5247437d4d953d907a8f3d994457

git clone https://github.com/zqhang/AnomalyCLIP.git third_party/AnomalyCLIP
git -C third_party/AnomalyCLIP checkout 3911738c0867544f545a076ad78f3f11d9ecbfdf
```

AnomalyCLIP 的 `requirements.txt` 把 torch 钉在 2.0.0。本项目不降级，统一使用 `env_cv` 里的 CUDA 版 PyTorch。
