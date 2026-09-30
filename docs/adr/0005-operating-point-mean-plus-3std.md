# 工作点取验证图像素的均值加 3 倍标准差

作业禁止用测试标签选阈值。MVTec AD 2 在无缺陷验证图的像素上取均值加 3 倍总体标准差，并用这个数同时做热力图二值化和图像判定；图像判定比较的是最大像素。AnomalyDINO 的图像级 AUROC 用另一套分数：距离最高的 1% patch 的平均值，官方实现是 `mean_top1p`。这两套数不互相替代。Part 1 没有 patch，检测分数改为其热力图上最高 1% 像素的平均。

## Considered Options

- 像素均值加 3 倍标准差做工作点，AnomalyDINO 检测分数用最高 1% patch（采用）
- 对图像分数再取均值加 3 倍标准差：比 MVTec AD 2 更严，也不是 AnomalyDINO 的检测分数
- 热力图最大像素同时当作检测分数：AnomalyDINO 比较过，不如最高 1% 的平均
