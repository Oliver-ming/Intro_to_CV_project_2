# MVTec 的官方少样本切片留在文件名前缀

AnomalyDINO 的三次重复不是随机抽样。官方代码按 `train/good` 文件名排序，第 `s` 次取 `[s*k:(s+1)*k]`。从训练集里随机抽走验证图会改变这些切片，复现数字对不上。阈值验证图改取排序后的尾部，最多 10 张，并且不覆盖 1/2/4/8/16-shot 的三次切片所能占用的最长前缀。这些文件留在磁盘上的 `train/good` 里。Part 1 的参考池用除尾部以外的正常图。

VisA 的 AnomalyDINO 用同一套切片，所以 Part 1 在 VisA 上也取这个尾部。掩码文件名是 `<stem>.png`，不是 MVTec 的 `<stem>_mask.png`。

## Considered Options

- 尾部做阈值验证，前缀留给官方切片（采用）
- 种子 0 随机抽 10 张：官方切片会变
- 用 `test/good` 定阈值：这些图属于官方测试划分
