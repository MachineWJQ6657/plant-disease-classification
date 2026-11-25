# 复现记录（2026-11-24/25，隔离副本，CPU）

环境：Python 3.12.3，torch 2.5.1 / torchvision 0.20.1（PyPI 安装，README 中的 CPU 索引 download.pytorch.org 被出口代理拒绝），
numpy 2.3.5、pandas 2.2.3、scikit-learn 1.8.0、Pillow 13.0.0（与 requirements.txt 一致）。无 GPU。

| 步骤 | 结果 |
|---|---|
| `sha256sum -c SHA256SUMS.txt` | 195 个文件全部 OK |
| `verify.py --out results/fresh_views/seed_664/665/666` | 三个种子均通过，独立重算的准确率、macro-F1 与保存结果一致 |
| `summarize_fresh.py --out results/fresh_views` | 通过；生成的 CSV/JSON 与原件逐字节一致（原件为 Windows CRLF，去掉 CR 后完全相同） |
| `python -m unittest discover -s tests -v` | 10 项全部通过 |
| `demo.py examples/field.jpg` / `field_second.jpg` | 与文档一致：第一张两个模型都判 late blight；第二张普通判 early blight、强增强判 late blight |
| 重新执行 `Tomato_Project.ipynb`（nbconvert，默认开关） | 21 个代码单元无错误；16 个单元输出文本完全相同，另 5 个仅差 Python/torch 版本号与对象内存地址，内部 PASS 检查均通过 |
| 用 `report/main.tex` 重新编译原报告 | 7 页，文本与已提交的 `Tomato_Report.pdf` 完全一致 |

## 未能运行（环境限制）
- **重新训练（GPU，torch 2.7.1+cu128）**：本环境无 GPU，且需要原始数据。
- **从原始图片重放（`evaluate.py`）与数据重建（`download.py`）**：原始数据下载源 `codeload.github.com` 返回 403，
  `huggingface.co` 连接失败，出口策略不允许，没有绕过。
  因此“24,558 个标签与重放一致”仅由仓库内的 `replay_verification.json` 记录支持，我没有独立重跑。
