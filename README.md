# plant-disease-classification

COMP 664 project: tomato disease classification across clean and field images, with six independently trained models across three paired seeds.

- [Executed Jupyter Notebook](submission/Tomato_Project.ipynb)
- [Technical report (PDF)](submission/Tomato_Report.pdf)
- [Slides (PowerPoint)](submission/Tomato_Slides.pptx) and [PDF](submission/Tomato_Slides.pdf)
- [Installation and reproduction](project/README.md)
- [Training protocol and evidence](project/results/fresh_views/)

优先打开 **submission/Tomato_Project.ipynb** 查看代码、真实训练记录、图表和保存输出。

本次已实际新增训练 6 个模型：3 个配对种子，每个种子包含普通增强和强增强，各训练 8 轮、每轮重新生成视图。原 ZIP 的两组权重和结果另行保留。

| 新实验，三种子均值 ± 样本标准差 | 普通训练 | 强裁剪和颜色增强 |
| --- | ---: | ---: |
| 干净准确率 | 97.26 ± 0.16% | 96.66 ± 0.58% |
| 田间准确率 | 18.78 ± 1.10% | 20.13 ± 0.74% |
| 干净 macro-F1 | 96.32 ± 0.18% | 95.71 ± 0.50% |
| 田间 macro-F1 | 7.76 ± 0.83% | 9.16 ± 0.73% |

强增强平均表现有所改善，但 3 个种子中只有 2 个改善，1 个略降；田间性能仍然很低。

```text
plant-disease-classification/
├── README.md
├── submission/
│   ├── Tomato_Project.ipynb   已执行 Notebook：37 个单元，21 个代码单元
│   ├── Tomato_Report.pdf      7 页 LaTeX 技术报告
│   ├── Tomato_Slides.pptx     12 页基础学术幻灯片
│   └── Tomato_Slides.pdf
├── project/
│   ├── README.md             完整安装、演示、重训和复算说明
│   ├── prepare.py            数据映射、划分与去重
│   ├── experiment_fresh.py   新增三种子、逐轮新视图训练
│   ├── summarize_fresh.py    匹配预算审计和种子汇总
│   ├── experiment.py         原训练实现与共享模型
│   ├── results/
│   │   ├── fresh_views/      新实验：协议、日志、汇总和 seed_664/665/666
│   │   └── …                 原实验的权重、预测和复核记录
│   ├── examples/             两张离线示例及来源
│   ├── report/               LaTeX 源文件及图表
│   ├── slides/               幻灯片源文件及数据
│   ├── tests/                10 项测试
│   └── provenance/           上游切分、叶片信息和下载校验值
├── docs/
│   ├── START_HERE_ZH.txt      中文结果解读和提交说明
│   └── PROJECT_UPDATE.md
└── SHA256SUMS.txt
```

## 查看与演示

Notebook 保存了完整顺序执行输出。其默认 Run All 重算三种子统计并加载新权重进行示例推理，不下载数据，也不重训。重新运行需要完整解压项目；单独下载 Notebook 可以查看保存输出。

以下命令均先进入 project/，按其中 README 安装 Python 3.12 展示环境：

```bash
cd project
python demo.py examples/field.jpg
python demo.py examples/field_second.jpg
python summarize_fresh.py --out results/fresh_views
```

演示默认使用预先固定的 seed 664 新模型。GPU 重训使用独立的 requirements-train.txt 环境；Notebook 展示依赖为 requirements-notebook.txt。请勿混用两个环境的 torch 固定版本。

提交前补上实际组员信息。完整原始数据不重复打包；固定版本下载器和完整文件清单均已提供。
