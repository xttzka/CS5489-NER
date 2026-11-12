# 成员 A 的实验结果与复现

这里汇总 A 已完成的数据准备、M0–M2 经典基线和分析实验。环境安装、原始数据来源、文件目录和组内交接约定见 [项目 README](README.md)。下列路径以项目根目录为起点。

## 数据与共用评分

- 六套数据已转换为逐句 JSONL，并生成统一 spaCy 词性版本、各测试集固定的 300 个样本 ID、C 选提示词用的 CoNLL-2003 验证集 300 条、实体熟悉度索引、500 对 v2 实体替换样本和三套 v1 扰动数据。原始分词和标签留在 `data/processed/`，统一词性版本位于 `data/processed_pos/`。[数据准备和输出路径](README.md#2-下载转换并准备六套数据)
- 模型答卷按样本输出 `sample_id` 和 `tags`，标签与原词元逐一对齐，使用 IOB2。实体的起点、终点和类型都一致才计为正确。完整测试核对全部 ID，固定子集核对各套 `test_300.txt`。[共用接口与命令](README.md#5-bc-的共用接口) · [精确匹配评分](eval/score_ner.py) · [共有三类评分](eval/score_transfer.py) · [已见与未见评分](eval/score_familiarity.py)

## M0–M2 正式基线

M0 从训练实体构建区分大小写的词典，保留只有一种训练类别的名称并优先匹配最长跨度。M1 用逻辑回归逐词预测，M2 用线性链 CRF 联合预测整句标签。两者在每项对照内共用同一组词汇、书写形式、词缀、句法和上下文特征。

下表为实体级 micro-F1，按 M0、M1、M2 的顺序列出。正式报告另含总体与分类型 P、R、F1、1,000 次句子重采样的 95% bootstrap 区间和三对模型的配对 F1 差值。三套全量域内结果也记录词元准确率。[正式基线报告](results/m0_m2_baselines.json)

| 测试集 | 全量域内 F1 | 固定 300 条 F1 |
|---|---:|---:|
| CoNLL-2003 | 56.73%／73.39%／81.40% | 60.17%／69.93%／81.04% |
| WNUT-17 | 0.00%／14.40%／16.13% | 0.00%／16.71%／20.33% |
| MIT Restaurant | 44.52%／68.94%／77.25% | 43.62%／67.71%／77.82% |
| CoNLL++ | — | 46.97%／65.40%／70.30% |
| WikiGold | — | 24.01%／41.43%／57.81% |
| BTC | — | 10.85%／26.10%／25.17% |

三套域内评测各用本数据集训练和原生标签。M1/M2 统一使用 spaCy 词性、关闭短语标签，并使用同样的 ±1 特征窗口。M1 将 L-BFGS 停止容差收紧到 `1e-5`，在各自验证集比较 `C=0.1,1,10,100`。CoNLL-2003 和 MIT Restaurant 选出 `C=10`，WNUT-17 选出 `C=100`。CoNLL++、WikiGold 和 BTC 的固定子集复用同一套 CoNLL-2003 模型。前两套保留 CoNLL 四类，BTC 映射到 `PER`、`LOC`、`ORG`。

CoNLL-2003 在原 POS 和 chunk 设置下用训练集文档分组五折选出 `c1=0.0001`、`c2=0.1`。该历史设置的五折平均 F1 为 **87.7603%**，全量重训后的验证集 F1 为 **88.8190%**。统一 POS 基线沿用这两个系数。同一组 20 对系数在统一 POS 下按相同的文档分组五折重算，这两个系数的平均 F1 为 **87.74%**，网格内最高为 `(0.001, 0.1)` 的 **87.79%**，五折标准差约 1 个百分点。[统一 POS 网格](results/m2_pos_cv_check.json)WNUT-17 和 MIT Restaurant 各用连续五折在本训练集选参，分别为 `(0.0001, 0.1)` 和 `(0.01, 0.5)`。窗口和分折对照使用对应的五折与固定正则化参数。[CoNLL-2003 历史网格](results/m2_cv_grid.json) · [两套原生域内交叉验证](results/m2_native_cv.json) · [窗口与分折对照](results/m2_cv_controls.json)

| 训练集五折 | ±1 平均 F1 | ±2 平均 F1 | 本次对照中较高的窗口 |
|---|---:|---:|---|
| CoNLL-2003，按文档分折 | 87.7603% | 87.9935% | ±2 |
| WNUT-17，按连续块分折 | 46.3344% | 45.9394% | ±1 |
| MIT Restaurant，按连续块分折 | 75.9896% | 75.8090% | ±1 |

CoNLL-2003 的原 POS 和 chunk 对照中，±2 模型在完整验证集得 **89.5520%**。固定 ±1 时，按文档分折、按原始顺序连续分块、随机按句分折的平均 F1 分别为 **87.7603%、86.9084%、92.0547%**。五折合计，验证折中来自同折训练文档的句子分别有 **0、128、14,038** 条，总数都是 14,041 条。正式基线表使用统一词性和共同的 ±1 特征。[分折成员与逐折分数](results/m2_cv_controls.json)

## 已完成的对照实验

| 实验 | 口径和结果文件 |
|---|---|
| 时间泛化与实体熟悉度 | 同一组 CoNLL-2003 训练的 M0–M2，分别评估 3,453 条 CoNLL-2003 和 1,840 条 CoNLL++ 完整测试。M1/M2 使用统一词性，按四类报告总体、已见和未见 F1，各附 1,000 次句子重采样的 95% 区间。已见按完整词元序列 `casefold` 后是否出现在训练金标实体中划分，预测的误报按预测本身的名称归组。[M0–M2 年代对照](results/m0_m2_temporal.json) · [M2 熟悉度](results/m2_seen_unseen_f1.json) |
| 统一词性与跨领域迁移 | 正式三源五目标矩阵统一在共有三类上训练和测试。CoNLL-2003 四类训练、三类评分的结果独立记录。附熟悉度、大小写分组召回和实体重叠分析。[三类迁移矩阵](results/m2_transfer_matrix.json) · [CoNLL 四类训练参照](results/m2_pos_transfer.json) · [分组召回](results/m2_transfer_recall_groups.json) · [重叠分析](results/m2_transfer_overlap.json) |
| G1–G5、G2 与小写 | G1–G5 在 CoNLL-2003 同一组训练五折上逐组移除，使用原词性和短语标签。G2 在统一词性下比较三套完整域内测试及 CoNLL-2003 到 BTC 的迁移，衡量显式书写形式特征。小写实验使用同一个统一词性模型比较原始与全小写验证集。另有统一词性的三类验证集对照。[G1–G5](results/m2_feature_ablation.json) · [G2](results/m2_g2_transfer.json) · [小写](results/m2_pos_casing.json) · [词性对照](results/m2_pos_validation.json) |
| 泛化特征 | 固定统一词性 M2 的训练设置，单独加入 GeoNames 地名词典或 GloVe 6B 100d 聚类特征。`k=100, 500, 1000` 是预设比较条件，聚类中心只用有向量的训练集词型拟合。在两套完整新闻测试的四类上报告总体、已见和未见 F1。[额外特征报告](results/m2_extra_features.json) |
| 实体替换与文本扰动 | M0、M1、M2 各训练一次，在相同的 500 对 v2 样本以及完整 CoNLL-2003 的原文和三种 v1 扰动上比较。原／替换实体命中数依次为 M0 448／0、M1 475／311、M2 470／339。结果含配对区间及共有三类全小写与 BTC 掉分的描述性对照。[统一脚本](models/classic/run_robustness.py) · [汇总结果](results/m0_m2_robustness.json) |
| 错误分类与抽样 | 共用工具把范围错、类别错、两者皆错、误报和漏检分开。正式 M2 示例从 762 条类别错或误报中以种子 42 抽出 60 条。句外生成实体通过 LLM 原始输出核查。[错误分析工具](eval/analyze_ner_errors.py) · [M2 示例汇总](results/error_analysis/conll2003_m2/summary.json) |

## 复现顺序

实测使用 Windows x64 和 Python 3.11.16，经典模型在 CPU 上运行。按照 [安装环境](README.md#1-安装环境) 分别建立两个虚拟环境，在 PowerShell 设置 `$exp` 和 `$pos`。两份依赖文件列出了需要安装的固定版本。

| 环境 | 固定版本 |
|---|---|
| `$exp` 经典模型和绘图 | NumPy 2.0.1、SciPy 1.16.0、scikit-learn 1.9.1、joblib 1.6.0、narwhals 2.26.0、threadpoolctl 3.7.0、sklearn-crfsuite 0.5.0、python-crfsuite 0.9.12、Matplotlib 3.11.0。[requirements.txt](requirements.txt) |
| `$pos` 词性标注 | NumPy 2.4.6、spaCy 3.8.18、Thinc 8.3.14、blis 1.3.4、`en_core_web_sm` 3.8.0。[requirements-pos.txt](requirements-pos.txt) |

按 [原始数据清单](README.md#2-下载转换并准备六套数据) 运行统一准备入口。需要下载时加 `--download`，已有处理好的 JSONL 时可用 `--derived` 重建派生数据。

```powershell
& $exp -B -m data.prepare --all --download --pos-python $pos
```

随后运行调参和正式基线。`tune_crf` 是 CoNLL-2003 的调参入口，可用与当前代码和依赖匹配的历史报告复用折内结果，必要时重建验证集答卷。正式答卷写入 `preds/<数据集>/{m0,m1,m2}_baseline/0.jsonl`。

```powershell
& $exp -B -m models.classic.tune_crf --jobs 3
& $exp -B -m models.classic.m2_native_cv --jobs 3
& $exp -B -m models.classic.run_baselines --jobs 1
& $exp -B -m models.classic.m2_cv_controls --jobs 3
& $exp -B -m models.classic.m2_pos_cv_check --jobs 5
```

分析先生成 CoNLL-2003 迁移答卷，再运行依赖这些答卷的实验。其余输出和答卷位置见 [分析脚本清单](README.md#4-运行迁移年代和鲁棒性分析)。

```powershell
& $exp -B -m models.classic.m2_pos_transfer
& $exp -B -m eval.score_m2_familiarity
& $exp -B -m models.classic.run_temporal_baselines
& $exp -B -m eval.analyze_transfer_recall
& $exp -B -m models.classic.m2_transfer_matrix
& $exp -B -m eval.analyze_m2_transfer_overlap
& $exp -B -m models.classic.m2_feature_ablation --jobs 3
& $exp -B -m models.classic.m2_g2_transfer
& $exp -B -m models.classic.m2_pos_validation
& $exp -B -m models.classic.m2_pos_casing
& $exp -B -m models.classic.run_robustness --replacement-version v2
```

复跑额外特征时，将组内保留的 **2026-11-10 GeoNames 快照** `cities15000.zip` 和 `countryInfo.txt` 放在 `data/external/gazetteers/`，从 Stanford 的 `glove.6B.zip` 解出 `glove.6B.100d.txt` 放在 `data/external/glove/`。GeoNames 的公开下载是滚动更新的，原实验以报告中 `resources` 的 SHA-256 为准。运行后答卷写入 `preds/{conll2003,conllpp}/m2_extra_<条件>/0.jsonl`。[GeoNames 下载目录](https://download.geonames.org/export/dump/) · [Stanford GloVe 下载](https://huggingface.co/stanfordnlp/glove/blob/main/glove.6B.zip) · [资源校验值](results/m2_extra_features.json)

```powershell
& $exp -B -m models.classic.m2_extra_features
```

错误分析示例从正式 CoNLL-2003 M2 答卷生成。每次运行选一个新的输出目录，里面会有 `errors.jsonl`、`summary.json` 和 `review.csv`。

```powershell
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/m2_baseline/0.jsonl --output-dir results/error_analysis/conll2003_m2_rerun --sample-size 60 --seed 42
```

### 实体向量聚类和集成

CoNLL-2003 的 5,648 个测试实体中，GloVe 覆盖 5,534 个。`k=4` 的 ARI 为 **0.2536**、NMI 为 **0.2625**。逐实体分簇记录写入本地 `output/entity_clustering/glove_assignments.jsonl`，其中 668 个 MISC 实体另有子簇编号。B 提供两份 BERT 向量后，可加 `--base-bert <文件>` 和 `--fine-tuned-bert <文件>`，三种表示会在同一批有向量的实体上计算指标。[聚类脚本](analysis/cluster_entities.py) · [GloVe 结果](results/entity_clustering_glove.json)

```powershell
& $exp -B -m analysis.cluster_entities
```

CoNLL-2003 固定 300 条中，M0、M1、M2 以实体范围和类型做多数投票，得到 **78.94% F1**，M2 单独为 **81.04%**。两者差值为 **−2.11 个百分点**，配对 95% 区间为 **−4.26 至 +0.31 个百分点**。报告还列出每对模型漏报、误报及总体错误的重叠。可以重复传 `--pred` 和 `--group` 来比较更多模型或投票组，所有答卷使用同一批样本和同一套标签。[集成脚本](eval/ensemble_ner.py) · [投票结果](results/m0_m2_ensemble.json)

```powershell
& $exp -B -m eval.ensemble_ner --gold data/processed/conll2003/test.jsonl --subset-ids data/subsets/conll2003/test_300.txt --pred M0=preds/conll2003/m0_baseline/0.jsonl --pred M1=preds/conll2003/m1_baseline/0.jsonl --pred M2=preds/conll2003/m2_baseline/0.jsonl --group classical=M0,M1,M2 --output results/m0_m2_ensemble.json
```

## 报告图表和原句预测

从准备好的 JSONL 计算六套数据的探索统计。表格和图可从已提交的结果 JSON 单独生成。A 的 CSV 也由这些 JSON 汇总，`source_ref` 可以定位到每个指标的原位置。[探索统计](results/data_exploration.json) · [A 的结果表](results/A.csv)

```powershell
& $exp -B -m analysis.data_exploration
& $exp -B -m analysis.make_tables
& $exp -B -m analysis.make_figures
& $exp -B -m eval.result_tables export-a
```

| 输出 | 内容 |
|---|---|
| [F1](report/figures/f1.pdf) · [T1](report/tables/t1.tex) | 六套数据的类别分布、大小写、两两名称重叠、样本量和长度 |
| [F3a](report/figures/f3a.pdf) · [T2](report/tables/t2.tex) | M2 特征消融和 A 的调参记录 |
| [F5](report/figures/f5.pdf) | 实体熟悉度、替换、年代迁移和额外特征 |
| [F6b](report/figures/f6b.pdf) · [F7](report/figures/f7.pdf) | 实体重叠与迁移掉分、G2 消融及扰动对照 |
| [T4-A](report/tables/t4_a.tex) | 固定 300 条上的三种单模型和经典方法投票 |

收到 B、C 的答卷及结果 CSV 后，可以使用同一投票脚本跑跨范式组合。C 的两版提示词各需一对人工复核表，再用 [双人一致性工具](README_BC.md#3-自动错误分类与人工复核)计算 κ。

原句提取入口支持 `m0`、`m1`、`m2`。下面的命令会返回实体原文、类别和在输入文本中的字符范围，`start` 从 0 开始，`end` 指向实体结束后的第一个字符。[预测入口](scripts/predict.py)

```powershell
& $exp -B -m scripts.predict --model m2 --text "John Smith visited London."
```
