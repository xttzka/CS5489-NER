# CS5489 多领域命名实体识别

这个仓库整理了 A 的数据准备、M0–M2 经典基线、M2 分析实验，以及供 B、C 共用的评分和错误分析脚本。命令中的路径从项目根目录算起。汇总 JSON 记录分数、输入和答卷的位置。完整数据、逐条答卷和人工检查表由下列命令在各自电脑上生成。

## 1. 安装环境

实测环境为 Windows x64、Python 3.11.16，经典模型在 CPU 上运行。实验和 POS 标注分别使用两个虚拟环境，以复用原实验的依赖版本。两份依赖清单分别是 [requirements.txt](requirements.txt) 和 [requirements-pos.txt](requirements-pos.txt)。先在项目根目录打开 PowerShell。

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
py -3.11 -m venv .venv-pos
& ./.venv-pos/Scripts/python.exe -m pip install -r requirements-pos.txt
$exp = "./.venv/Scripts/python.exe"
$pos = "./.venv-pos/Scripts/python.exe"
```

每次新开 PowerShell 时重新设置 `$exp` 和 `$pos`。已有 3.11 解释器时也可以用它执行 `python -m venv`。macOS 或 Linux 上用 `python3.11 -m venv` 创建环境，解释器分别在 `.venv/bin/python` 和 `.venv-pos/bin/python`。手头已有 `data/processed_pos/` 时，评分和模型实验只用第一套环境。

在已经跑过 A 的实验的原项目目录里，可以直接设置 `$exp = "./.venv-m2/Scripts/python.exe"` 和 `$pos = "./.venv-pos/Scripts/python.exe"`，沿用本机原有环境。

## 2. 下载、转换并准备六套数据

CoNLL-2003 的转换脚本读 ZIP 内的 `train.txt`、`valid.txt` 和 `test.txt`，并保留原文件中的文档边界。原下载地址来自数据集加载脚本。[CoNLL-2003 加载脚本](https://huggingface.co/datasets/eriktks/conll2003/blob/main/conll2003.py#L152)

```powershell
New-Item -ItemType Directory -Force data/raw/conll2003 | Out-Null
if (-not (Test-Path data/raw/conll2003/conll2003.zip)) {
    Invoke-WebRequest -Uri 'https://data.deepai.org/conll2003.zip' -OutFile data/raw/conll2003/conll2003.zip
}
Get-FileHash data/raw/conll2003/conll2003.zip -Algorithm SHA256
& $exp -B -m data.prepare_conll2003
& $exp -B -m data.prepare_external_ner --download
& $exp -B -m data.prepare_external_ner
```

现有 CoNLL-2003 ZIP 的 SHA-256 是 `96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3`。若下载地址不可达，从组内资料取得相同 ZIP 后再转换。其余五套由 `--download` 下载并校验，下一条命令才做转换。两个步骤都支持 `--dataset conllpp` 等单数据集选项。

| 数据集目录 | 训练／验证／测试条数 | 原始文件 | 来源 |
|---|---:|---|---|
| `conll2003` | 14,041／3,250／3,453 | `data/raw/conll2003/conll2003.zip` | [加载脚本](https://huggingface.co/datasets/eriktks/conll2003/blob/main/conll2003.py#L152) |
| `wnut17` | 3,394／1,009／1,287 | `data/raw/wnut17/` 下的 `wnut17train.conll`、`emerging.dev.conll`、`emerging.test.annotated` | [WNUT-17 数据仓库](https://github.com/leondz/emerging_entities_17) |
| `mit_restaurant` | 6,894／766／1,521 | `data/raw/mit_restaurant/` 下的 `restauranttrain.bio`、`restauranttest.bio` | [MIT 下载页](https://sls.csail.mit.edu/downloads/restaurant/) |
| `conllpp` | 测试 1,840 | `data/raw/conllpp/conllpp.txt` | [CoNLL++ 数据仓库](https://github.com/ShuhengL/acl2023_conllpp) |
| `wikigold` | 测试 1,696 | `data/raw/wikigold/wikigold.conll.txt` | [WikiGold 数据目录](https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold) |
| `btc` | 测试 2,000 | `data/raw/btc/f.json` | [Broad Twitter Corpus 数据仓库](https://github.com/GateNLP/broad_twitter_corpus) |

MIT Restaurant 的 766 条验证数据由原训练文件按项目预处理规则划出。数据清单、下载地址和校验值见 [prepare_external_ner.py](data/prepare_external_ner.py)。

```powershell
& $exp -B -m data.make_eval_subsets
& $exp -B -m data.make_entity_index
& $pos -B -m data.add_pos_tags
& $exp -B -m data.make_entity_replacement
& $exp -B -m data.make_perturbation_sets
```

| 文件 | 作用及运行方式 | 输出位置 |
|---|---|---|
| [prepare_conll2003.py](data/prepare_conll2003.py) | `python -m data.prepare_conll2003`。将原始 ZIP 转成 JSONL，保留文档号、词性和短语标签。可用 `--archive`、`--output-dir` 改路径。 | `data/processed/conll2003/{train,validation,test}.jsonl` |
| [prepare_external_ner.py](data/prepare_external_ner.py) | `python -m data.prepare_external_ner --download` 下载。去掉 `--download` 后转换其他五套数据。 | `data/raw/<数据集>/`、`data/processed/<数据集>/` |
| [make_eval_subsets.py](data/make_eval_subsets.py) | `python -m data.make_eval_subsets`。用种子 42 生成或核对每套测试集的固定 300 个样本 ID。 | [data/subsets/](data/subsets/) 下各数据集的 `test_300.txt` |
| [make_entity_index.py](data/make_entity_index.py) | `python -m data.make_entity_index`。记录测试实体的范围、类别和见过／没见过。查找时将完整实体名忽略大小写，与 CoNLL-2003 训练实体比较。 | `data/derived/entity_index/<数据集>/test_v1.jsonl` |
| [add_pos_tags.py](data/add_pos_tags.py) | 用 POS 环境执行 `python -m data.add_pos_tags`。在原分词上补 spaCy 词性。可用 `--dataset` 只处理一套。 | `data/processed_pos/<数据集>/<划分>.jsonl` |
| [make_entity_replacement.py](data/make_entity_replacement.py) | `python -m data.make_entity_replacement`。读取统一 POS 的 CoNLL-2003 和 CoNLL++ 测试数据，生成 500 对 v2 实体替换样本。 | `data/derived/entity_replacement/conll2003/` 下的 `original_v2.jsonl`、`replaced_v2.jsonl`、`replacements_v2.jsonl` |
| [make_perturbation_sets.py](data/make_perturbation_sets.py) | `python -m data.make_perturbation_sets`。用统一 POS 的 CoNLL-2003 测试集生成全小写、5% 英文字母替换、去标点三个版本。 | `data/derived/perturbations/conll2003/` 下的 `lowercase_v1.jsonl`、`spelling_noise_v1.jsonl`、`no_punctuation_v1.jsonl`、`changes_v1.jsonl`、`manifest_v1.json` |
| [ner_label_mapping.json](data/ner_label_mapping.json) | 评分器读取的标签映射。CoNLL-2003 与 CoNLL++ 的年代对照用四类，共有标签迁移评测用 `PER`、`LOC`、`ORG`。 | 该 JSON 文件本身 |

## 3. 运行正式基线

M0 用训练集实体词典做精确匹配。M1 用逻辑回归逐词预测。M2 用 CRF 联合预测整句标签。CoNLL-2003 的多轮 `c1`、`c2` 调参已经归到 `tune_crf`。最早的五折与补充候选报告用于复用已有分数，最终选参以 [m2_cv_grid.json](results/m2_cv_grid.json) 为准。

```powershell
& $exp -B -m models.classic.tune_crf --jobs 3
& $exp -B -m models.classic.m2_native_cv --jobs 3
& $exp -B -m models.classic.run_baselines --jobs 1
```

CoNLL-2003 以文档为单位做五折，按训练集五折平均 F1 选参。最终 `c1=0.0001`、`c2=0.1`，五折平均实体 F1 为 **87.7603%**，全量重训后的验证集 F1 为 **88.8190%**。克隆后，`tune_crf` 可用已提交的调参报告复用 20 组网格分数，并重建缺少的验证答卷。如果只想运行正式基线，且两份选参报告及其对应训练数据已经齐备，可直接运行 `run_baselines`。

| 文件 | 作用及运行方式 | 报告和答卷 |
|---|---|---|
| [features.py](models/classic/features.py) | M1、M2 和消融脚本导入的共用特征代码。 | 通过调用它的实验产生结果 |
| [m0_dictionary.py](models/classic/m0_dictionary.py) | `python -m models.classic.m0_dictionary`。单独生成 M0 验证集答卷。 | `preds/conll2003/m0/0.jsonl` |
| [m1_logistic.py](models/classic/m1_logistic.py) | `python -m models.classic.m1_logistic`。单独生成 M1 验证集答卷。 | `preds/conll2003/m1/0.jsonl` |
| [m2_crf.py](models/classic/m2_crf.py) | `python -m models.classic.m2_crf`。默认参数的 M2 验证集实验，也是从头调参时的参照。 | `preds/conll2003/m2/0.jsonl` |
| [tune_crf.py](models/classic/tune_crf.py) | `python -m models.classic.tune_crf --jobs 3`。CoNLL-2003 调参入口，补齐缺少的前序报告和答卷后运行最终网格。 | [m2_cv_grid.json](results/m2_cv_grid.json)、`preds/conll2003/m2_cv_grid/0.jsonl` |
| [m2_cross_validate.py](models/classic/m2_cross_validate.py) | `python -m models.classic.m2_cross_validate` 做最早的五折。加 `--extend` 做补充候选。两阶段均由 `tune_crf` 在需要时调用。 | [m2_cv.json](results/m2_cv.json)、[m2_cv_extra.json](results/m2_cv_extra.json)，及 `preds/conll2003/{m2_cv,m2_cv_extra}/0.jsonl` |
| [m2_grid_search.py](models/classic/m2_grid_search.py) | `python -m models.classic.m2_grid_search` 汇合前两阶段候选，完成最终网格。直接调用时须备齐前序报告及需要的验证答卷。 | 同 `tune_crf` 的最终报告与答卷 |
| [m2_native_cv.py](models/classic/m2_native_cv.py) | `python -m models.classic.m2_native_cv --jobs 3`。为 WNUT-17 和 MIT Restaurant 分别做各自训练集上的五折调参。 | [m2_native_cv.json](results/m2_native_cv.json) |
| [run_baselines.py](models/classic/run_baselines.py) | `python -m models.classic.run_baselines --jobs 1`。生成三套完整域内测试和六套固定 300 条的 M0–M2 基线，含实体级指标和 bootstrap。 | [m0_m2_baselines.json](results/m0_m2_baselines.json)、`preds/<数据集>/{m0,m1,m2}_baseline/0.jsonl` |
| [baseline_statistics.py](eval/baseline_statistics.py) | `run_baselines` 导入的实体级指标、置信区间和模型差值计算。 | 写入正式基线报告 |

## 4. 运行迁移、年代和鲁棒性分析

下列命令承接第 2、3 节。从统一词性迁移答卷开始，再跑读取这些答卷的分析。CoNLL-2003 域内正式基线使用原词性和短语标签。统一词性实验使用 spaCy 词性并关闭短语标签，分别以各自报告为准。

```powershell
& $exp -B -m models.classic.m2_pos_transfer
& $exp -B -m eval.score_m2_familiarity
& $exp -B -m models.classic.run_temporal_baselines
& $exp -B -m eval.analyze_transfer_recall
& $exp -B -m models.classic.m2_transfer_matrix
& $exp -B -m eval.analyze_m2_transfer_overlap
& $exp -B -m models.classic.m2_g2_transfer
& $exp -B -m models.classic.m2_feature_ablation --jobs 3
& $exp -B -m models.classic.m2_pos_validation
& $exp -B -m models.classic.m2_pos_casing
& $exp -B -m models.classic.m2_pos_replacement --version v2
& $exp -B -m models.classic.m2_pos_perturbations
```

| 文件 | 输入和作用 | 报告和答卷 |
|---|---|---|
| [m2_pos_transfer.py](models/classic/m2_pos_transfer.py) | 用 CoNLL-2003 统一 POS 训练，评测五个完整测试集的共有三类。 | [m2_pos_transfer.json](results/m2_pos_transfer.json)、`preds/<数据集>/m2_pos_transfer/0.jsonl` |
| [score_m2_familiarity.py](eval/score_m2_familiarity.py) | 用迁移答卷和实体索引算见过／没见过的 F1。保留两套新闻的四类分数及五套迁移的三类分数。 | [m2_seen_unseen_f1.json](results/m2_seen_unseen_f1.json) |
| [run_temporal_baselines.py](models/classic/run_temporal_baselines.py) | 在两套完整新闻测试集上用同一个 M1 预测，生成 M0/M1 的完整答卷，并与已有 M2 统一 POS 答卷一起按见过／没见过评分。 | [m0_m2_temporal.json](results/m0_m2_temporal.json)、`preds/{conll2003,conllpp}/{m0,m1}_temporal/0.jsonl` |
| [analyze_transfer_recall.py](eval/analyze_transfer_recall.py) | 从迁移答卷拆出实体熟悉度、大小写和 BTC 账号等分组召回率。 | [m2_transfer_recall_groups.json](results/m2_transfer_recall_groups.json) |
| [m2_transfer_matrix.py](models/classic/m2_transfer_matrix.py) | 接上 CoNLL-2003 迁移参照，用共有三类比较 CoNLL-2003、WNUT-17 和两者合并训练。 | [m2_transfer_matrix.json](results/m2_transfer_matrix.json)、`preds/<数据集>/m2_pos_transfer_<训练来源>/0.jsonl` |
| [analyze_m2_transfer_overlap.py](eval/analyze_m2_transfer_overlap.py) | 读取迁移矩阵，计算训练与测试实体名称重叠和迁移分数下降的关系。 | [m2_transfer_overlap.json](results/m2_transfer_overlap.json) |
| [m2_g2_transfer.py](models/classic/m2_g2_transfer.py) | 用统一 POS 比较移除 G2 书写形式特征前后的域内结果及 CoNLL-2003 到 BTC 的迁移。 | [m2_g2_transfer.json](results/m2_g2_transfer.json)、`preds/<数据集>/m2_g2_<训练源>_<条件>/0.jsonl`，部分参照复用迁移答卷 |
| [m2_feature_ablation.py](models/classic/m2_feature_ablation.py) | 读取最终网格报告，在相同五折上分别移除 G1–G5 特征组。 | [m2_feature_ablation.json](results/m2_feature_ablation.json) |
| [m2_pos_validation.py](models/classic/m2_pos_validation.py) | 用最终网格的验证集答卷作参照，检查统一词性的三类验证集表现。 | [m2_pos_validation.json](results/m2_pos_validation.json)、`preds/conll2003/m2_pos/0.jsonl` |
| [m2_pos_casing.py](models/classic/m2_pos_casing.py) | 比较同一模型在原始验证句和全小写验证句上的表现。 | [m2_pos_casing.json](results/m2_pos_casing.json)、`preds/conll2003/m2_pos_casing_<条件>/0.jsonl` |
| [m2_pos_replacement.py](models/classic/m2_pos_replacement.py) | 读取 v2 配对数据，比较替换前后四类 F1，并逐个核对被替换的实体。 | `results/m2_pos_replacement_v2.json`、`preds/conll2003/m2_pos_replacement_v2_<条件>/0.jsonl` |
| [m2_pos_perturbations.py](models/classic/m2_pos_perturbations.py) | 核对扰动清单的哈希，比较原始测试和三种扰动的四类 F1、召回率。 | [m2_pos_perturbations.json](results/m2_pos_perturbations.json)、`preds/conll2003/m2_pos_perturbations_<条件>/0.jsonl` |

500 对 v2 样本中，M2 圈对了 470 个原实体和 339 个替换实体。逐条结果含实体原文，保存在本机的替换报告中。

### 地名词典和 GloVe 聚类

额外特征实验还读取 `m2_pos_transfer`、`score_m2_familiarity` 的结果及答卷。将三个资源放在下列位置后，执行 `& $exp -B -m models.classic.m2_extra_features`。

| 资源 | 放置位置 | 来源 |
|---|---|---|
| 城市列表压缩包 | `data/external/gazetteers/cities15000.zip` | [GeoNames 下载目录](https://download.geonames.org/export/dump/) |
| 国家信息 | `data/external/gazetteers/countryInfo.txt` | [GeoNames 下载目录](https://download.geonames.org/export/dump/) |
| GloVe 100 维词向量 | 从 `glove.6B.zip` 解出 `glove.6B.100d.txt`，放到 `data/external/glove/` | [Stanford GloVe 文件](https://huggingface.co/stanfordnlp/glove/blob/main/glove.6B.zip) |

[m2_extra_features.py](models/classic/m2_extra_features.py) 比较词典特征及 100、500、1,000 个 GloVe 簇。结果写到 [m2_extra_features.json](results/m2_extra_features.json)，答卷位于 `preds/{conll2003,conllpp}/m2_extra_<条件>/0.jsonl`。脚本校验每个资源的 SHA-256。GeoNames 地址会滚动更新，复跑原实验使用组内保留的 2026-11-10 文件快照。资源校验值见结果文件的 `resources` 字段。

## 5. B、C 的共用接口

人工答案每行含 `sample_id`、`tokens`、`tags` 等字段。模型答卷每行含相同的 `sample_id` 和逐词 BIO `tags`，沿用人工答案的分词顺序。完整测试要覆盖全部样本。固定 300 条实验只写清单里的 ID，并用 `--subset-ids` 核对覆盖范围。

```json
{"sample_id":"conll2003:train:90","tags":["O","O","O","O","B-LOC","O","B-ORG","O","O"]}
```

下面第一条命令用仓库自带的一句答卷演示评分，预期 F1 为 **0.5**。其余命令用前面生成的真实答卷演示不同任务。B、C 可把 `--pred` 换成自己的答卷路径。

```powershell
& $exp -B -m eval.score_ner --gold data/processed/conll2003/train.jsonl --pred preds/conll2003/demo/0.jsonl --allow-subset
& $exp -B -m eval.score_ner --gold data/processed/conllpp/test.jsonl --pred preds/conllpp/m1_baseline/0.jsonl --subset-ids data/subsets/conllpp/test_300.txt
& $exp -B -m eval.score_transfer --gold data/processed/wnut17/test.jsonl --pred preds/wnut17/m2_pos_transfer/0.jsonl --target-dataset wnut17 --prediction-schema conll2003
& $exp -B -m eval.score_familiarity --dataset conllpp --pred preds/conllpp/m1_temporal/0.jsonl
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/m2_baseline/0.jsonl --output-dir results/error_analysis/conll2003_m2_rerun --sample-size 60 --seed 42
```

| 文件 | 评分方式 | 输出位置 |
|---|---|---|
| [score_ner.py](eval/score_ner.py) | `python -m eval.score_ner`。范围与类别均对才算实体正确。固定子集传 `--subset-ids`。单句演示用 `--allow-subset`。 | 终端 JSON |
| [score_transfer.py](eval/score_transfer.py) | `python -m eval.score_transfer`。按目标数据集和预测的原标签体系映射共有三类。 | 终端 JSON |
| [score_familiarity.py](eval/score_familiarity.py) | `python -m eval.score_familiarity`。输入 CoNLL-2003 或 CoNLL++ 的完整四类答卷，计算总体及两组熟悉度分数。 | 终端 JSON |
| [analyze_ner_errors.py](eval/analyze_ner_errors.py) | `python -m eval.analyze_ner_errors`。按样本 ID 对照同标签体系的答卷，列出漏圈、误圈、范围错、类别错和两者皆错。从类别错和误圈中固定种子抽样。每次指定一个新的 `--output-dir`。 | 指定目录下的 `errors.jsonl`、`review.csv`、`summary.json`。示例为 [M2 错误汇总](results/error_analysis/conll2003_m2/summary.json) |


