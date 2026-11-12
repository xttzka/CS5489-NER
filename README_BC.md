# CS5489 NER 小组项目 · B、C 数据与评测交接

把本文件和已经克隆的项目一起交给能操作文件和终端的 AI，或在 AI 已打开项目时粘贴下方指令。首次使用时，先接受仓库邀请，再克隆项目。执行需要项目目录的读写权限、Python 3.11，以及下载依赖和原始数据所需的网络或组内文件副本。以下命令都从项目根目录运行。[组内 GitHub 仓库](https://github.com/xttzka/CS5489-NER) · [项目 README](README.md)

```powershell
git clone https://github.com/xttzka/CS5489-NER.git
cd CS5489-NER
```

## 给 AI 的任务指令

```text
请在我已克隆并打开的 CS5489 NER 项目中，按 README_BC.md 实际执行数据准备和检查。先确认 Python 3.11、项目根目录以及两套环境，再复用已有原始数据，下载缺失文件并校验。转换全部六套数据，核对各划分条数。验证六份固定测试集 300 条清单和 CoNLL-2003 验证集 300 条清单，生成见过／没见过实体索引、统一 POS 数据、500 对实体替换样本及三套扰动测试集。检查产物和目录，最后列出实际执行的命令、各划分条数、每份清单的 ID 数量以及输出路径。遇到网络或校验错误时，保留已核对的产物并说明缺失文件、来源和校验要求。

如果我提供了真实的 B 或 C 模型答卷、数据集、评测范围与标签体系，请核对 JSONL，执行相应评分，并用共享脚本生成自动错误分类与人工复核表。C 的两版提示词分别处理，按各自的候选池抽样。答卷尚未提供时，先完成所有数据准备，列出待输入的答卷路径。若需要核验脚本，可以用仓库的一句评分样例或本文的 M2 演示。把演示标为 A 的样例或 M2 结果，人工复核栏留给实际评阅人。项目数据和完整答卷保存在本地。
```

## 1. 环境与数据准备

Windows PowerShell 下使用两套虚拟环境。已有符合依赖清单的环境时，把 `$exp` 和 `$pos` 指向相应的 Python。A 的本地实验环境目录名是 `.venv-m2`，新克隆可以使用下面的 `.venv`。

```powershell
py -3.11 -m venv .venv
py -3.11 -m venv .venv-pos
$exp = "./.venv/Scripts/python.exe"
$pos = "./.venv-pos/Scripts/python.exe"
& $exp -m pip install -r requirements.txt
& $pos -m pip install -r requirements-pos.txt
```

Linux 或 macOS 用 `python3.11 -m venv .venv` 和 `python3.11 -m venv .venv-pos`。在 shell 中设置 `exp=./.venv/bin/python` 与 `pos=./.venv-pos/bin/python`，把 PowerShell 命令开头的 `& $exp`、`& $pos` 分别写为 `"$exp"`、`"$pos"`。两套依赖见 [实验环境](requirements.txt) 和 [POS 环境](requirements-pos.txt)。

### 下载和转换

下面的命令会下载缺失文件，核验版本，转换六套数据，并生成固定子集、实体索引、统一 POS、替换和扰动测试集。原始文件齐备时去掉 `--download`。CoNLL-2003 下载源不可达时，将组内保留的 ZIP 放在 `data/raw/conll2003/conll2003.zip` 后重试。预期 SHA-256 为 `96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3`。[准备入口](data/prepare.py) · [其余五套的来源与校验值](data/prepare_external_ner.py)

```powershell
& $exp -B -m data.prepare --all --download --pos-python $pos
```

在 Linux 或 macOS，设置前文的 `exp`、`pos` 后运行 `"$exp" -B -m data.prepare --all --download --pos-python "$pos"`。逐步复跑的命令见 [项目 README](README.md#2-下载转换并准备六套数据)。

| 数据集目录 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|
| `conll2003` | 14,041 | 3,250 | 3,453 |
| `wnut17` | 3,394 | 1,009 | 1,287 |
| `mit_restaurant` | 6,894 | 766 | 1,521 |
| `conllpp` | — | — | 1,840 |
| `wikigold` | — | — | 1,696 |
| `btc` | — | — | 2,000 |

正式划分写入 `data/processed/<dataset>/{train,validation,test}.jsonl`，仅生成表中有条数的文件。MIT Restaurant 将原始训练文件中按原顺序最后连续的 766 条划作验证，前 6,894 条用于训练。CoNLL++、WikiGold、BTC 的项目划分只有测试集。训练使用 `train`，调参和选模型使用 `validation` 或训练集交叉验证，配置确定后再对 `test` 评分。

### 固定子集、索引和派生测试集

```powershell
& $exp -B -m data.prepare --derived --pos-python $pos
```

| 产物 | 路径与核验 |
|---|---|
| 固定 300 条 | `data/subsets/<dataset>/test_300.txt`。每套测试集按 `sample_id` 排序后，用固定种子 42 抽取 300 条并保存排序后的 ID。脚本会核对仓库已有清单。实际评测按清单取题，完整测试使用全部测试 ID。 |
| C 的提示词验证 300 条 | [validation_300.txt](data/subsets/conll2003/validation_300.txt)。从 `data/processed/conll2003/validation.jsonl` 按相同规则和种子 42 抽取。C 用同一份验证清单比较提示词配置，选定配置后再按各数据集的 `test_300.txt` 做测试。 |
| 见过／没见过实体索引 | `data/derived/entity_index/<dataset>/test_v1.jsonl`。六套各有一份。每行一个标准实体，记录 `sample_id`、起止位置、原类别及 `seen_in_training`。以完整实体词序列忽略大小写，与 CoNLL-2003 训练集标注过的实体比较。 |
| 统一 POS | `data/processed_pos/<dataset>/<split>.jsonl`。spaCy 在原分词上补词性，样本 ID、词序和实体标签可以直接对应 `processed`。 |
| 实体替换 v2 | `data/derived/entity_replacement/conll2003/` 下的 `original_v2.jsonl`、`replaced_v2.jsonl` 和 `replacements_v2.jsonl`。500 对句子，PER、LOC、ORG 各 150 对，MISC 50 对。按同一个 `sample_id` 比较替换前后的答卷。 |
| 三套扰动 v1 | `data/derived/perturbations/conll2003/` 下的 `lowercase_v1.jsonl`、`spelling_noise_v1.jsonl` 和 `no_punctuation_v1.jsonl`，每套 3,453 条。 `manifest_v1.json` 与 `changes_v1.jsonl` 记录规则和改动。各版本用自己的 `tokens` 与 `tags` 评分，删标点后词位置会移动。 |

做固定 300 条推理时，先读取该数据集的 `test_300.txt`，再用每个 ID 到同一数据集的 `test.jsonl` 里找对应行。例如 CoNLL++ 用 `data/subsets/conllpp/test_300.txt` 配 `data/processed/conllpp/test.jsonl`。按清单顺序取出恰好 300 行，用 `sample_id` 记录答卷归属，把 `tokens` 交给模型，`tags` 留给评分。完整测试、实体替换和三套扰动分别从上表对应的文件读取。

A 的新版基线使用统一 spaCy POS 和验证集选出的 M1 配置。对照替换或扰动时，复用表中的 v2 配对样本和 v1 扰动样本，按相同的 ID 与 A 的三模型结果比较。[正式基线](results/m0_m2_baselines.json) · [替换与扰动结果](results/m0_m2_robustness.json) · [固定 300 条投票示例](results/m0_m2_ensemble.json)

## 2. B、C 答卷和评分

标准答案和答卷都用 UTF-8 JSONL，每行一个 JSON 对象。标准答案含 `sample_id`、`tokens`、`tags`、`dataset`、`split` 和可用的文档、词性字段。模型答卷只需对应的 `sample_id` 和 `tags`。例如仓库的一句评分演示中有这条答卷。 [演示文件](preds/conll2003/demo/0.jsonl)

```json
{"sample_id":"conll2003:train:90","tags":["O","O","O","O","B-LOC","O","B-ORG","O","O"]}
```

用各自模型生成真实答卷，放到 `preds/<dataset>/<method>/<seed>.jsonl`。每个原始词对应一个标签，标签数与 `tokens` 相同。 `B-` 是实体开头，`I-` 继续同类实体，`O` 表示实体外。完整评测要求答卷 ID 恰好覆盖整个测试集，固定 300 条要求恰好覆盖清单中的 ID。两种范围的答卷以及同一方法的不同提示词、派生版本使用不同方法目录名。评分以实体范围和类别同时正确为准。

- **同一套原生标签**使用 `eval.score_ner`。CoNLL-2003、CoNLL++、WikiGold 用 `PER/LOC/ORG/MISC`。WNUT-17 保留 `person`、`location`、`corporation`、`group`、`product`、`creative-work`。MIT Restaurant 使用 `Amenity`、`Cuisine`、`Dish`、`Hours`、`Location`、`Price`、`Rating`、`Restaurant_Name`。BTC 用 `PER/LOC/ORG`。
- **共有三类迁移**使用 `eval.score_transfer`。 `--target-dataset` 填标准答案所在数据集，`--prediction-schema` 填模型实际输出的标签体系。只输出 `PER/LOC/ORG` 时，可用 `conll2003` 对这三类做同名映射。WNUT-17 的 `corporation` 和 `group` 对应 `ORG`，`person` 对应 `PER`，`location` 对应 `LOC`。共有三类评测省略 CoNLL 的 `MISC` 和 WNUT 的 `product`、`creative-work`。MIT Restaurant 使用自己的标签。 [标签映射](data/ner_label_mapping.json)

以下命令以 B 的 CoNLL-2003 答卷路径为例。运行前把 `bert_full`、`bert_300` 和种子改成真实答卷的目录和文件名。评分命令输出 JSON，保存或交接时记上输入路径、方法、种子、评测范围和标签体系。

```powershell
& $exp -B -m eval.score_ner --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/bert_full/0.jsonl
& $exp -B -m eval.score_ner --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/bert_300/0.jsonl --subset-ids data/subsets/conll2003/test_300.txt
```

C 已有验证集答卷时，可将下例的 `llm_validation_300` 换成实际答卷所在目录。每种提示词配置都用同一份验证清单。

```powershell
& $exp -B -m eval.score_ner --gold data/processed/conll2003/validation.jsonl --pred preds/conll2003/llm_validation_300/0.jsonl --subset-ids data/subsets/conll2003/validation_300.txt
```

按 CoNLL-2003 标签体系输出的 WNUT-17 答卷，可以用映射评分评估固定 300 条。

```powershell
& $exp -B -m eval.score_transfer --gold data/processed/wnut17/test.jsonl --pred preds/wnut17/bert_transfer_300/0.jsonl --target-dataset wnut17 --prediction-schema conll2003 --subset-ids data/subsets/wnut17/test_300.txt
```

## 3. 自动错误分类与人工复核

使用 [共享错误分析脚本](eval/analyze_ner_errors.py)。标准答案和答卷需使用相同标签体系。共有三类实验如需做错误分析，可先按映射将双方投影到 `PER/LOC/ORG`，保留标准答案的 `tokens` 和双方的 ID、词位置。脚本先找到完全正确的实体，再对剩余重叠实体一对一匹配，记录漏圈 `missed`、误圈 `spurious`、范围错 `boundary`、类别错 `type`、范围和类别都错 `boundary_and_type`。位置从 0 开始，结束位置不含在实体范围内。

### 完整测试和固定 300 条

```powershell
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/bert_full/0.jsonl --output-dir results/error_analysis/b/bert_conll2003_full --sample-size 60 --seed 42
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/bert_300/0.jsonl --subset-ids data/subsets/conll2003/test_300.txt --output-dir results/error_analysis/b/bert_conll2003_300 --sample-size 60 --seed 42
```

### C 的两版提示词

C 可把同一提示条件下不同数据集的结果合成一个抽样池。按相同顺序提供每套的标准答案、真实答卷和固定清单。不同数据集的 ID 必须互不重复。下面演示 CoNLL-2003 和 WNUT-17 均采用原生标签时，对两版提示分别抽样。若只做一套数据，三个参数各保留一个对应文件。每个输出目录须是新目录。

```powershell
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl data/processed/wnut17/test.jsonl --pred preds/conll2003/llm_plain/0.jsonl preds/wnut17/llm_plain/0.jsonl --subset-ids data/subsets/conll2003/test_300.txt data/subsets/wnut17/test_300.txt --output-dir results/error_analysis/c/plain --sample-size 60 --seed 42
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl data/processed/wnut17/test.jsonl --pred preds/conll2003/llm_guideline/0.jsonl preds/wnut17/llm_guideline/0.jsonl --subset-ids data/subsets/conll2003/test_300.txt data/subsets/wnut17/test_300.txt --output-dir results/error_analysis/c/guideline --sample-size 60 --seed 42
```

每个 `--output-dir` 下会生成 `errors.jsonl`、`summary.json`、`review.csv`。默认从类别错和误圈中用种子 42 抽 60 条。若合格条数少于 60，脚本取全部并在汇总中记录实际条数。两位评阅人分别复制同一份复核表，保留 `error_id`，独立填写 `judgment` 和 `notes`。人工判断使用 `guideline_difference`（标注规范差异）、`gold_error`（标准答案有误）或 `other`（其他）。LLM 输出了原句里没有的实体时，须对照 C 另存的原始模型输出核查。BIO 答卷只记录能落到原句词位置的标签。

### 可选演示

一句演示答卷的预期 F1 是 0.5。要核验自动错误分类，可用 A 的 CoNLL-2003 正式 M2 答卷。若缺少该答卷，完成数据准备后执行下面的正式基线命令。已具备相应选参报告时，可直接运行第三条。 [正式基线入口](README.md#3-运行正式基线)

```powershell
& $exp -B -m models.classic.tune_crf --jobs 3
& $exp -B -m models.classic.m2_native_cv --jobs 3
& $exp -B -m models.classic.run_baselines --jobs 1
```

```powershell
& $exp -B -m eval.score_ner --gold data/processed/conll2003/train.jsonl --pred preds/conll2003/demo/0.jsonl --allow-subset
& $exp -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/m2_baseline/0.jsonl --output-dir results/error_analysis/bc_m2_demo --sample-size 60 --seed 42
```

对同一版提示词，两名评阅人在各自的 `review.csv` 中填写 `judgment`，使用 `guideline_difference`（规范不一致）、`gold_error`（金标错误）或 `other`（其他）。两份都填完后用下列命令算 Cohen’s κ 和每位评阅人的规范差异占比。另一版提示词使用自己的一对复核表及不同的输出文件。

```powershell
& $exp -B -m eval.analyze_ner_errors agreement --review-a output/review_a.csv --review-b output/review_b.csv --agreement-output output/agreement.json
```

## 4. 共用结果表和计时交接

A 的已测结果见 [results/A.csv](results/A.csv)。在项目根目录按成员选择一条命令创建自己的 CSV。脚本会在文件已存在时提示，已有结果可以继续追加。

```powershell
& $exp -B -m eval.result_tables template --output results/B.csv
& $exp -B -m eval.result_tables template --output results/C.csv
```

每行写一个数值。例如固定 300 条上的 BERT F1 可以填 `member=B`、`experiment=baseline`、`method=BERT`、`condition=standard`、`dataset=conll2003`、`split=test`、`scope=test_300`、`label_space=LOC+MISC+ORG+PER`、`metric=f1`、`unit=ratio`。标签体系按类别名排序后用 `+` 连接。同时填训练来源、种子、实际分数和来源路径。总体用 `entity_type=all`、`entity_group=all`，类别或熟悉度分组填相应名称。常用范围是 `full`、`test_300`、`validation_300`、`cv_fold` 和 `cv_mean`。比例值写成 0 到 1 之间的小数，百分点评分差用 `percentage_point`。 `config_json` 使用 JSON 对象，`source_ref` 填产生该数值的结果文件或其中的具体位置。

计时结果用 `experiment=timing`。分别记录 `fit_seconds`、`inference_sentences_per_second`、`gpu_vram_mb`（有 GPU 实测时）和 `parameter_count`。在 `config_json` 记录硬件、设备、批大小、精度、预热与重复次数，以及推理计时是否含分词、解析、文件写入等。把 F1 和速度放在一起比较时，对齐数据集、样本范围、标签体系和计时边界，并注明不同设备的影响。A 的 CPU 结果和测量范围可在 [计时报告](results/m0_m2_timing.json) 查到。

三份结果到齐后，可统一核对表头、数值和重复项，并合成一份表。

```powershell
& $exp -B -m eval.result_tables merge --input results/A.csv results/B.csv results/C.csv --output results/team_summary.csv
& $exp -B -m analysis.make_figures --team-results results/team_summary.csv --scope full
```

成本图按数据集、成员、方法、训练来源和标签体系匹配 `condition=standard` 的总体 `baseline/f1` 和 `timing/inference_sentences_per_second`。同一方法有多条种子记录时，图上使用平均 F1 和中位吞吐。固定 300 条的对照另传 `--scope test_300`，两项指标也都填 `test_300`。同一张图的数据使用一致的标签体系，记录设备和计时边界以便比较。

## 5. 交给组内的文件

- 报出实际使用的 `data/processed/`、`data/processed_pos/`、`data/subsets/`、`data/derived/` 文件及条数。派生评测的原句、替换和扰动答卷应分别记录对应文件。
- 交接真实答卷 `preds/<dataset>/<method>/<seed>.jsonl`、对应评分 JSON 及成员 CSV，注明标签体系、完整测试或固定 300 条，以及模型或提示词版本。错误分析交接相应目录的 `summary.json`，完整错误与人工复核表通过组内文件共享。
- B 的未微调、微调 BERT 实体向量各用一个 JSONL。每条写 `sample_id`、`start`、`end`、`vector`，可附 `entity_type`。起点按原始词元从 0 计，终点为右开，`vector` 是实体内子词向量的平均。A 按相同实体范围对齐两份向量与 GloVe。[聚类脚本](analysis/cluster_entities.py)
- 生成数据或真实答卷尚有缺项时，列出具体路径和当前缺失的输入，方便下一位接着执行。
