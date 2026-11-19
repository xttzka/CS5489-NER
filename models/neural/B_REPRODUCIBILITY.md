# Role B 实验复现与交接说明

本说明记录角色 B 神经模型实验的输入、环境、执行顺序和待完成项。当前本地的训练、验证集调参和 v2 的 162 项正式评测已完成，结果与报告的整理入口分别为 `results/b/` 和 `report/b/`。历史原结果、日志与冻结清单保留在原位置；结果应以完成日志、验证集选择证据、冻结清单和评分产物为准，不要把 smoke test 或验证集结果当作测试结果。所有相对路径从仓库根目录开始。运行重建命令前必须检查队列、输出路径和已有文件，避免并行占用设备或覆盖产物。

## 1. 硬件、解释器和环境

- 当前机器为 Apple M4、arm64、16 GB 统一内存；主实验环境为 `/opt/anaconda3/envs/cs5489/bin/python`，Python 3.11.18。核心版本：PyTorch 2.5.1、Transformers 4.46.3、NumPy 1.26.4、pytorch-crf 0.7.2、tokenizers 0.20.3、huggingface_hub 0.36.2。
- 在本次自动化会话中：受限沙箱内 `torch.backends.mps.is_available()` 返回 false，已获授权的沙箱外训练返回 true。此现象是该会话的设备访问限制，不能据此断言机器不支持 MPS，也不能推断另一台机器的权限状态。设备检查要在实际运行训练的同一环境下做，队列会拒绝静默转 CPU：

  ```sh
  /opt/anaconda3/envs/cs5489/bin/python -c 'import torch; print(torch.__version__, torch.backends.mps.is_available())'
  ```

- 在另一台计算机可新建 Python 3.11 环境并安装上述核心版本；平台和依赖解析可能影响重现。根目录 `requirements.txt` 是 A 的经典方法和绘图环境，并非 B 的 PyTorch 环境。完整当前环境应在实际机器导出并随实验保存：

  ```sh
  /opt/anaconda3/envs/cs5489/bin/python -m pip freeze > output/b_neural_pip_freeze.txt
  .venv-b-classic/bin/python -m pip freeze > output/b_classic_pip_freeze.txt
  .venv-b-pos/bin/python -m pip freeze > output/b_pos_pip_freeze.txt
  ```

  这些命令会写出新文件；先确认没有同名重要文件。`.venv-b-classic` 使用 `requirements.txt`，包括 sklearn 及绘图；`.venv-b-pos` 使用 `requirements-pos.txt`，包括 spaCy 3.8.18 和 en_core_web_sm 3.8.0。它们与神经模型环境隔离。
- `bert-base-cased` 使用本地 Hugging Face 缓存。当前 `refs/main` 指向 `cd5ef92a9fb2f889e972770a36d4ed042daf221e`；复现时应保留模型快照和检查其哈希，不应仅假定未来的 `main` 不变。正式队列按离线模式运行，预训练模型需提前缓存。

## 2. 数据与外部资源

数据来源、原始文件路径和各自校验见根目录 README 及数据准备脚本。六套统一数据位于 `data/processed/`；三套有训练集的数据还包含各自验证集。`data/subsets/` 的固定 300 句 ID、`data/processed_pos/`、`data/derived/` 中的索引、替换及扰动数据都需要与团队接口版本一致。实体替换和扰动会读取测试源数据以构建固定输入；构建由数据负责人或其批准的统一脚本完成，**这些数据和测试分数不可参与模型选择或调参**。

完整数据准备入口参见根目录 README；可在已准备好两个隔离环境后执行：

```sh
.venv-b-classic/bin/python -B -m data.prepare --all --download --pos-python .venv-b-pos/bin/python
```

下载需要网络及相应许可，重建已有派生物之前先检查脚本行为和团队提供的版本，不覆盖已冻结文件。MIT Restaurant 验证集按项目规则从训练数据划分。由于训练与评测高度依赖固定 ID 和版本，应保存数据及清单 SHA-256。

BiLSTM-CRF 另需要 `data/external/glove/glove.6B.100d.txt`，其 SHA-256 应为 `be4367dd257eb945217234f16307c5c74236b648a222cc0b4ffd0dda6a3350b6`。它与数据集不同，是外部预训练词向量；不要将文件纳入 Git。验证方法：

```sh
shasum -a 256 data/external/glove/glove.6B.100d.txt
```

蒸馏实验还需要 C 提供的 MIT 训练句伪标签文件 `preds/mit_restaurant/qwen7b_k0_plain_train_2000_v1/0.jsonl`；脚本会对句子 ID 和 token 做对齐检查。缺少这份输入时不得伪造标签或把实验标为完成。

## 3. 独立复现的顺序

1. 检查代码版本、数据清单、原始资源哈希、环境和 GPU。本次实验开始前的 Git 基线为 `877d4e5e96e3e0e5003a10e71843181351119f8c`，**该历史 commit 本身不包含之后新增的角色 B 文件**；应使用实际交付的 B 代码版本，并保存其快照及校验和。
2. 先只用 train 和 validation 训练域内候选、学习曲线、三类迁移源、蒸馏对照与四种表示。运行队列之前，使用以下只读命令检查清单：

   ```sh
   conda activate cs5489
   python -B -m models.neural.run_b_training --group all --dry-run
   python -B -m models.neural.run_b_representations --device mps --dry-run
   ```

   `run_b_training` 是针对本次已有实验**补缺**的队列，并不是空仓库的一键完整训练：CoNLL 全量三种子、既有域内 seed 0、CoNLL 学习曲线等历史基线也必须先按相同配置复现或从可信记录恢复。需要保存每次对应的完整输出目录和训练日志；只存在最佳 checkpoint 不能证明所有计划 epoch 已完成。各模型 CLI 的 `--help` 和日志中保存的具体命令可用于逐项复现。不要从旧结果名推测训练超参数。
3. 复现时需完成原计划 §4 各模型、各领域的 dev 超参数比较、获胜配置的三种子训练及源域迁移配置复核。当前本地已完成，候选、平局与选择规则记录在 `results/b/all_models_experiment_protocol.json`；候选改变多个参数时不可把提升归因给单个参数。搜索结果在被忽略的 `output/b_tuning/selection.json`，不能仅因代码文件存在就视为搜索完成。学习曲线和蒸馏保留先前的固定设置，要注明与调优域内模型的区别。
4. 等全部训练和验证选择通过校验后固定配置、三种子、checkpoint 与数据/代码哈希，生成不可变的冻结清单；仅此后生成预先规定的测试预测与分数。原仓库已有 A/C 测试结果，因此不能声称项目整体从未暴露过测试信息。`models.neural.finish_b --run --wait-until-ready` 会等待训练并核查搜索证据；不要把 `--allow-partial` 当作正式实验完成。
5. 评分须使用项目实体级精确匹配、固定子集及一致标签映射；报告总体和逐类型 P/R/F1、句子级 1,000 次配对 bootstrap 区间，并将模型随机种子均值±样本标准差与 seed 0 的 bootstrap 区间分开说明。六方法、三领域及六个 300 子集的表，以及 F2/F3/F4/F6、F8 交接数据，必须从已核验结果生成。F4 的 MISC 各簇仍需基于固定抽样上下文实际检查；不得把生成审阅表等同于完成解释。

## 4. 运行监控、故障恢复与产物

训练和表示队列共用 `output/b_training_queue/queue.lock`，应串行使用 GPU。每项训练保留独立配置、验证预测、结果和日志，队列 journal 在 `output/b_training_queue/`。完整运行日志中的 epoch 和结尾状态、配置与数据哈希均是复现证据。失败后先看相应日志及磁盘空间；已保存产物和日志不得删除、覆盖或仅按 checkpoint 认定完成，按队列的恢复检查生成新的 attempt。`--dry-run` 先核查将跳过或重跑哪些任务。测试阶段也应保留冻结清单、task 级别元数据、输入和模型哈希、预测与评分；发现错误时保留旧产物并明确记录修订，不能根据测试结果重新调参。

本地的主模型原句预测入口 `models.neural.predict_primary_b` 已依据冻结清单选择 M4，并已用非测试原句验证，例如：

```sh
python -B -m models.neural.predict_primary_b \
  --device cpu \
  --text 'Barack Obama visited New York.'
```

该入口默认依赖被忽略的 `output/b_finish_v2/frozen.json`、模型及验证证据等，并校验源码和模型文件。原冻结清单包含这台 Mac 的绝对路径：仅克隆仓库，或在另一台计算机原样复制清单，都不能保证直接运行；应保留原件，另建可验证的路径映射或适配版本，不能静默修改原清单。输出字符区间以 Python 字符索引表示，右端不包含。计划书指定的共享 `scripts/predict.py` 当前仍只支持 M0–M2、默认 M2；是否与主模型入口整合应由团队协调。

`.gitignore` 排除了 `output/`、原始/处理/派生数据、绝大多数 `preds/`、模型权重和虚拟环境。仅提交或推送代码、`results/B.csv` 不会自动包含 checkpoint、完整预测或训练日志。交接应另行提供允许共享的原件：冻结清单、选择与训练日志、checkpoint/tokenizer、数据和固定 ID 的哈希、评分结果及代码/环境版本。传输渠道由团队另定，并遵循各数据集和模型的分发许可。

## 5. 报告状态

B 的方法、§5 主结果、§6.1 表示分析与 §6.3 迁移讨论已形成英文草稿并编译独立报告，当前交付路径为 `report/b/role_b_report.pdf` 和 `report/b/sections/`；它不是整组最终报告。F4 的 MISC 结果已有助手辅助样例审阅，仍需独立人类复核；A/C 的章节、参考文献及团队视频也需要整合。C 的零样本结果和 A 的 CRF/重叠率分析由相应负责人提供，不应冒充 B 已完成。F8 的种子级表及共享限制另见 `B_HANDOFF.md`。

## 6. Tokenizer 对齐修复及 v2 重评

v1 正式评测在 BTC 的 `btc:test:834` 遇到独立字符 U+200E：快速 tokenizer 会将它清除，导致原词与 subword 数不一致。`output/b_alignment_audit_v1/alignment_audit.json` 在不读取测试标签的情况下检查了旧冻结清单所引用的 11 个唯一输入，记录每个输入的 SHA、全部受影响词及超长统计。v1 清单与已产生的结果必须原样保留。

v2 规则仅在某个原词没有产生任何 subword 时，在**模型输入**中使用模型自带 `[UNK]` 占位；原始词数、顺序、BIO 输出索引与数据文件不变。各次 BERT/Span 预测的 profile 记录替换次数、样本 ID、词索引与 Unicode 码点，冻结 BERT 表示和向量导出在日志记录。该推理修复发生在模型训练与旧版部分测试之后，不代表训练使用了修复代码，也不依据测试分数选择模型或版本。

v2 修订运行已经在 `output/b_finish_v2/` 完成。需要独立重建时须使用新的、尚不存在的目录，并传入旧清单及已保存的对齐审计；这不是空仓库一键复现，需预先获得完整训练产物和依赖。下例仅用于说明，不要覆盖现有 v2：

```sh
python -B -m models.neural.finish_b --run --wait-until-ready \
  --work-dir output/b_finish_reproduction_v1 \
  --amend-from output/b_finish_v1/frozen.json \
  --alignment-audit output/b_alignment_audit_v1/alignment_audit.json
```

新清单记载父清单 digest、修改原因及每个受影响文件的新旧 SHA；同一批冻结任务在全新评测目录从头执行，不能复用旧预测。`version_comparison/` 对 v1 中已经完成的每个任务逐 ID 比较 v1/v2 的预测序列，报告样本及词级差异，不依据分数挑选版本。模型权重、验证集选择和旧产物均不覆盖。`models.neural.predict_primary_b` 默认读取 `output/b_finish_v2/frozen.json`，可用 `--manifest` 明确指定其他版本；v1 推理需要在隔离副本中恢复其原代码，当前修复代码会因 SHA 校验而拒绝冒充 v1。
