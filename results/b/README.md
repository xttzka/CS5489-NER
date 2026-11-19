# 角色 B 的实验结果

这个目录整理现有 `results/B.csv` 与 `results/B_*.{csv,json,md}`；**根目录原文件均保留**。原计划及团队共享脚本仍读取 `results/B.csv`，不要因这里有副本就删除或改写根目录原件。这里没有覆盖 A 或 C 的文件。

## 按模型查看的结果

| 文件 | 模型 | 来源和范围 |
|---|---|---|
| `bert_results.csv` | M4 / BERT | 从 `results/B.csv` 精确筛选 `method=M4` 的所有指标行，包括学习曲线、迁移等条件；不是只有主结果。 |
| `bilstm_crf_results.csv` | M3 / BiLSTM-CRF | 同一原表精确筛选 `method=M3`，包含表示对照等条件。 |
| `span_bert_results.csv` | Span / BERT Span | 同一原表精确筛选 `method=Span` 的全部指标行。 |
| `all_models_results.csv` | 三个神经模型 | `results/B.csv` 的逐字节副本，保留所有原字段；权威兼容路径仍是 `results/B.csv`。 |

三个模型的表是原始指标明细的分组视图，包含 precision、recall、F1、各类指标及置信区间等；一行不是一次独立实验。不要把这三份分表再追加回总表，否则会重复统计。`source_ref` 和部分 JSON 用相对仓库根目录的路径指向 `output/`：代码库不含这些被 Git 忽略的模型、预测、数据和完整日志，队友需要另行获取并处理路径；不可直接修改已签名的冻结清单冒充原始证据。

## 学习曲线

- `bert_learning_curve_by_seed.csv`、`bert_learning_curve_summary.csv`、`bert_learning_curve_notes.md`：BERT 学习曲线，给 C；其中历史训练 ID 的部分记录只是按规则重建，不能证明与 C 的 LoRA 数据逐条相同。

## 统计、图和可追溯记录

- `bert_vs_span_boundary_test_seed0.json` / `bert_vs_span_boundary_validation_seed0.json`：BERT 与 Span 边界错误对照；严格区分测试与验证。
- `all_models_paired_test_seed0.json` / `all_models_paired_validation_seed0.json`：多模型成对分析，分别来自测试与验证。
- `glove_bert_f4_clustering_metrics.json`、`glove_bert_f4_redraw_manifest.json`：GloVe、原始 BERT、微调 BERT 的 F4 信息；`glove_bert_entity_vectors_manifest.json` 和 `glove_validation_entity_vectors_metadata.json` 是向量来源清单，并不包含向量本身。
- `classic_baselines_reference.csv` 和 `crf_a_f6b_overlap_reference.json`：A 的经典模型/CRF 参考，文件名明确区分，并非新训练的 B 模型。
- `all_models_experiment_protocol.json`：训练与评测方案。
- `all_models_alignment_version_comparison.json`：BTC 控制字符推理修复前后版本的对照。
- `all_models_original_delivery_manifest.json`：原始交付记录；保留其原有路径和哈希，路径描述的是当时的根目录文件，而非自动指向本目录。
- `file_index.json`：本次整理逐个文件的源路径、目标路径与 SHA-256，以及三个模型结果分表行数和哈希。

## 交接边界

本目录是代码库中的结果索引；模型权重、逐句预测、三种实体向量和冻结清单仍在被 `.gitignore` 忽略的 `output/`，应按团队约定另行交接。正式测试期间的 tokenizer 修复和 A/C 测试信息此前已存在的限制，参见协议、版本对照和 B 报告；不要声称完整实验过程严格只运行了一次测试。

`git_push_file_list.txt` 列出本次建议提交的全部具体路径：整个 B 神经模型代码目录、本结果目录、`report/b/`，以及团队接口所需的根目录 `results/B.csv`。它是整理时的静态清单；如果以后新增文件，应重新核查。旧 `results/B_*` 及 `report/b_handoff/` 等历史原件保留在本机，不必重复加入这次提交。
