# 角色 B：神经模型代码与交接入口

本目录包含角色 B 的训练、预测、评测、调参及画图脚本。当前本地已完成 M3 BiLSTM-CRF、M4 BERT 和 Span 三类模型的训练以及 v2 的 162 个正式评测条件；数字表示评测条件，不是 162 个独立模型。此次整理不改变历史训练、预测、冻结清单或已有模型文件。

## 从哪里找当前结果和报告

从仓库根目录查看：

- `results/b/README.md`：按 `bert`、`bilstm_crf`、`span_bert` 分类的结果索引、汇总、学习曲线和已知局限。
- `results/B.csv`：团队原先约定的根目录汇总表，与 `results/b/all_models_results.csv` 内容一致；不可将分类表再次追加到该表。
- `report/b/README.md`：报告、表格、图和视频讲稿索引。`report/b/role_b_report.pdf` 是 B 的独立报告草稿，不是全组最终报告。
- `models/neural/B_REPRODUCIBILITY.md`：已有实验的环境、数据依赖、复现顺序和 BTC 对齐修复说明。
- `models/neural/B_HANDOFF.md`：F8、模型文件和报告的团队交接说明。

`models/neural/B_methods_draft.tex`、`B_classic_methods_draft.tex`、`B_results_analysis_draft.tex`、`B_misc_cluster_review.md` 是历史交付源文件，保留当时内容和路径以供审计；`B_video_script.md` 是此前讲稿来源，仍引用旧图路径。新的便携报告与讲稿请用 `report/b/`。`models.neural.deliver_b` 是生成旧 `results/B_*` 与 `report/b_handoff/` 的历史交付工具，其既有目标存在时会拒绝覆盖；不要误将它当成新目录的重建命令。

## 代码范围

- `bilstm_crf.py`、`bert_baseline.py`、`bert_extended.py`、`span_bert.py`：三类模型与训练。
- `bilstm_representations.py`：四种输入表示对照。
- `predict_neural.py`、`predict_sentence_b.py`、`predict_primary_b.py`：模型和原句预测。`predict_primary_b` 在本 Mac 上依据冻结清单选择 M4，并已验证一条非测试原句。
- `run_b_training.py`、`run_b_tuning.py`、`run_b_representations.py`、`run_b_evaluation.py`、`finish_b.py`：已有实验的训练/评测流程；它们需要已准备的数据和部分历史训练产物，不是空仓库一键复现。
- `score_b.py`、`compare_boundary_b.py`、`additional_b_analysis.py`、`plot_*_b.py`、`export_*_entities*.py`：评分、边界、分析、图表与向量导出。
- 其余 `.py` 包括模型选择审计、数据汇总、旧输出兼容和交付；部分需读取现有团队 `data/`、`eval/`、`models/llm/common.py` 和 `analysis/`，并非本目录单独即可运行。

## 运行及共享边界

仓库忽略了 `output/`、原始与处理后数据、绝大多数逐句预测、GloVe 和模型权重。仅推送代码和结果表并不包含 checkpoint、tokenizer、三种向量、训练日志、`output/b_finish_v2/frozen.json` 或可完整重现的环境；须与 A、C 约定许可允许的文件交接方式。主模型是 `output/bert_tuning_conll2003_alternate_seed0/best_model/`，CoNLL 的 10% BERT 对照是 `output/bert_conll2003_10pct_seed0/best_model/`。

`results/B.csv` 的 `source_ref`、部分审计文件和签名冻结清单包含原 Mac 的绝对路径；在其他机器上不能保证直接解析。`predict_primary_b.py` 还校验清单、数据、验证证据、代码和 checkpoint 的哈希，移动后不能假定开箱即用。保留原始签名清单，再设计可验证的路径重映射或生成另一个审计版本；不要直接改写原清单。共享的 `scripts/predict.py` 仍只支持 M0–M2、默认 M2，与 B 主模型的统一入口尚待团队协商。

## 实验解释与未完成的团队工作

- 参数搜索只比较了每领域两组联合参数，不能将变化归因于单个参数；学习曲线的小样本和大样本训练轮次也不同。
- F8 有 11 条历史运行的训练样本 ID 仅能按规则重建，不能据此断言 B 与 C 训练样本完全一致。
- v1 测试过程中发现 BTC 的 U+200E 子词对齐问题；v2 保留模型/参数，记录修复后重新评测。v1 已完成的 48 项与 v2 预测一致，相关对照保存在 `results/b/all_models_alignment_version_comparison.json`。A/C 测试结果在本次选择前已存在，不能声称整个团队完全未接触测试信息。
- F4 原数值库发出警告，独立 OpenBLAS 重绘后指标及聚类成员经核验一致；`report/b/figures/f4_entity_representations_checked.pdf` 是复核图。MISC 的 144 条样例由助手辅助检查，独立人类复核仍待完成。
- `results/b/all_models_costs.csv` 部分条件无独立 profile；内存仅为 MPS 采样最大值，不是精确峰值或独立显存。
- B 的报告和结果是在本地整理，不能视为已经向 A/C 实际交接，也不能代替团队合稿、复现验证、人工判断及视频录制。
