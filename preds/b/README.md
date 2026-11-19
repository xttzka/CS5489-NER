# 角色 B 的逐条预测

这里保存 B 的正式评估预测，来自 `output/b_finish_v2/evaluation/tasks/` 的冻结任务；训练过程中的验证预测和 smoke test 不包含在内。

目录：`preds/b/<目标数据集>/<模型>/<实验>__train-<训练来源>__<评测范围>__<条件>/seed<种子>.jsonl`。`bert`、`bilstm_crf`、`span_bert` 分别对应 M4、M3、Span。路径中的数据集是测试目标，训练数据来源见 `train-...`。

实验名：`domain` 域内主实验；`generalization300` 固定 300 句泛化；`learning` 学习曲线；`transfer` 跨领域迁移；`temporal` 时间泛化；`representation` 输入表示对比；`robustness` 输入扰动和替换；`distill` 蒸馏对照；`lora_match` LoRA 对照。`full` 与 `test_300` 代表不同评测范围；具体实验信息以 `manifest.json` 为准。

每行 JSON 记录一个 `sample_id` 和 BIO `tags`，不包含原句或金标签；评测需准备相应输入与标签。尤其扰动/实体替换任务必须使用清单中对应的派生输入与标签；`test_300` 还需要对应固定子集 ID。跨领域任务可能使用统一标签映射，不要混用原生标签评分。训练样本选择、标签空间和测试集信息应核对相关实验协议。

`manifest.json` 记录每份预测对应的任务、输入、数据范围、种子、原路径及 SHA-256；原路径指本次整理来源，克隆代码库不代表这些被忽略的 `output/` 文件也会出现。预测不包含模型权重、输入数据或评估结果；此前报告的汇总指标见 `results/B.csv` 与 `results/b/`。
