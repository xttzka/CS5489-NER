# 角色 B：视频讲稿草案

原计划把 **0:25–1:25** 分配给 B、A 共同介绍主结果和记忆分析，把 **1:25–2:15** 分配给 B 介绍跨领域结果。下文的第一段故意留出约 20 秒给 A 或组内过渡；谁具体讲哪一句由你们讨论。两段都是可朗读的英文草稿，正式使用前需根据整组幻灯片调整、录音并计时。

## 0:25–1:25：主结果与实体记忆（B、A 共享）

**B 的英文稿：**

> On the full CoNLL test set, the dictionary baseline reaches 56.7 F1, compared with 90.6 for fine-tuned BERT. We selected BERT using validation data. Looking at entity names, its F1 is 94.6 on names seen during training and 85.6 on unseen names. In a separate 500-sentence comparison, replacing entity names lowers F1 from 95.0 to 92.7. These results suggest that familiar names help, while they do not establish that memorization is the only explanation.

**中文提纲：**先用 F2 说明方法序列的实际结果，再用 F5 的“见过／未见”和实体替换结果说明名称依赖。两个 F1 是不同实体分组的描述性结果；替换实验使用独立构造的 500 对句子。这个片段预留约 20 秒供 A 补充其记忆分析和过渡。

**长度与时长：**按空格和标点统计约 74 个英文词；按每分钟 110–130 词粗估约 34–40 秒。数字口读和换页会增加用时，需实际计时。

**画面：**展示 [F2](figures/f2_method_comparison.pdf)，随后展示 [B 的神经模型 F5 补充图（PDF）](figures/f5_neural_entity_familiarity.pdf) 或 [PPT 用 PNG](figures/f5_neural_entity_familiarity.png) 的 (a)、(b) 面板。该补充图展示 M3、M4 和 Span；A 的 `report/figures/f5.pdf` 展示 M0–M2，两个文件可按各自方法范围并列使用。不要暗示 A 的原图已包含 M4。

## 1:25–2:15：跨领域泛化（B）

**英文稿：**

> For cross-domain transfer, we evaluate only the shared person, location, and organization types. The CoNLL-trained BERT reaches 92.35 F1 on CoNLL, 79.31 on WikiGold, and 38.30 on BTC. Training on both CoNLL and WNUT raises the observed BTC result to 46.85 and the WNUT result from 48.01 to 54.95. These are seed-zero comparisons, not isolated causal effects of adding one dataset. In A's CRF analysis, name overlap and transfer loss have a descriptive correlation of minus 0.57; that association does not establish causation.

**中文提纲：**用 F6(a) 说明统一三类标签下的迁移矩阵；域内和跨域数字均是 seed 0 的完整测试集结果。合并训练在 BTC、WNUT 上数值更高，但同时改变了训练数据和设置，不能把差值直接解释成某单一因素的因果效应。F6(b) 的相关性来自 A 的 **CRF（M2）**，不是 BERT。

**长度与时长：**同一统计方式约 83 个英文词；按每分钟 110–130 词粗估约 38–45 秒。数字口读和换页会增加用时，需实际计时。

**画面：**展示 [F6(a)：BERT 迁移矩阵](figures/f6a_bert_transfer.pdf)；最后一句展示 [F6(b)：A 的 CRF 实体重叠分析](figures/f6b_crf_overlap_from_a.pdf)，图注明确相关性是描述性的。

## 统计与演练说明

- F2、F5 所述指标为实体级精确匹配 F1；上述神经模型及迁移矩阵均为固定的 seed 0，不能把句子级 bootstrap 区间说成随机种子之间的波动。
- “Seen”表示实体名称经大小写折叠后在训练实体名称中出现过，匹配不依赖类型。其余属于“unseen”。F5 的替换和原句是 **500 个配对样本**；说的是这组样本上的前后比较，而不是将全测试集 F1 与替换集 F1 混合。
- F6 的数字来自共同 PER/LOC/ORG 标签空间的完整测试集，不能直接与原始标签空间下的全量 F1 或固定 300 句结果混作同一口径。
- B 的 F5 图的 (a)、(c) 误差线是单个模型的 1,000 次句子 bootstrap 95% 区间，并非三随机种子的标准差；(b) 展示点估计，配对差值的 95% 区间及输入哈希见 `report/b/figures/f5_neural_entity_familiarity.metadata.json`。CoNLL 与 CoNLL++ 是两个不同测试集，(c) 连线只是帮助展示数值变化，并非对句子进行配对。
- 这是根据实验结果编写的讲稿，不包含真实录音；A 的讲话、F5 画面更新、组内排练及最终视频录制仍需要团队完成。建议至少排练两次，以实际语速和换页时间确定最终文本。

## 可追溯来源

- 论文式报告与 F2 数字：`report/b/sections/results_analysis.tex`；机器可读原始配对比较：`output/b_finish_v2/analysis/paired_models_conll_seed0.json`。
- BERT 的 seen/unseen：`output/b_finish_v2/additional_analysis/familiarity__M4__conll2003.json`。
- BERT 实体替换：`output/b_finish_v2/additional_analysis/paired__M4__replacement_replaced_v2.json`。
- B 的 F5 补充图：`models/neural/plot_f5_neural_b.py`，附 `report/b/figures/f5_neural_entity_familiarity.metadata.json` 中的输入、图文件哈希和完整图注。
- F6 迁移矩阵：`report/b/sections/results_analysis.tex`、`results/B.csv`（按任务与标签空间筛选）。
- A 的 M2 重叠分析：`results/B_f6b_from_a_m2.json`（12 个源外领域×类型点，描述性 Pearson 相关系数约 -0.569）。
