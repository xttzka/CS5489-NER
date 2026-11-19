# 角色 B 报告和图表

此目录是角色 B 本地交付副本，供 A 汇总，不是全组最终报告。旧的
`report/b_handoff/` 与 `report/figures/b_*` 按原样保留，以兼容已冻结的
实验和交付审计；此副本的逐文件来源与 SHA-256 记录在 `sources.json`。

- `role_b_report.pdf`：B 部分的英文独立展示稿（M0-M2 作为有归属的对照）。
- `role_b_report.tex`：独立报告入口；从仓库根目录编译，引用 `sections/`、
  `tables/` 和 `figures/`。需要 `pdflatex`、`geometry`、`lmodern`、`microtype`、
  `graphicx` 和 `booktabs`。
- `sections/`：方法和结果分析；`tables/`：T3 的三张表。
- `figures/f2_*`、`f3b_*`、`f4_*`、`f6a_*`：B 主要交付图；F4 为经过
  OpenBLAS 独立复核的版本。`f5_neural_*` 为 B 的神经模型补充图（包括 PPT
  用 PNG），不替代 A 的完整 F5；`f6b_*_from_a.pdf` 使用 A 的 CRF/M2 结果，
  不是 BERT 相关分析。
- `video_script.md`：B 的视频讲稿草案，尚需团队排练和真人录制。
- `reviews/`：MISC 的助手辅助样例审阅，不等同于独立人类复核。

报告是草稿：A/C 正文、参考文献、MISC 的独立人工复核，以及全组最终排版和
视频尚需团队整合。数字使用已冻结的评测；BTC tokenizer 工程修复、11 条历史
训练ID仅可重建等限制已写入 `sections/results_analysis.tex`。

图片、表格可以随代码共享；输入数据、实体向量、模型权重和完整预测按计划不
进入最终代码 ZIP，团队需要另行约定交接。`output/` 下的来源路径和模型权重不
会因仅提交本目录而自动分享。

`sources.json` 中的 source 是整理前的历史来源路径；若接收方只收到此目录、
没有旧 `report/b_handoff/` 和旧图文件，可以读取目标文件及其 SHA，但无法原地
重新对比那些源文件。F5 的原始 metadata 也保留绘图时的旧输出路径，它们是历史
记录；本目录中的对应图文件见 `figures/f5_neural_entity_familiarity.*`。

例如，从仓库根目录执行以下命令两次，第二次解析交叉引用：

```sh
pdflatex -interaction=nonstopmode -halt-on-error report/b/role_b_report.tex
```
