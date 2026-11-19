# MISC entity subclusters: preliminary assisted review

The following interpretation was prepared by the assistant, not by an
independent human annotator. It examines the frequent forms and the first four
of the 12 deterministically sampled contexts per cluster in
`output/b_finish_v2/analysis/misc_review.md`; it is a qualitative sample, not
a full annotation of all occurrences. All three representations use the same
904 matched CoNLL validation MISC spans; cluster IDs have no meaning across
representations. The original run emitted numerical-library warnings. An
independent OpenBLAS recomputation reproduced every cluster membership for all
5,840 matched entities and all 904 MISC entities; details are in
`output/b_f4_checked_v1/misc_cluster_audit.json`. The original and checked
figures and logs remain separate.

| Representation | Cluster | Occurrences | Tentative reading of the inspected examples |
|---|---:|---:|---|
| GloVe | 0 | 157 | Heterogeneous: the sample contains a wartime event (*Blitz*), a currency symbol, and population names. |
| GloVe | 1 | 233 | Frequent country-derived adjectives and nationalities, such as *Dutch*, *Canadian*, and *Australian*. |
| GloVe | 2 | 213 | Many nationality, ethnic, religious, or geopolitical references, including *Israeli*, *Chechen*, and *Iraqi Kurds*; these categories overlap rather than forming a clean label taxonomy. |
| GloVe | 3 | 301 | Mixed geographic adjectives, sporting competitions and other named events or agreements, including *GERMAN CUP* and *Treaty of Baden*. |
| Base BERT | 0 | 102 | Many uppercase headings or abbreviated expressions, including *WORLD CUP*, *GMT*, and *FTSE 100*. |
| Base BERT | 1 | 586 | A large mixed cluster: common nationality adjectives, a cycling event and a novel title occur in the inspected examples. |
| Base BERT | 2 | 145 | Frequently sports competitions and sports-related contexts, with examples such as *Pro Bowl*, *World Cup*, and *Euro 96*. |
| Base BERT | 3 | 71 | Inspected examples are nationality adjectives at the beginning of their sentences; position or headline style may contribute, but this is not established causally. |
| Fine-tuned BERT | 0 | 440 | Many nationality or geopolitical adjectives, including *Algerian*, *Soviet*, *Nigerian* and *English*. |
| Fine-tuned BERT | 1 | 191 | Many sporting-event and league names, mixed with other expressions such as a novel title and *Moslem*. |
| Fine-tuned BERT | 2 | 133 | A heterogeneous remainder including nationality references, a currency symbol and a geographic modifier. |
| Fine-tuned BERT | 3 | 140 | Mixed national/regional, religious and other expressions; inspected examples also include *GMT* and *Olympic*. |

For the frozen primary model's CoNLL test seed 0, MISC has F1 77.96%
(95% sentence-bootstrap interval 75.21%–80.57%), lower as a point estimate
than LOC, ORG and PER. The diverse examples above show that MISC is a broad
annotation category, which is a plausible hypothesis about difficulty. The
cluster inspection alone does **not** demonstrate that this diversity caused
the lower F1; establish that with an appropriate labeled error analysis rather
than treating cluster membership as a ground-truth explanation. The worksheet
and cluster assignments allow a human researcher to review or revise these
suggestions.
