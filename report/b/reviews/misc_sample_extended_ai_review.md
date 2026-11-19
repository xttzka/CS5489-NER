# MISC subclusters: extended assistant review of the sampled contexts

**Reviewer and status.** This is an AI assistant's qualitative reading of the
worksheet, not an independent human annotation. All 12 sampled contexts for
each of the 12 representation–cluster combinations were read: 144 sampled
occurrence entries in total. Some entries may refer to the same underlying
span across different representations, so these are not 144 independent
entities. A human researcher still needs to inspect and, where appropriate,
revise these interpretations. This document does not satisfy any requirement
for two independent human judgments or inter-rater agreement.

The review uses the CoNLL-2003 **validation** set. The three representations
share 904 matched MISC spans from 5,840 matched entities. All counts below are
entity **occurrences**, not distinct strings or sentences. Each representation
partitions the same 904 spans separately; a cluster number has no inherent
meaning and does not match the same number in another representation. The
independent OpenBLAS audit reproduced the membership of all 904 MISC spans in
all three representations; see `output/b_f4_checked_v1/misc_cluster_audit.json`.

**How to read the evidence.** The 12 rows in each worksheet cluster are
deterministically sampled uniformly without replacement from occurrences in
that cluster (seed 42). Repeated mentions are eligible separately, and the
same source sentence can contribute more than one entity. The worksheet's
“Most frequent” list instead counts surface strings across the **whole
cluster**, using their exact case-sensitive spelling, and displays the ten
most frequent forms. Thus a top form may not appear in the 12-context sample,
and sample composition must not be presented as the cluster-wide category
distribution. Qualitative categories below are informal and can overlap.

## GloVe

### Cluster 0 — 157 occurrences, 106 distinct surfaces

All 12 sampled forms: *Blitz; C$; Afrikaners; Ukrainians; Ontario-based; RBI;
Eurodollars; Sleeping Bread; GMT; EUROBONDS; Afrikaans; Tunisian*.

**Reading.** This is a heterogeneous cluster in the displayed sample. The
contexts include a historical bombing campaign, currency/financial terms,
population references, a place-derived modifier, a baseball statistic, a
novel title, a time-zone abbreviation and a language. Its top full-cluster
forms also include *RBI* (7) and *C$* (6), supporting the presence of repeated
technical expressions without implying that finance or sports dominate.
**Counterexample and uncertainty.** Neither a nationality label nor a
financial label describes *Sleeping Bread*, *Blitz*, and *Afrikaans*. The
context and original NER labels are treated as given, not independently
adjudicated. This strengthens the earlier review's “heterogeneous” assessment
and makes the financial, sports and literary examples explicit.

### Cluster 1 — 233 occurrences, 45 distinct surfaces

All 12 sampled forms: *Dutch; Canadian; Canadian; Australian; Dutch; German;
IRISH; Chinese; French; German; Polish; Irish*.

**Reading.** All 12 samples are country- or region-derived forms used in
different contexts, including bonds, a person, sports, government and the
press. Full-cluster frequent forms include *Russian* (22), *German* (21) and
*Dutch* (15). **Counterexample and uncertainty.** “Nationality adjective” is
an incomplete grammatical description: *IRISH* occurs in a headline and
*Dutch* can be used substantively. The samples support a nationality/geographic
theme, but not a claim that every one of the 233 occurrences has the same
grammatical role. This refines the earlier “country-derived adjectives and
nationalities” description without changing its main interpretation.

### Cluster 2 — 213 occurrences, 47 distinct surfaces

All 12 sampled forms: *Israeli; Chechen; Iraqi Kurds; Jews; Israeli;
Palestinian; Algerian; Moslem; Saudi; Egyptian; Israeli; Iraqi*.

**Reading.** The sample groups nationality, geopolitical, ethnic and religious
references, often in political, military or conflict-related news contexts.
Across the whole cluster, frequent forms include *Iraqi* (30), *Israeli* (26)
and *Moslem* (12). **Counterexample and uncertainty.** *Jews* and *Moslem*
cannot be described simply as country names; *Iraqi Kurds* combines multiple
descriptors. Context themes are observations in these 12 examples, not
estimates for the whole cluster. The earlier review already identified the
overlap among these categories; the additional contexts support keeping that
qualification.

### Cluster 3 — 301 occurrences, 134 distinct surfaces

All 12 sampled forms: *African; S. KOREAN; GERMAN CUP; Treaty of Baden;
Solidarity Meeting for Sarajevo; RUGBY LEAGUE; The Crying Game; Republican;
Atlanta Games; S. Korean; 1998 WORLD CUP; World Cup*.

**Reading.** The sample mixes geopolitical modifiers with sports competitions,
events and other named expressions. Whole-cluster frequent forms include
*European* (23), *World Cup* (19) and the separately counted *WORLD CUP* (13).
**Counterexample and uncertainty.** *Treaty of Baden* is an agreement and *The
Crying Game* is a film title, so “sports only” is too narrow; *Republican* in
the sampled context modifies a stadium and should not automatically be
interpreted as a political party. This preserves the earlier mixed-cluster
reading and adds film and context-sensitive modifier counterexamples.

## Unfine-tuned BERT

### Cluster 0 — 102 occurrences, 55 distinct surfaces

All 12 sampled forms: *FTSE 100; WORLD CUP; GMT; HONG KONG OPEN; WORLD CUP;
WESTERN DIVISION; MAJOR LEAGUE; TROFEJ BEOGRAD TOURNAMENT; F-16; 1997 FED CUP;
AXE; AMERICAN FOOTBALL-RANDALL*.

**Reading.** Uppercase sports headlines and abbreviated or technical forms are
prominent in this sample; whole-cluster frequent forms include *WORLD CUP*
(13), *C$* (6) and several division names (4 each). **Counterexample and
uncertainty.** *FTSE 100*, *GMT*, *F-16* and *AXE* occur in running text in
the sampled contexts, and their semantic functions vary. This is therefore
not an all-headline or purely sports cluster. The prior review's “many
uppercase headings or abbreviated expressions” remains appropriate if kept as
a qualified description.

### Cluster 1 — 586 occurrences, 219 distinct surfaces

All 12 sampled forms: *Arab; Zairean; Tour of the Netherlands; Sleeping Bread;
Boxer Rising; Bertillon 166; German; European; German; Yugoslav; American;
Islamic*.

**Reading.** The largest base-BERT MISC cluster contains many nationality or
geopolitical terms, but the sample also includes a cycling event, two novel
titles and a historical uprising. Its full-cluster frequent forms include
*Iraqi* (29), *Israeli* (26) and *German* (17). **Counterexample and
uncertainty.** The event, uprising and titles rule out a clean nationality-only
label. The sample supports the earlier “large mixed cluster” interpretation,
with broader named-event and literary counterexamples than the initial four
contexts showed.

### Cluster 2 — 145 occurrences, 56 distinct surfaces

All 12 sampled forms: *European; Pro Bowl; World Cup; Euro 96; Venice Film
Festival; ERA; African Nations Cup; U.S. Amateur Championship; World Cup;
Olympic; Olympic; Labor Day*.

**Reading.** Sporting competitions and sports contexts are frequent among
the displayed examples. Whole-cluster top forms include *European* (19),
*World Cup* (19), *RBI* (7), *U.S. Open* (6) and *Wimbledon* (6).
**Counterexample and uncertainty.** *Venice Film Festival* and *Labor Day*
are non-sport events, and *ERA* is a sports statistic rather than a
competition. *European* describes the qualifying group in its sampled
context. Thus the prior “frequently sports-related” characterization is more
defensible than a sports-event-only name.

### Cluster 3 — 71 occurrences, 40 distinct surfaces

All 12 sampled forms: *Russian; Polish; Indian; Australian; Chinese; Russian;
Dutch; Romanian; Nigerian; German; Libyans; Chinese*.

**Reading.** All 12 sampled mentions are country-derived references at token
index 0. An additional direct check of the complete assignment file finds
**71 of 71** mentions in this cluster start at token index 0. This is a
cluster-wide positional observation, separate from the 12-example qualitative
sample. **Counterexample and uncertainty.** *Libyans* is a population noun,
not an adjective; initial position is also not synonymous with a headline,
because some sampled examples are running sentences. Position may be
associated with the contextual representation; the observed cluster alone
does not establish causation. This strengthens the earlier sample-only note
about sentence-initial position while correcting the assumption that every
mention is an adjective.

## Fine-tuned BERT

### Cluster 0 — 440 occurrences, 89 distinct surfaces

All 12 sampled forms: *Algerian; Soviet; Nigerian; English; Palestinians;
Iraqi; German; Chinese; Poles; Kurdish; Indonesian; Israeli*.

**Reading.** All sampled mentions are nationality, geopolitical or population
references. The cluster's top forms include *Iraqi* (30), *Israeli* (26),
*Russian* (22) and *German* (21). **Counterexample and uncertainty.**
*Palestinians* and *Poles* are population nouns rather than adjectives;
*Kurdish* and *Soviet* need not be treated as simple present-day state
nationalities. The earlier characterization is supported, but “nationality
adjectives only” would overstate the evidence.

### Cluster 1 — 191 occurrences, 94 distinct surfaces

All 12 sampled forms: *A World of Things; GREATER MILWAUKEE OPEN; Moslem;
Fed Cup; Euro 96; Euro 96; Hong Kong Open; Trades Disputes Act; Nortel Open;
British Masters; National League; World Series of Golf*.

**Reading.** The sample and full-cluster frequent forms show many named sports
events, competitions and leagues; *World Cup* (19) and *WORLD CUP* (13) are
separate exact-case counts. **Counterexample and uncertainty.** *A World of
Things* is a novel title, *Trades Disputes Act* is legislation, and *Moslem*
is a religious descriptor in a party-related context. These make the earlier
qualified sports-focused reading appropriate, but a pure sports category
would be misleading.

### Cluster 2 — 133 occurrences, 103 distinct surfaces

All 12 sampled forms: *New York-based; Iraqis; C$; IRISH; Nasdaq; TROFEJ
BEOGRAD TOURNAMENT; Eurodollar; HONG KONG OPEN; In the Year of January;
SCOTTISH; Libyans; Sleeping Bread*.

**Reading.** These contexts include a geographic modifier, populations or
nationality-related terms, financial terms, competitions and novel titles.
The full-cluster most frequent form is *C$* (6); 103 distinct forms across 133
occurrences also indicate high surface-form variety. **Counterexample and
uncertainty.** No single theme covers *C$*, *HONG KONG OPEN* and *Sleeping
Bread*. “Heterogeneous” is supported; calling this a residual category or
inferring why the model groups these spans would require further evidence.
This expands the earlier description beyond the first four contexts.

### Cluster 3 — 140 occurrences, 65 distinct surfaces

All 12 sampled forms: *South Korean; GMT; Olympic; European; Korean;
Kalashnikov; European; Swede; North American; Nasdaq; Moslem; Moslem*.

**Reading.** The sample mixes regional or country-related terms, religious
references, sports-related modifiers and other named expressions. In the full
cluster, *European* (15) and *Moslem* (11) are among the frequent forms.
**Counterexample and uncertainty.** *GMT* is a time-zone abbreviation,
*Nasdaq* occurs as an index name and *Kalashnikov* as a weapon name in these
contexts. They prevent a nationality/religion-only description. This adds
financial and weapon-name examples to the earlier qualified mixed-cluster
reading.

## Comparison with the preliminary review and limits

The earlier `models/neural/B_misc_cluster_review.md` inspected frequent forms
and only the **first four** sampled contexts per cluster. Its broad themes
remain plausible after reading all 12, but the extended samples show additional
counterexamples: for instance, literature and finance in GloVe 0; a film
festival and Labor Day in base-BERT 2; legislation and a religious descriptor
in fine-tuned BERT 1; and finance, sports headlines and novels in fine-tuned
BERT 2. Base-BERT cluster 3 now has an independently counted 71/71
sentence-initial pattern in the assignment file; this is descriptive, not a
causal explanation.

The frozen primary model's CoNLL **test** seed 0 has MISC F1 77.96% with a
95% sentence-bootstrap interval of 75.21%–80.57%; it is the lowest point
estimate among its four entity types. These sampled clusters are from
**validation**, however, and sample diversity does not prove why test F1 is
lower. Use labeled error analysis and appropriate comparisons to investigate
that question. F4's scatter uses 1,000 matched spans, whereas the clustering
and worksheet above use all 5,840 matched spans and 904 MISC spans; the
sample populations and their ARI/NMI figures should not be conflated.

Source worksheet: `output/b_finish_v2/analysis/misc_review.md`. The sampled
rows can be traced by their span IDs in the worksheet and companion JSON. The
review is exploratory, not a replacement for human validation or a formal
agreement study.
