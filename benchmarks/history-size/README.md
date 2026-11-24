# Does additional writing help verify a suspected pair?

We applied the unchanged IRC-trained adaptive model to Wikipedia edit summaries. Wiki-Socks groups confirmed alternate accounts by investigation and supplies comparisons from other editors. The text is the editor's description of an edit. It differs from the article text, a talk-page discussion, and an ordinary social-media post. Dates and page names select comparable histories. Only the summaries enter the model. [Wiki-Socks dataset and label definitions](https://github.com/lraszewski/wiki-socks/tree/b488e891e0840bb7d89c2dc03280af18d403f064)

## Design

**Exact page and month.** We screened investigations whose metadata indicated at least 100 positive contributions, 50 control contributions, and 60 days of activity. We required distinct confirmed handles in the same investigation for the genuine comparison and one comparison writer identified by the source as a control. Each genuine continuation summary is paired with a control summary from the same page and calendar month. Both later histories occur on pages absent from the anchor and start at least 60 days after its final summary. This requires the same control writer to have enough eligible summaries. To compare budgets, we use the same accounts and nested writing samples. Each cutoff is the number of summaries supplied **per handle**.

**Cross-investigation controls.** To accommodate longer histories we built a second cohort using only investigations with two confirmed handles that could each supply 50 qualifying summaries. We paired different investigations, requiring the early and later pages to be disjoint for both genuine and swapped continuations. The minimum time gap of 60 days applies to the swapped comparisons too. For the two later histories, the minimum, quartiles, and maximum date each had to agree within 92 days. For either anchor, switching from its genuine continuation to the other investigation's continuation had to change the full-history content-word cosine by at most 0.05. This controls one measure of vocabulary overlap. It allows different subjects with similar vocabulary overlap, and matching at 50 summaries does not guarantee equal overlap in a five-summary subset. We select one anchor per disjoint investigation pair, giving one genuine and one control comparison at each budget.

In both protocols we strip section headings and wiki markup, remove common automated summaries, require four or more words, and deduplicate normalized text within each handle. We exclude writing shared by the early and genuine later histories; the exact protocol also removes writing shared between the matched references. Live model cleaning can further reduce the number of summaries used. Residual templates, tools, and editing conventions can still affect the scores. The code selects eligible cases in stable hash order and samples without access to the model scores. No account is reused in a protocol. Controls approximate different operators according to the dataset. Unrecorded ownership links are possible.

We report ROC AUC, Brier loss with a 50% prior, and how often the genuine continuation outranks the control for the **same** early account (ties count half). Paired bootstrap intervals resample independent case units, retaining the same cases at each budget. They condition on the sampled summaries and fitted model. An inconclusive prediction receives the prior. These probabilities were calibrated on IRC. Absolute probabilities for separately maintained social-media accounts require target-domain calibration.

## Results

The exact design screened 9,360 investigations and retained 72 independent cases at five summaries per handle. The genuine continuation outranked its same-page, same-month control with **79.2% concordance**. Across all 144 comparisons, AUC was **0.720** and Brier loss was **0.217** at the balanced prior. The model produced estimates on all 144 comparisons and used fewer than five independent summaries on a side in two. Zero cases met the exact control requirements at 20 or 50 summaries.

The longer-history design screened 9,388 investigations. Of those, 553 supplied qualifying genuine histories. Matching disjoint investigations on time and lexical overlap produced **12 independent case pairs**, each yielding one genuine comparison and one mismatch. We used these same 12 case pairs at all three sizes. After examining the initial nested draw we repeated it with 19 more orderings of the same summaries. The resampling is a sensitivity analysis and does not create additional writers. We average each metric over the 20 draws. Paired intervals resample the 12 case pairs together, preserving the class pairing and all draws.

| Summaries per handle | Mean AUC | Genuine continuation outranks control | Mean Brier loss |
|---|---:|---:|---:|
| 5 | 0.511 | 50.8% | 0.293 |
| 20 | 0.540 | 57.5% | 0.284 |
| 50 | 0.583 | 66.7% | 0.283 |

The mean AUC increase from 5 to 50 was 0.072, with an approximate paired 95% interval of −0.031 to +0.191. Brier loss at every budget exceeded the 0.250 loss from always returning the balanced prior. With all 50 summaries, the genuine continuation ranked above the alternative in eight of 12 cases. The original single draw produced 0.722 AUC at 20 summaries. Across 20 draws, that was the maximum, and the mean was 0.540. The full 50-summary profiles are identical across draws. The results suggest that more writing can improve ranking, but the size of the gain is uncertain and the transferred probabilities performed poorly.

The two cohorts contain different controls and qualify under different rules. Compare history sizes **within** the 12-case cohort. The five-summary measurement from the 72 exact-match cases measures a separate task.

## Reproduce

From the installed author-match checkout:

```sh
git clone --no-checkout https://github.com/lraszewski/wiki-socks.git data/wiki-socks
export OPENBLAS_NUM_THREADS=1
python scripts/experiment_history_size.py data/wiki-socks runs/adaptive.json.gz runs/exact-history-size.json
python scripts/history_size_cross_case.py data/wiki-socks runs/adaptive.json.gz runs/cross-case-history-size.json
```

Both scripts read the pinned Git objects directly. Reports contain aggregate findings and checkpoint fingerprints. [Exact matching](results.json) · [Cross-investigation matching](cross-case.json)
