# Testing a suspected pair of accounts

The decision is whether the writing strengthens a suspicion about a particular pair. We tested ten scoring specifications using the same source-account exclusions as the released adaptive model. The source supplies IRC session nickname changes as positive labels. It contains no record of how a human would nominate a suspicious pair.

## Selection

The baseline ranker was already frozen on separate proxy writers. The new supervised character model used 9,597 eligible proxy pairs. It learned which shared 2–5 character sequences predict a common writer, using up to 48,000 corpus-weighted features. Other candidates used character TF-IDF cosine alone, bidirectional compression of original and word-masked text, or common post openings, endings and masked-character trigrams. Fusions added those scores to the frozen ranker margin.

We fitted each candidate’s shallow probability mapping on four folds of calibration windows and predicted the fifth. Each class received equal weight and each represented window within a class contributed equally to the fit. The selection score averaged log loss on all 262 out-of-fold predictions and on the 118 trials matched for vocabulary overlap. Matching used the same anchor writing, reference length in posts, and an absolute vocabulary-cosine difference no greater than 0.05. Folds kept entire windows together. We selected the candidate with the lowest loss, then fitted it on all 19 development windows.

| Development method | AUC | Selection loss |
|---|---:|---:|
| Ranker margin alone | 0.805 | **0.5854** |
| Ranker + post-boundary habits | 0.801 | 0.5866 |
| Ranker + learned character patterns and cosine | **0.807** | 0.5871 |
| Ranker + compression | 0.800 | 0.5886 |
| All signals | 0.794 | 0.5924 |
| Learned character patterns alone | 0.711 | 0.6503 |
| Character TF-IDF cosine alone | 0.613 | 0.6786 |
| Compression alone | 0.531 | 0.6926 |

## Evaluation

The selected experiment uses the existing ranker with a simpler window-weighted probability mapping. The release uses the same ranker with history-sensitive calibration.

| Evaluation | Pairs | Released AUC | Selected AUC | Released Brier | Selected Brier |
|---|---:|---:|---:|---:|---:|
| Recorded IRC renames | 164 | 0.754 | 0.749 | 0.2022 | 0.2026 |
| Vocabulary-matched subset | 102 | 0.680 | 0.675 | 0.2335 | 0.2351 |
| Four external IRC communities, same-handle proxy | 462 | 0.795 | 0.793 | 0.1894 | 0.1882 |

Brier losses use a balanced prior. At a threshold set from development mismatches (90th percentile), the selected method flagged 32 of 82 genuine continuations and 4 of 82 mismatches. On the vocabulary-matched subset it flagged 11 of 51 genuine continuations and 3 of 51 mismatches. The proportion of genuine pairs among flagged pairs in a real shortlist depends on its base rate and on how the shortlist was assembled.

Retain the released account model. None of the paired AUC differences established a gain. The harder evaluation contains only 16 positive owners, each contributing multiple trials. All evaluations reuse previously studied populations, and the character-model family was added after the first experimental round. Within each round selection used development predictions only. The external test measures recognition across conversations on a single handle. Separately maintained social accounts nominated by independent evidence would provide a more direct test.

## Reproduction

Prepare the linked and proxy datasets following the [adaptive benchmark](../account-adaptive/README.md). From the repository root run:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/experiment_suspected_pairs.py \
  data/linked-controlled.jsonl runs/adaptive.json.gz data/irc-prepared data/external-prepared \
  data/exclude-authors.txt data/exclude-windows.txt runs/suspected-pairs.json
```

The report contains all ten development results, final comparisons at two priors, grouped bootstrap intervals, operating-point counts, and the checkpoint fingerprint. [Aggregate results](results.json)
