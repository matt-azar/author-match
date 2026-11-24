# Comparing account histories across different handles

This pilot asks whether two collections of independently supplied posts show evidence of one writer. The unit of prediction is a pair of histories. Post counts can differ. A likelihood ratio learned on held-out labeled writers updates the user's prior odds of a match. The data-only checkpoint is reproducible with the commands below.

## What the labels mean

The test uses **explicit nickname changes reported by an IRC server**. Messages before and after a rename provide a known link between two handles on one session. The control histories come from different handles active in the same window. A human operating multiple simultaneous connections or sharing an account can violate the ownership assumption. The measured probability concerns this operational definition of a match. The dataset does not independently establish real-world identities or accuracy across platforms.

The source converter retained 44 observed renames and 156 nearby control writers across 35 recording windows, with 3,112 messages before the benchmark's duplicate filtering. It redacted active nicknames from both classes. Each window supplied three to five controls. The splitter kept windows and their participants together. The sampler matched profile post counts and selected controls of similar writing volume within the same window. Both labels can therefore share vocabulary and timing. Known conversation IDs are kept apart across the two sides. Some messages lack conversation annotations. The underlying Ubuntu chat data is distributed under CC BY 4.0. [Kummerfeld et al., ACL 2019](https://aclanthology.org/P19-1374/) · [Source revision and data license](https://github.com/jkkummerfeld/irc-disentanglement/tree/82ed04f9627a45d0d6f2ec3c8683c88eb408a0d5)

## Fitting the model

1. Each account is represented by within-post character, word, and token-pattern distributions and by the means and dispersions of stylometric measurements across its posts. Symmetric comparisons of the two profiles yield 288 features. Handles, person IDs, timestamps, and window identifiers are unavailable to the classifier.
2. A regularized logistic regression learned feature weights from same-handle and different-handle histories on other Ubuntu writers. Training used 5,265 eligible profile pairs from 1,313 writers. We excluded all 35 linked-data windows and 388 source handle identifiers from the candidate training data.
3. A second, one-dimensional logistic fit mapped the classifier's score to a log likelihood ratio with equal weight on the two classes. It used 112 positive and 112 negative trials from the calibration partitions. Smoothing, shrinkage, and evidence bounds were based on an effective count of 23 distinct contributors. Repeated trials from one writer cannot count as independent people. Raw scores beyond the observed calibration range are clipped.
4. For a likelihood ratio `L` and candidate-pair prior `p`, the estimated match probability is `L*p / (1-p+L*p)`. Reports include the prior and likelihood ratio separately. History sizes must have at least six comparable calibration people for each checked class/role. For longer histories, the comparison can choose a deterministic subset without using match scores. The output reports how many posts and words contributed.

## Held-out results

The frozen model was evaluated on **120 sampled pairs** (60 positive, 60 negative) from 46 participating writer labels. The positive pairs came from **11 distinct rename cases**. Trials occupied 11 held-out recording windows. The model estimated a probability for all 120 pairs and used a representative subset for five. The two rows below use the same trials, reweighted to the stated candidate prevalence.

| Assumed prior | Model Brier loss | Always predicting prior | Model log loss | Always predicting prior |
|---|---:|---:|---:|---:|
| 50% | 0.1862 | 0.2500 | 0.5550 | 0.6931 |
| 10% | 0.0752 | 0.0900 | 0.2667 | 0.3251 |

Lower loss means better probability predictions. The model ranked a randomly selected positive pair above a randomly selected negative pair about 80% of the time (ROC-AUC 0.797, approximate 95% interval 0.641–0.906 from resampling complete windows). This sample is too small to validate fine distinctions between nearby probability values. Real Reddit, X, and cross-platform accounts remain outside the measured domain. Candidate pairs selected by a prior writing-style search may also have a different distribution of scores.

## Reproduce

From the repository root with author-match installed:

```sh
git clone https://github.com/jkkummerfeld/irc-disentanglement.git data/irc-source
git -C data/irc-source checkout 82ed04f9627a45d0d6f2ec3c8683c88eb408a0d5
python scripts/prepare_irc_pilot.py data/irc-source data/irc.jsonl
author-match prepare data/irc.jsonl data/irc-prepared
python scripts/prepare_irc_controlled_handles.py data/irc-source data/linked-controlled.jsonl --exclude-proxy-authors data/exclude-authors.txt --exclude-proxy-windows data/exclude-windows.txt
author-match train-accounts data/linked-controlled.jsonl runs/accounts.json.gz --pretrain data/irc-prepared --exclude-proxy-handles data/exclude-authors.txt --exclude-proxy-windows data/exclude-windows.txt --link-evidence same_session_rename --description "Observed IRC nickname changes; comparison handles were concurrently active in the same log window." --report runs/account-training.json
author-match evaluate-accounts data/linked-controlled.jsonl runs/accounts.json.gz --prior 0.5 --output runs/account-test.json
author-match evaluate-accounts data/linked-controlled.jsonl runs/accounts.json.gz --prior 0.1 --output runs/account-test-10.json
author-match compare-accounts examples/account-posts.jsonl examples/account-posts.jsonl --left-handle sample_a --right-handle sample_b --model runs/accounts.json.gz --prior 0.10
```

The final command uses two fictional accounts with five and seven posts. Both complete histories fit the pilot's evidence range. With this checkpoint the writing yields a likelihood ratio of 0.778, which moves the specified 10% prior to **7.96%**. This example demonstrates the CLI and the prior adjustment.

[Training summary](training.json) · [Balanced evaluation](test-balanced.json) · [10% prior evaluation](test-ten-percent.json)
