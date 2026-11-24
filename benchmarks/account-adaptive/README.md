# Adaptive account-history comparison

The model estimates how much the writing changes the odds that two handles share an author. It compares all independent posts on each side, including one post against a long history. The trained checkpoint is portable data-only JSON. The command accepts a prior or returns the evidence and three labeled prior scenarios.

## Method

**Represent each account as a distribution of posts.** Each post has measurements for capitalization, punctuation, word shapes, informal expressions, and function words. The model compares six distributions formed by three views of these measurements at two scales. For each, it measures average affinity between the accounts, distance between the two distributions, and a version corrected for self-comparisons where the history permits it. For instance, two writers may both use all caps and exclamation marks. These comparisons can distinguish a writer who typically uses them together from one who uses them in different posts. Fixed random Fourier features allow each distribution to be aggregated in batches without enumerating all pairs of posts. The ranker also sees pooled character and word-pattern similarities. These 306 inputs depend only on the supplied writing. [Kernel two-sample methodology](https://jmlr.org/papers/v13/gretton12a.html)

**Learn how the signals interact.** A regularized logistic model learns additive effects. A second model uses shallow boosted decision trees to learn conditional effects such as whether a punctuation similarity is informative given the other habits. Their scores receive equal weight before calibration. This pilot uses a CPU model and a fixed text representation. It can learn evidence about style variation present in the examples. Topic, cultural context, and changes in an individual's style can still affect its judgment. [Authorship and topic controls](https://aclanthology.org/2022.repl4nlp-1.26/)

**Train on variable writing budgets.** Training examples include single messages, uneven samples, and full histories. Each positive trial supplies the same anchor as its negative control. The positive and negative references match in post count and community composition, with comparable word totals. Known conversations stay separate across the two sides. The ranker trained on 9,597 usable same-handle or different-handle proxy trials from 2,087 Ubuntu writers. The data preparation excluded the 35 rename windows and their mapped handles from pretraining.

**Estimate evidence on different writers.** The ranker is frozen before calibration on 262 trials involving 79 other writer labels. Positive calibration examples come from 26 observed IRC nick changes. The calibration estimates a monotone score-to-log-odds curve whose slope varies with the amount of writing. Local support depends continuously on the number of posts and words on **both** sides. Reusing a writer across trials has a bounded contribution. Limited support shrinks the log evidence toward zero, and scores outside the observed calibration range are clipped. The result includes the amount of writing used and support diagnostics. These diagnostics measure the availability of similarly sized histories. They cannot establish that the relationship transfers to another platform.

**Apply the base rate for the particular search.** With likelihood ratio `L` and prior probability `p`, the estimated probability is `p*L/(1-p+p*L)`. Candidates shortlisted using these same writing features also require calibration on similarly shortlisted examples. If no base rate is available, the output gives three explicitly labeled scenarios. In the two fictional example profiles, all five and seven posts contribute. The released model produces a likelihood ratio of 1.818, which updates a 10% prior to 16.8%. The trained model uses same-session nickname changes as examples of common account operation. Interpreting those labels as real-world person identity assumes single-user accounts.

## Evaluation

The paired comparison reuses the 120 previously published held-out pairs from 11 IRC nickname changes in 11 windows. Of those pairs, 94 had unequal account sizes. The new model used every independent post in 120 comparisons. The earlier model used the full writing in 115. The original trial design had already been evaluated during development, so this is an exploratory comparison. The sparse variants use the same writers, with the retained posts selected by content hash before scoring.

| Task | Pairs | Baseline AUC | Adaptive AUC | Baseline Brier | Adaptive Brier |
|---|---:|---:|---:|---:|---:|
| Original IRC handle-change pairs | 120 | 0.797 | 0.822 | 0.186 | 0.177 |
| One post against a history | 120 | 0.500 | 0.805 | 0.250 | 0.193 |
| One post against one post | 120 | 0.500 | 0.646 | 0.250 | 0.234 |
| Four other IRC communities, same-handle proxy | 462 | 0.742 | 0.795 | 0.208 | 0.189 |

Brier loss measures probability error, and lower is better. This table uses a 50% prior. The earlier method abstained on all sparse pairs and receives the prior on those cases. On the original pairs, the 95% paired window-bootstrap interval for the AUC gain was −0.030 to +0.105. At a 10% prior, Brier loss moved from 0.0752 to 0.0737. The intervals and the small number of linked writers leave the size of the advantage uncertain.

**Topic sensitivity.** Plain word-count cosine, excluding 58 common terms, reached 0.721 AUC on the original pairs. We also matched positive and negative cases on identical anchor writing and reference post count, requiring their vocabulary cosines to differ by no more than 0.05. This retained 21 matched comparisons from nine rename writers. Lexical cosine scored 0.471 AUC, the earlier account model 0.694, and the adaptive model 0.730. Each account model ranked the genuine continuation higher on 16 of the 21 direct comparisons. Vocabulary matching is only a partial control for subject matter.

**New trial design.** Drawing directly from one through full histories gave 164 labeled pairs from 16 rename writers in 15 held-out windows. Adaptive AUC was 0.754 (95% window-bootstrap interval 0.615 to 0.864) and Brier loss was 0.202. This broader group includes account histories the original paired test never sampled.

**Transfer.** The four outside communities supplied 107 participating handles, including 42 with positive comparisons across conversations. Negatives came from the same community. Source-speaker exclusions, identity checks and screening for reused text were applied before evaluation. At a 10% prior, Brier loss fell from 0.0878 to 0.0784. AUC improved on Rust and Stripe, was nearly unchanged on MediaWiki, and declined from 0.654 to 0.645 on Ubuntu Meeting. The paired 95% interval for the overall AUC gain was −0.002 to +0.109. This measures cross-conversation recognition of an observed handle. Calibration for separately maintained social-media accounts, long time gaps, and cross-platform candidate searches still requires directly labeled examples from those settings.

## Reproduce

Follow the original account and external corpus preparation commands. Use the fixed source revision and source exclusions from the original pilot. The baseline checkpoint can be obtained by training the original pooled model with `--method baseline`. Use Python 3.12, scikit-learn 1.8.0, and one BLAS thread to minimize numeric differences. [Account data preparation](../account-profiles/README.md) · [External data preparation](../ubuntu-irc/README.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
author-match train-accounts data/linked-controlled.jsonl runs/adaptive.json.gz \
  --method adaptive --pretrain data/irc-prepared \
  --exclude-proxy-handles data/exclude-authors.txt --exclude-proxy-windows data/exclude-windows.txt \
  --link-evidence same_session_rename \
  --description "Observed IRC nickname changes; comparison handles were concurrently active in the same log window." \
  --report runs/adaptive-training.json
author-match evaluate-accounts data/linked-controlled.jsonl runs/adaptive.json.gz \
  --prior 0.5 --bootstrap-samples 1000 --output runs/native.json
python scripts/compare_account_methods.py data/linked-controlled.jsonl runs/accounts.json.gz \
  runs/adaptive.json.gz --output runs/paired.json
python scripts/evaluate_account_transfer.py data/external-prepared \
  --exclude-writing data/irc-prepared data/linked-controlled.jsonl \
  --model baseline runs/accounts.json.gz --model adaptive runs/adaptive.json.gz --output runs/external.json
```

Release checkpoint SHA-256 is `9a564ce310f316fbc1fda7b88726138514ed602c5c19e3e245b37202e707d478`. [Training](training.json) · [Same-pair comparison](paired.json) · [Native trial design](native.json) · [External communities](external.json)
