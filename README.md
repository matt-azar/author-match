# author-match

Compare writing across two account histories, including accounts with different handles and unequal numbers of posts. The account model learns which writing habits recur together across an individual’s posts. It compares the two distributions of writing styles and estimates how much the evidence changes the odds of a shared writer. A separate single-post verifier returns a decision for two individual texts.

## Install and try it

Requires Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Compare two accounts

Supply a profile checkpoint and one JSONL export per account. Each line contains one original post, for example:

```json
{"handle":"writer_a","text":"i tried it twice this morning and got the same thing both times"}
{"handle":"writer_a","text":"maybe sunday? i should have more time to look then"}
```

```sh
author-match compare-accounts first.jsonl second.jsonl --model runs/accounts.json.gz --prior 0.10
```

For a shared export, use its path twice with `--left-handle NAME --right-handle OTHER`. JSON arrays of post strings, JSONL.gz, and directories with one UTF-8 text file per post are also supported. Only text contributes to the score. Handles select the two histories.

The prior is your estimated fraction of genuinely linked pairs among comparable candidates **before** reading their writing. A prior of 0.10 assumes about one genuine link in ten such pairs. The model estimates a likelihood ratio for the writing and multiplies the prior odds by that ratio. For example, a likelihood ratio of 3 changes a 10% prior to 25%, and a 1% prior to about 2.9%. Omit `--prior` to see the likelihood ratio and probability scenarios for 1%, 10%, and 50% starting probabilities. The model uses no assumed population match rate.

The command returns `estimated_same_person_probability` for a model calibrated on labeled different-handle pairs, along with the prior, evidence ratio, label interpretation, amount of writing used, and calibration support. The pilot uses reported IRC nickname changes as evidence of a common session operator. Applying that result to social accounts assumes the same relationship between writing and ownership. A model trained entirely on sections of the same account returns `same_account_proxy_probability` instead.

The adaptive model accepts any unequal or equal counts, including a single usable post. It removes duplicate and shared posts and uses every remaining independent post. A history of short replies and occasional formal explanations contributes both styles and how frequently they occur. The score combines similarities in character patterns and word choice with the distribution of habits within each post. A calibration model learns how the score relates to authorship across different quantities of writing. Weak support reduces the evidence smoothly toward the prior. Empty histories and substantial copied passages can still yield `inconclusive`.

The public benchmark compares the old and adaptive models on identical writing, including one-post stress tests and a transfer test in four other communities. The labeled handle links are IRC session renames. Calibration across separate social-media accounts remains an open measurement. [Adaptive methodology, benchmark, and reproduction](benchmarks/account-adaptive/README.md) · [5, 20, and 50 summaries per handle on Wikipedia](benchmarks/history-size/README.md)

### Train a profile checkpoint

A labeled JSONL file has one post per row with `person_id`, `handle`, and `text`. All known handles for one owner share a person ID. Supply `id`, `thread`, and `community` when available; optional `split_group` keeps all participants in a study window together and restricts its negative controls to that window. Preserve the original post boundaries. Each split needs writers with multiple handles and enough independent comparison writers.

```sh
author-match train-accounts owners.jsonl runs/accounts.json.gz
author-match evaluate-accounts owners.jsonl runs/accounts.json.gz --split test --prior 0.10
```

With limited ownership labels, use `--pretrain PREPARED_DIRECTORY` to learn writing-style weights from a separate collection, then calibrate and test on linked handles. Keep people, aliases, and source windows disjoint from the pretraining material. Pointing `train-accounts` at a prepared directory alone fits the same-account proxy model. Training defaults to `--method adaptive`. Use `--method baseline` to reproduce the original pooled account model. Both checkpoint formats load through the same comparison command. [Original pooled benchmark](benchmarks/account-profiles/README.md)

## Compare two individual posts

```sh
author-match compare first.txt second.txt --model runs/model.json.gz
```

This command uses the separate single-post checkpoint and returns a symmetric similarity score and one of `same_author`, `different_author`, or `inconclusive`. The Ubuntu pilot recovered 52 of 1,069 genuine pairs and falsely matched 7 of 1,069 different-writer pairs on its held-out test. [Single-post benchmark](benchmarks/ubuntu-irc/README.md)

### Train the single-post model

Supply one independent post per JSON line. `author` can be an opaque account ID. `thread` identifies the conversation across the entire dataset. `community` is optional but needed to compare community conditions. A numeric Unix `timestamp` enables time-gap reporting. Only the post text reaches the model.

```json
{"id":"p1","author":"a17","thread":"t48","community":"forum-a","text":"wait so does that work on weekends too??"}
```

ConvoKit's `utterances.jsonl`, its gzip equivalent, and corpus directories with `conversations.json` are also supported. Use many authors with independent posts across conversations. Inspect the preparation report to see how many usable pairs each partition contains. Cornell documents individual comments from 100 communities in its Reddit sample. If using that source, its published downloader (`from convokit import download; print(download("reddit-corpus-small"))`) returns the directory to pass to `prepare`. [ConvoKit Reddit sample](https://convokit.cornell.edu/documentation/reddit-small.html)

```sh
author-match prepare posts.jsonl data/prepared
author-match train data/prepared runs/model.json.gz --report runs/selection.json
author-match evaluate data/prepared runs/model.json.gz --split dev --output runs/dev.json
author-match compare first.txt second.txt --model runs/model.json.gz
```

Training compares word similarity, case-sensitive character trigrams, and logistic regression on symmetric style features. It selects the best development recall at the target false-match rate and uses eligible-pair ROC-AUC to break ties. A separate group of writers determines the final cutoffs. `--method style`, `--method char`, or `--method word` trains a specific baseline. CPU checkpoints use data-only JSON or gzipped JSON. Files with a `.joblib` suffix use Python object serialization and should come from a trusted source.

For a quick demonstration, run `python examples/make_demo.py posts.jsonl`, then `author-match prepare posts.jsonl data/demo --pairs-per-author 32`. Use `data/demo` as the training directory. The generator deliberately embeds style patterns, so use its scores only to check that the pipeline works.

## What the benchmark measures

- **Independence.** Training, development, calibration, and test partitions have disjoint accounts and threads. Same-account examples come from different threads. The sampler replaces a positive partner with a different writer in the same community and length band where possible. It includes same-thread negatives and caps pair appearances per account. Examples with conflicting copies are removed before assignment.
- **Style.** The cleaner preserves punctuation, casing, informal spelling, emoji, and paragraph boundaries. It strips block quotes, reposts, and common signature or code blocks, and masks handles and link targets. Character TF–IDF measures shared three-character sequences. The learned model also measures punctuation habits, contraction use, function words, emoji, and repeated letters. Metadata is used for labels and testing.
- **Operating point.** Default calibration targets 1% false same-author calls among all different-author pairs and 5% false different-author calls among all same-author pairs. A call must pass both the pooled threshold and the threshold for the shorter post's length band. Unsupported bands abstain. These are sample targets. The measured test rates may differ.
- **Coverage.** Recall counts all genuine pairs. False-match rate counts all different-author pairs. Coverage is the fraction receiving a definitive decision. Read them together. These denominators refer to sampled pairs after cleaning and deduplication. The JSON report includes a decision table, per-band and per-condition counts, author-bootstrap uncertainty intervals, and ranking metrics.
- **Independent tests.** For a community-transfer experiment, prepare a separate dataset with `--holdout-community NAME` (repeatable). For a particular known bot or shared account, use `--exclude-account ID`. After freezing the model and settings, evaluate `--split test`. A same-account label remains a proxy for shared human authorship. Test each new platform on independently labeled writing before interpreting its scores.

For an independent external corpus, use `author-match prepare external.jsonl data/external --external` and `author-match evaluate data/external runs/model.json.gz --split external`. All external messages are available for evaluation. The evaluator rejects overlapping writer IDs and rejects an ordinary test split from a different preparation. Give writer IDs a consistent source namespace when combining platforms.

**Multiple posts.** Add `--bundle-mode 1x5 --bundle-mode 5x5` to training to calibrate these modes separately. Evaluate with `--mode 1x5` or `--mode 5x5`. At inference, use `compare-bundles --model MODEL --left A.txt --right B1.txt B2.txt B3.txt B4.txt B5.txt`. Supply independent posts per side. The model takes the median of eligible component scores, requires a majority to be eligible, and treats each bundle pair as one observation. A model lacking enough independent calibration writers will abstain.

## Optional STAR encoder

Install `python -m pip install -e '.[neural]'`. The frozen encoder uses a pinned revision of STAR. A local Transformers checkpoint directory can be supplied with `--checkpoint`. Use `--offline` to limit loading to local files. [STAR model and training paper](https://huggingface.co/AIDA-UPM/star)

```sh
author-match train data/prepared runs/star.joblib --method star --device cuda
author-match finetune data/prepared runs/encoder --device cuda --steps 100
author-match train data/prepared runs/tuned.joblib --method star --checkpoint runs/encoder --device cuda
```

Fine-tuning samples multiple authors and two posts in distinct conversations per author, then optimizes supervised contrastive loss. The saved encoder can be calibrated or compared with the CPU baselines (`train --include-star`). Neural artifacts reference their encoder. Keep the local checkpoint available at the recorded path when reopening a model.

## Verification

```sh
python -m unittest discover -s tests -v
```

The optional neural test runs forward scoring, optimizer steps, and checkpoint reload on a tiny local RoBERTa model. It requires the neural dependencies and runs offline.
