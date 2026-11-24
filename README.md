# author-match

Compare the writing style of two informal English posts by authors the model may never have seen. The CLI returns a symmetric similarity score and one of `same_author`, `different_author`, or `inconclusive`. Empty, copied, extremely short, and unsupported inputs can produce an inconclusive decision. The score ranks stylistic similarity. Its numerical value is not a probability of shared identity.

## Install and try it

Requires Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Train a CPU checkpoint with the commands below, or use one you already have. To compare two UTF-8 files:

```sh
author-match compare first.txt second.txt --model runs/model.json.gz
```

The pilot recovered 52 of 1,069 genuine pairs and falsely matched 7 of 1,069 different-writer pairs on held-out Ubuntu conversations. On four other chat channels it recovered 25 of 278 and falsely matched 6 of 278. At these thresholds most pairs receive `inconclusive`. Reddit accuracy is unmeasured. [Pilot results and reproduction](benchmarks/ubuntu-irc/README.md)

## Train on your own data

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
