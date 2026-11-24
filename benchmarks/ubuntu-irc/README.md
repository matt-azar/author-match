# Ubuntu IRC single-message benchmark

This pilot compares two individual chat messages using text alone. It tests whether an account uses recognizable writing habits across independent conversations. The labels are IRC nicknames, so a shared or reassigned nickname can introduce label error.

The fitted CPU checkpoint is a logistic regression model using writing-style features and character trigram similarity. On development writers it recovered 5.18% of same-account pairs at the provisional false-match operating point, compared with 4.99% for character trigrams alone and 4.07% for word similarity. After selecting the model, we fitted its decision thresholds on a separate calibration split. The small difference between candidates leaves their ordering uncertain.

| Frozen evaluation | Pairs / distinct participating writers | Genuine matches found | False matches | Any definitive decision | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Held-out Ubuntu writers and threads | 2,138 / 498 | 52 / 1,069 (4.86%) | 7 / 1,069 (0.65%) | 231 / 2,138 (10.80%) | 0.635 |
| Four other IRC channels | 556 / 132 | 25 / 278 (8.99%) | 6 / 278 (2.16%) | 78 / 556 (14.03%) | 0.587 |

The 1% false-match target was exceeded in the other channels. On Ubuntu, the approximate 95% author-bootstrap interval for its false-match rate was 0.10%–1.81%. Both tests contain many short messages and few long ones. Neither dataset establishes Reddit, cross-platform, or person-level accuracy. Coverage refers to sampled pairs after filtering and deduplication.

| Shorter message on Ubuntu test | Genuine pairs | Genuine matches found | False matches / different-writer pairs | Coverage |
|---|---:|---:|---:|---:|
| Under 10 words | 828 | 29 (3.50%) | 5 / 828 | 9.60% |
| 10–24 words | 226 | 23 (10.18%) | 2 / 226 | 15.27% |
| 25–49 words | 15 | 0 | 0 / 15 | 10.00% |
| 50 words and over | 0 | — | — | — |

The upper threshold was disabled for 25+ words due to sparse calibration data. An unmeasured length range has no established recall. The lower threshold still allowed some different-author decisions in the 25–49 word band.

## Source and protocol

Primary data came from 58,686 messages in 153 annotated Ubuntu IRC windows (6,318 nicknames and 9,146 conversation components). The converter removes system events, known bots and leading direct-address nicknames, hashes speaker IDs, and reconstructs threads from the annotated reply edges. The original windows span 2004–2018. Their timestamps were omitted because a message's absolute time cannot always be recovered from the displayed clock. Pasted commands and technical vocabulary can still influence similarity. The authors release the data under CC BY 4.0. [Kummerfeld et al., ACL 2019](https://aclanthology.org/P19-1374/) · [Publisher data and license](https://github.com/jkkummerfeld/irc-disentanglement/tree/82ed04f9627a45d0d6f2ec3c8683c88eb408a0d5)

The separate evaluation uses 1,982 messages in 200 annotated conversations from MediaWiki, Rust, Stripe, and Ubuntu Meeting. The converter excludes nicknames appearing in the original Ubuntu source context, drops ambiguous cross-network nicknames, and removes matching source text as described in the converter. All four channels are used together to test transfer. Same-writer trials come from separate conversations. They do not establish cross-platform identity. [Gouravajhala et al., ALTA 2023](https://alta2023.alta.asn.au/files/13.pdf)

Both corpora were then processed with preparation format 2, seed 42, and a cap of 12 pair appearances per writer. Preparation discarded blank or fully quoted messages and ambiguous copies. All model-selection and calibration decisions preceded final scoring. The two test sets were scored with the same frozen checkpoint. Reports contain aggregate data only.

## Reproduce

From a checkout with the CPU dependencies installed:

```sh
git clone https://github.com/jkkummerfeld/irc-disentanglement.git data/irc-source
git -C data/irc-source checkout 82ed04f9627a45d0d6f2ec3c8683c88eb408a0d5
python scripts/prepare_irc_pilot.py data/irc-source data/irc.jsonl
python scripts/prepare_irc_pilot.py --external data/irc-source data/irc-external.jsonl
author-match prepare data/irc.jsonl data/irc-prepared
author-match prepare data/irc-external.jsonl data/external-prepared --external
author-match train data/irc-prepared runs/irc-model.json.gz --report runs/selection.json
author-match evaluate data/irc-prepared runs/irc-model.json.gz --split test --output runs/test.json
author-match evaluate data/external-prepared runs/irc-model.json.gz --split external --output runs/external.json
```

The measured run used CPython 3.12, scikit-learn 1.8.0, NumPy 2.3.5, and SciPy 1.17.0. Exact floating-point scores may vary by version or platform. The checkpoint from this run has SHA-256 is `f8a46e2493820428fb0a26d0ca93069a396c3f1e4340ed4316446503122101d0`.

[Training corpus manifest](data.json) · [External corpus manifest](external-data.json) · [Model selection](selection.json) · [Ubuntu test report](test.json) · [External test report](external.json)
