"""Evaluate frozen account models against same-handle histories in unseen communities."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from compare_account_methods import compare_predictions

from author_match.accounts import _identities, _source_digest, compare_accounts
from author_match.data import SPLITS, load_split
from author_match.experiment import write_json
from author_match.profile_features import usable_post_key
from author_match.profile_io import load_profile_model
from author_match.profile_probability import eval_report, quality_intervals
from author_match.profile_trials import proxy_posts, read_linked_posts, sample_trials
from author_match.text import clean_text, comparison_words


def fingerprint(text):
    text = clean_text(text)
    return usable_post_key(text, comparison_words(text))


def excluded_writing(sources):
    keys = set()
    for source in sources:
        if source.is_dir():
            for split in SPLITS:
                keys.update(fingerprint(p.text) for p in load_split(source, split)[0])
        else:
            keys.update(fingerprint(p.text) for p in read_linked_posts(source))
    keys.discard(None)
    return keys


def transfer_trials(posts, excluded, seed):
    """Keep both endpoints in the same community, including all negative controls."""
    grouped = defaultdict(list)
    counts = Counter()
    for post in posts:
        if not post.community or not post.thread:
            raise ValueError('each external post requires a community and a conversation')
        key = fingerprint(post.text)
        if key is None:
            counts['no_usable_prose'] += 1
        elif key in excluded:
            counts['seen_source_writing'] += 1
        else:
            grouped[post.community].append(post)
    trials = []
    for community, records in sorted(grouped.items()):
        for trial in sample_trials(records, label_basis='account_proxy', seed=seed):
            trials.append((community, trial))
    return trials, dict(counts)


def evaluate(artifact, trials, priors, seed, bootstrap):
    rows = []
    fallback = Counter()
    for community, trial in trials:
        result = compare_accounts(artifact, trial.left, trial.right, prior=0.5)
        estimated = result['status'] == 'estimated'
        if not estimated:
            fallback[result['reason']] += 1
        rows.append((community, trial, result['log_likelihood_ratio'] if estimated else 0., estimated))

    def summary(items):
        labels = [trial.same for _, trial, _, _ in items]
        evidence = [log_lr for _, _, log_lr, _ in items]
        data = {'trials': len(labels), 'same': sum(labels), 'returned_estimates': sum(x[3] for x in items),
                'participating_handles': len({p for _, t, _, _ in items for p in (t.left_person, t.right_person)}),
                'positive_handles': len({t.left_person for _, t, _, _ in items if t.same}), 'metrics': {}}
        if len(set(labels)) == 2:
            for prior in priors:
                report = eval_report(labels, evidence, prior)
                data['metrics'][str(prior)] = {k: report[k] for k in ('brier', 'log_loss', 'roc_auc', 'prior_only')}
        return data

    overall = summary(rows)
    if not overall['metrics']:
        raise ValueError('the transfer test needs at least one positive and one negative pair')
    intervals = quality_intervals([r[1].same for r in rows], [r[2] for r in rows],
                                  [r[1].left_person for r in rows], [r[1].right_person for r in rows],
                                  priors[0], seed=seed, bootstrap_samples=bootstrap)
    overall['intervals_at_first_prior'] = intervals['intervals_95']
    return {'overall': overall, 'by_community': {c: summary([r for r in rows if r[0] == c])
            for c in sorted({r[0] for r in rows})}, 'inconclusive_reasons': dict(fallback)}, [r[2] for r in rows]


def person_weights(trials, seed, bootstrap):
    """Resample shared trial endpoints together. An author contributes once to a positive pair."""
    people = sorted({p for _, t in trials for p in (t.left_person, t.right_person)})
    index = {p: i for i, p in enumerate(people)}
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(people), [1 / len(people)] * len(people), size=bootstrap)
    a = [index[t.left_person] for _, t in trials]
    b = [index[t.right_person] for _, t in trials]
    same = [t.same for _, t in trials]
    return draws[:, a] * np.where(same, 1, draws[:, b])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('external', type=Path, help='prepared independent external corpus')
    parser.add_argument('--model', nargs=2, metavar=('NAME', 'CHECKPOINT'), action='append', required=True)
    parser.add_argument('--exclude-writing', type=Path, nargs='+', required=True,
                        help='training prepared corpus and linked-owner JSONL, including calibration writing')
    parser.add_argument('--prior', type=float, nargs='+', default=[0.5, 0.1])
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models = {name: Path(path) for name, path in args.model}
    if len(models) != len(args.model):
        parser.error('model names must be unique')
    if args.bootstrap < 0:
        parser.error('bootstrap draws must be nonnegative')
    inputs = [args.external, *models.values(), *args.exclude_writing]
    if args.output.resolve() in {p.resolve() for p in inputs}:
        parser.error('choose a separate output path')
    posts = proxy_posts(load_split(args.external, 'external')[0])
    identities = _identities(posts)
    trials, excluded = transfer_trials(posts, excluded_writing(args.exclude_writing), args.seed)
    result = {'label': 'same observed handle in distinct annotated conversations',
              'controls': 'different handles within the same community and distinct conversations',
              'seed': args.seed, 'bootstrap_by_handle': args.bootstrap,
              'inconclusive_evaluation': 'predict the prior',
              'source_sha256': _source_digest(args.external),
              'excluded_source_sha256': [_source_digest(p) for p in args.exclude_writing],
              'input_posts': len(posts), 'input_handles': len({p.person for p in posts}),
              'input_communities': len({p.community for p in posts}),
              'input_conversations': len({p.thread for p in posts}), 'excluded': excluded, 'models': {}}
    evidence = {}
    for name, path in models.items():
        artifact = load_profile_model(path)
        if identities.intersection(artifact.fit_people):
            raise ValueError('overlapping external and model-fit account identifiers')
        output, evidence[name] = evaluate(artifact, trials, args.prior, args.seed, args.bootstrap)
        result['models'][name] = {'checkpoint_sha256': _source_digest(path),
                                  'feature_version': artifact.provenance['feature_version'], **output}
    if len(models) == 2:
        baseline, candidate = models
        weights = person_weights(trials, args.seed, args.bootstrap)
        labels = [t.same for _, t in trials]
        result['paired_comparison'] = {
            'baseline': baseline, 'candidate': candidate,
            'method': 'Percentile bootstrap resampling handles at both endpoints within the observed communities',
            'note': 'Positive gains favor the candidate. Intervals condition on fitted models and the observed communities.',
            'by_prior': [compare_predictions(labels, evidence[baseline], evidence[candidate], weights, prior)
                         for prior in args.prior],
        }
    write_json(args.output, result)


if __name__ == '__main__':
    main()
