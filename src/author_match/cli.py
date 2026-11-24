"""Command-line interface to author-match."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .data import load_split, prepare
from .experiment import bundle_policy, compare, evaluate, load_model, save_model, train, write_json


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def rate(value):
    number = float(value)
    if not 0 < number < 0.5:
        raise argparse.ArgumentTypeError("must be greater than 0 and less than 0.5")
    return number


def neural_args(parser):
    parser.add_argument("--checkpoint", help="local STAR-compatible checkpoint directory")
    parser.add_argument("--device", default="cpu", help="e.g. cpu, cuda, mps")
    parser.add_argument("--offline", action="store_true", help="use only local checkpoint files")


def parser():
    root = argparse.ArgumentParser(prog="author-match", description=__doc__)
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="split original posts and sample pairs")
    p.add_argument("input", type=Path, help="normalized or ConvoKit JSONL[.gz], or corpus directory")
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pairs-per-author", type=positive, default=12)
    p.add_argument("--external", action="store_true", help="prepare an independent evaluation corpus in full")
    p.add_argument("--holdout-community", action="append", default=[], help="reserve this community for test (repeatable)")
    p.add_argument("--exclude-account", action="append", default=[], help="omit a known bot or shared account (repeatable)")
    p = commands.add_parser("train", help="select on development authors and calibrate the winner")
    p.add_argument("data", type=Path)
    p.add_argument("model", type=Path, help="output .json[.gz] for CPU or trusted local .joblib for any method")
    p.add_argument("--method", choices=("auto", "word", "char", "style", "star"), default="auto",
                   help="auto compares word, char, and style")
    p.add_argument("--bundle-mode", choices=("1x5", "5x5"), action="append", default=[],
                   help="also calibrate this multi-post mode on reserved writers (repeatable)")
    p.add_argument("--include-star", action="store_true", help="include STAR in auto selection")
    p.add_argument("--false-match-rate", type=rate, default=0.01)
    p.add_argument("--false-reject-rate", type=rate, default=0.05)
    p.add_argument("--report", type=Path, help="save the model selection and calibration summary")
    neural_args(p)
    p = commands.add_parser("evaluate", help="report errors and abstentions at frozen thresholds")
    p.add_argument("data", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("--split", choices=("dev", "calibration", "test", "external"), default="dev")
    p.add_argument("--mode", choices=("1x1", "1x5", "5x5"), default="1x1")
    p.add_argument("--output", type=Path)
    p.add_argument("--bootstrap-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("compare", help="compare two UTF-8 text files using a local model")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p = commands.add_parser("compare-bundles", help="compare two sets of independent posts")
    p.add_argument("--left", type=Path, nargs="+", required=True)
    p.add_argument("--right", type=Path, nargs="+", required=True)
    p.add_argument("--model", type=Path, required=True)
    p = commands.add_parser("finetune", help="fine-tune a STAR encoder using training authors only")
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path, help="output encoder directory")
    p.add_argument("--steps", type=positive, default=100)
    p.add_argument("--authors-per-batch", type=positive, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    neural_args(p)
    return root


def run(args):
    if args.command == "prepare":
        return prepare(args.input, args.output, seed=args.seed, pairs_per_author=args.pairs_per_author,
                       holdout_communities=args.holdout_community, excluded_accounts=args.exclude_account,
                       external=args.external)
    if args.command == "train":
        if args.include_star and args.method != "auto":
            raise ValueError("--include-star requires --method auto")
        if args.report and args.report.resolve() == args.model.resolve():
            raise ValueError("choose separate paths for the model and report")
        methods = ["word", "char", "style"] if args.method == "auto" else [args.method]
        if args.include_star:
            methods.append("star")
        artifact = train(args.data, methods, false_match_rate=args.false_match_rate,
                         false_reject_rate=args.false_reject_rate, bundle_modes=args.bundle_mode,
                         neural_options={"checkpoint": args.checkpoint, "device": args.device,
                                         "local_files_only": args.offline})
        save_model(args.model, artifact)
        if args.report:
            write_json(args.report, artifact.provenance)
        return artifact.provenance
    if args.command == "evaluate":
        if args.bootstrap_samples < 0:
            raise ValueError("--bootstrap-samples must be nonnegative")
        if args.output and args.output.resolve() == args.model.resolve():
            raise ValueError("choose separate paths for the model and report")
        result = evaluate(args.data, args.split, load_model(args.model),
                          bootstrap_samples=args.bootstrap_samples, seed=args.seed, mode=args.mode)
        if args.output:
            write_json(args.output, result)
        return result
    if args.command == "compare":
        return compare(load_model(args.model), args.left.read_text(encoding="utf-8"),
                       args.right.read_text(encoding="utf-8"))
    if args.command == "compare-bundles":
        from .bundles import compare_bundles
        mode = "x".join(map(str, sorted([len(args.left), len(args.right)])))
        artifact = load_model(args.model)
        return compare_bundles(artifact.verifier, [p.read_text(encoding="utf-8") for p in args.left],
                               [p.read_text(encoding="utf-8") for p in args.right], bundle_policy(artifact, mode))
    from .neural import fine_tune_star
    posts, _ = load_split(args.data, "train")
    losses = fine_tune_star(posts, args.output, checkpoint=args.checkpoint,
                            device=args.device, steps=args.steps, authors_per_batch=args.authors_per_batch,
                            learning_rate=args.learning_rate, seed=args.seed, local_files_only=args.offline)
    return {"checkpoint": str(args.output), "steps": len(losses), "last_loss": losses[-1]}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"author-match: {error}", file=sys.stderr)
        return 2
