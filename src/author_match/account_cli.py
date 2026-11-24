"""Account-history commands."""

from pathlib import Path

from .accounts import train_accounts, compare_accounts, evaluate_accounts
from .experiment import write_json
from .profile_io import load_profile_model, read_history, save_profile_model


def _prior(value):
    p = float(value)
    if not 0 < p < 1:
        from argparse import ArgumentTypeError
        raise ArgumentTypeError("prior must be strictly between zero and one")
    return p


def add_commands(commands):
    parser = commands.add_parser("train-accounts", help="fit and calibrate an account-history model")
    parser.add_argument("data", type=Path, help="linked-owner JSONL or prepared proxy corpus directory")
    parser.add_argument("model", type=Path, help="output data-only .json[.gz] checkpoint")
    parser.add_argument("--method", choices=("adaptive", "baseline"), default="adaptive",
                        help="adaptive set comparison (default) or v1 pooled baseline")
    parser.add_argument("--pretrain", type=Path, help="fit ranking coefficients on other prepared accounts")
    parser.add_argument("--exclude-proxy-handles", type=Path, help="source account IDs to exclude from pretraining, one per line")
    parser.add_argument("--exclude-proxy-windows", type=Path, help="source post-ID prefixes to exclude, one per line")
    parser.add_argument("--link-evidence", choices=("provided_owner_labels", "same_session_rename", "split_account_history"),
                        help="interpretation of the source labels")
    parser.add_argument("--description", help="short description of where the authorship labels come from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path)
    parser = commands.add_parser("compare-accounts", help="estimate shared authorship across two account histories")
    parser.add_argument("left", type=Path, help="JSON/JSONL export or directory of text files")
    parser.add_argument("right", type=Path, help="second export (may be the same multi-account file)")
    parser.add_argument("--left-handle", help="select this account from the first export")
    parser.add_argument("--right-handle", help="select this account from the second export")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prior", type=_prior,
                        help="estimated same-person frequency for candidate pairs before reading their writing; omit for scenarios")
    parser = commands.add_parser("evaluate-accounts", help="test a frozen account probability model")
    parser.add_argument("data", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--split", choices=("test", "calibration", "external"), default="test")
    parser.add_argument("--prior", type=_prior, default=0.5, help="target prevalence (default: balanced benchmark)")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--output", type=Path)


def run(args):
    if args.command == "train-accounts":
        inputs = {p.resolve() for p in (args.data, args.pretrain, args.exclude_proxy_windows,
                                        args.exclude_proxy_handles) if p}
        if args.model.resolve() in inputs or (args.report and args.report.resolve() in inputs | {args.model.resolve()}):
            raise ValueError("choose separate paths for inputs, model, and report")
        model = train_accounts(args.data, seed=args.seed, pretrain=args.pretrain,
                               exclude_proxy_handles=args.exclude_proxy_handles,
                               exclude_proxy_windows=args.exclude_proxy_windows,
                               description=args.description, link_evidence=args.link_evidence, method=args.method)
        save_profile_model(args.model, model)
        if args.report:
            write_json(args.report, model.provenance)
        return model.provenance
    artifact = load_profile_model(args.model)
    if args.command == "compare-accounts":
        return compare_accounts(artifact, read_history(args.left, args.left_handle),
                                read_history(args.right, args.right_handle), prior=args.prior)
    if args.output and args.output.resolve() in (args.model.resolve(), args.data.resolve()):
        raise ValueError("choose a separate output path")
    result = evaluate_accounts(args.data, artifact, split=args.split, prior=args.prior,
                               bootstrap_samples=args.bootstrap_samples)
    if args.output:
        write_json(args.output, result)
    return result
