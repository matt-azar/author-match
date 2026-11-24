"""Learn evidence from account-pair features and apply a declared match prior."""

from dataclasses import dataclass
from math import expm1, isfinite, log, log1p

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AccountModel:
    """Portable parameters. The calibrated margin estimates log LR for a score."""

    feature_means: list[float]
    feature_scales: list[float]
    coefficients: list[float]
    intercept: float
    calibration_raw_min: float
    calibration_raw_max: float
    calibration_center: float
    calibration_scale: float
    calibration_slope: float
    calibration_intercept: float
    max_abs_log_lr: float
    train_counts: dict[str, int]
    calibration_counts: dict[str, int]
    calibration_status: str
    train_author_count: int | None = None
    calibration_author_count: int | None = None

    def __post_init__(self):
        numeric = (float, int)
        for name in ("feature_means", "feature_scales", "coefficients"):
            vector = getattr(self, name)
            if (not isinstance(vector, list) or not vector
                    or any(type(v) not in numeric or not isfinite(v) for v in vector)):
                raise ValueError(f"{name} must be a nonempty list of finite numeric parameters")
        if len(self.feature_scales) != self.feature_count or len(self.coefficients) != self.feature_count:
            raise ValueError("inconsistent feature dimensions")
        for name in ("intercept", "calibration_raw_min", "calibration_raw_max", "calibration_center",
                     "calibration_scale", "calibration_slope", "calibration_intercept", "max_abs_log_lr"):
            value = getattr(self, name)
            if type(value) not in numeric or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(self.feature_scales) <= 0 or self.calibration_scale <= 0:
            raise ValueError("scaling factors must be positive")
        if (self.calibration_raw_min > self.calibration_raw_max
                or self.calibration_slope < 0 or self.max_abs_log_lr < 0):
            raise ValueError("invalid calibration limits or slope")
        for counts in (self.train_counts, self.calibration_counts):
            if (not isinstance(counts, dict) or set(counts) != {"same", "different"}
                    or any(type(n) is not int or n < 1 for n in counts.values())):
                raise ValueError("same and different pair counts must be positive integers")
        for n in (self.train_author_count, self.calibration_author_count):
            if n is not None and (type(n) is not int or n < 1):
                raise ValueError("author counts must be positive integers when provided")
        max_support = min(self.calibration_counts.values())
        if self.calibration_author_count is not None:
            max_support = min(max_support, self.calibration_author_count)
        if self.max_abs_log_lr > log1p(max_support):
            raise ValueError("evidence cap exceeds the recorded calibration support")
        if self.calibration_status not in ("fitted", "no_score_variation", "nonpositive_association"):
            raise ValueError("unknown calibration status")
        if self.calibration_status != "fitted" and (self.calibration_slope or self.calibration_intercept):
            raise ValueError("a neutral calibrator must have zero slope and intercept")

    @property
    def effective_calibration_size(self):
        """Recover the effective class size recorded by the fitted evidence cap."""
        return round(expm1(self.max_abs_log_lr))

    @property
    def feature_count(self):
        return len(self.feature_means)


def _matrix(values, width=None):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0 or (width is not None and x.shape[1] != width):
        raise ValueError("expected a two-dimensional matrix with the model's feature width")
    if not np.isfinite(x).all():
        raise ValueError("profile features must be finite")
    return x


def _labels(values, size):
    y = np.asarray(values)
    if y.shape != (size,) or not np.isin(y, (0, 1)).all():
        raise ValueError("each pair needs one binary label")
    return y.astype(bool)


def _counts(y):
    return {"same": int(y.sum()), "different": int((~y).sum())}


def _prior(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("prior must be a number strictly between zero and one")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError("prior must be a number strictly between zero and one")
    return value


def _calibration_fit(raw, y, counts, effective):
    # Equal total weight for the two classes gives calibrated log odds under
    # an equal prior. Those log odds estimate the score's log likelihood ratio.
    weights = np.where(y, effective / counts["same"], effective / counts["different"])
    center = float(np.average(raw, weights=weights))
    scale = float(np.sqrt(np.average((raw - center) ** 2, weights=weights)))
    max_abs_log_lr = log1p(effective)
    if scale <= np.finfo(float).eps * max(1.0, abs(center)):
        return center, 1.0, 0.0, 0.0, max_abs_log_lr, "no_score_variation"
    x = (raw - center) / scale
    if x[y].mean() <= x[~y].mean():
        # A model whose margin reverses on independent authors supplies no
        # validated directional evidence. Zero log LR leaves the prior intact.
        return center, scale, 0.0, 0.0, max_abs_log_lr, "nonpositive_association"

    # Smoothed targets prevent an arbitrarily large margin even for perfectly
    # separated finite calibration samples. A symmetric target preserves the
    # equal prior when the two classes have different sample sizes.
    shrink = 1 / (effective + 2)
    target = np.where(y, 1 - shrink, shrink)

    def objective(theta):
        slope, intercept = theta
        z = slope * x + intercept
        loss = np.dot(weights, np.logaddexp(0.0, z) - target * z) + 0.5 * slope ** 2
        residual = weights * (expit(z) - target)
        return loss, np.array([residual @ x + slope, residual.sum()])

    fit = minimize(objective, [0.0, 0.0], method="L-BFGS-B", jac=True,
                   bounds=[(0.0, None), (None, None)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("score calibration failed to converge")
    return center, scale, float(fit.x[0]), float(fit.x[1]), max_abs_log_lr, "fitted"


def fit_account_model(train_X, train_y, cal_X, cal_y, *, train_authors=None, calibration_authors=None,
                      calibration_effective_people=None):
    """Fit a ranking classifier and then calibrate its score on held-out authors.

    The caller constructs independent author partitions before forming pairs.
    Passing both sets of participant IDs also checks their separation and limits
    calibration strength by the number of distinct calibration participants.
    Pass the smaller of the two classes' distinct owner counts as
    calibration_effective_people when people recur across calibration trials.
    """
    features = _matrix(train_X)
    xcal = _matrix(cal_X, features.shape[1])
    y, ycal = _labels(train_y, len(features)), _labels(cal_y, len(xcal))
    train_counts, cal_counts = _counts(y), _counts(ycal)
    if min(*train_counts.values(), *cal_counts.values()) < 1:
        raise ValueError("training and calibration each require both pair classes")
    n_train = n_cal = None
    if (train_authors is None) != (calibration_authors is None):
        raise ValueError("supply both training and calibration author sets")
    if train_authors is not None:
        train_ids, cal_ids = set(train_authors), set(calibration_authors)
        if not train_ids or not cal_ids or train_ids.intersection(cal_ids):
            raise ValueError("training and calibration authors must be nonempty and disjoint")
        n_train, n_cal = len(train_ids), len(cal_ids)

    effective = min(cal_counts.values())
    if n_cal is not None:
        effective = min(effective, n_cal)
    if calibration_effective_people is not None:
        if (isinstance(calibration_effective_people, (bool, np.bool_))
                or not isinstance(calibration_effective_people, (int, np.integer))
                or calibration_effective_people < 1):
            raise ValueError("calibration_effective_people must be a positive integer")
        effective = min(effective, int(calibration_effective_people))

    scaler = StandardScaler().fit(features)
    estimator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                   solver="lbfgs").fit(scaler.transform(features), y)
    raw = estimator.decision_function(scaler.transform(xcal))
    center, scale, slope, bias, bound, status = _calibration_fit(raw, ycal, cal_counts, effective)
    return AccountModel(
        feature_means=scaler.mean_.tolist(), feature_scales=scaler.scale_.tolist(),
        coefficients=estimator.coef_[0].tolist(), intercept=float(estimator.intercept_[0]),
        calibration_raw_min=float(raw.min()), calibration_raw_max=float(raw.max()),
        calibration_center=center, calibration_scale=scale,
        calibration_slope=slope, calibration_intercept=bias, max_abs_log_lr=bound,
        train_counts=train_counts, calibration_counts=cal_counts, calibration_status=status,
        train_author_count=n_train, calibration_author_count=n_cal,
    )


def _predict(model, x):
    features = _matrix(x, model.feature_count)
    raw = ((features - model.feature_means) / model.feature_scales) @ model.coefficients + model.intercept
    if not np.isfinite(raw).all():
        raise ValueError("feature magnitudes exceed the model's numeric range")
    supported = np.clip(raw, model.calibration_raw_min, model.calibration_raw_max)
    z = model.calibration_slope * ((supported - model.calibration_center) / model.calibration_scale)
    evidence = np.clip(z + model.calibration_intercept, -model.max_abs_log_lr, model.max_abs_log_lr)
    return evidence, raw, supported


def predict_log_lr(model, X):
    """Vectorized log likelihood ratios, bounded to the observed calibration margin range."""
    return _predict(model, X)[0]


def posterior(log_likelihood_ratio, prior):
    """Apply Bayes' odds rule with a caller-supplied same-account prior."""
    prior = _prior(prior)
    x = np.asarray(log_likelihood_ratio, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("log likelihood ratios must be finite")
    result = np.where(x == 0, prior, expit(x + logit(prior)))
    return float(result) if result.ndim == 0 else result


def score(model, features, prior=0.5):
    """Return one conditional match probability and its class-prior-independent evidence."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 1:
        raise ValueError("one account comparison requires one feature vector")
    evidence, raw, supported = _predict(model, x[None, :])
    log_lr = float(evidence[0])
    return {
        "probability": posterior(log_lr, prior),
        "prior": float(prior),
        "log_likelihood_ratio": log_lr,
        "extrapolated": bool(raw[0] != supported[0]),
    }


def _labeled_evidence(labels, log_lrs):
    evidence = np.asarray(log_lrs, dtype=float)
    if evidence.ndim != 1 or not np.isfinite(evidence).all():
        raise ValueError("expected a vector of finite log likelihood ratios")
    return evidence, _labels(labels, len(evidence))


def eval_report(labels, log_likelihood_ratios, prior=0.5, bins=10):
    """Evaluate under a target prior by reweighting each true class to its prevalence."""
    prior = _prior(prior)
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    evidence, y = _labeled_evidence(labels, log_likelihood_ratios)
    counts = _counts(y)
    if min(counts.values()) < 1:
        raise ValueError("target-prior evaluation requires both pair classes")
    weights = np.where(y, prior / counts["same"], (1 - prior) / counts["different"])
    prob = posterior(evidence, prior)
    log_odds = evidence + logit(prior)
    log_loss = np.dot(weights, np.logaddexp(0, np.where(y, -log_odds, log_odds)))
    bin_ids = np.minimum((prob * bins).astype(int), bins - 1)
    reliability = []
    calibration_error = 0.0
    for i in range(bins):
        group = bin_ids == i
        mass = float(weights[group].sum())
        predicted = observed = None
        if mass:
            predicted = float(np.average(prob[group], weights=weights[group]))
            observed = float(np.average(y[group], weights=weights[group]))
            calibration_error += mass * abs(observed - predicted)
        reliability.append({
            "lower": i / bins, "upper": (i + 1) / bins, "pairs": int(group.sum()),
            **_counts(y[group]), "target_weight": mass,
            "mean_probability": predicted, "observed_same_fraction": observed,
        })
    return {
        "prior": prior, "pairs": len(evidence), **counts,
        "roc_auc": float(roc_auc_score(y, evidence)),
        "brier": float(np.dot(weights, (prob - y) ** 2)),
        "log_loss": float(log_loss),
        "prior_only": {
            "brier": prior * (1 - prior),
            "log_loss": -prior * log(prior) - (1 - prior) * log1p(-prior),
        },
        "expected_calibration_error": calibration_error,
        "reliability": reliability,
    }


def quality_intervals(labels, log_lrs, left_people, right_people, prior=0.5, *, seed=42, bootstrap_samples=300):
    """Approximate holdout-person uncertainty conditional on a frozen model.

    Both endpoints of a different-person pair must occur in the resample.
    For a same-person pair the person contributes one multiplicity, not its
    square. Reweight each resampled class to the requested prior.
    """
    evidence, y = _labeled_evidence(labels, log_lrs)
    left, right = np.asarray(left_people), np.asarray(right_people)
    if left.shape != y.shape or right.shape != y.shape or np.any((left == right) != y):
        raise ValueError("each trial needs two person identifiers consistent with its label")
    people, ids = np.unique(np.concatenate((left, right)), return_inverse=True)
    a, b = np.split(ids, 2)

    def trial_counts(multiplicity):
        return multiplicity[a] * np.where(y, 1, multiplicity[b])

    return _quality_bootstrap(y, evidence, prior, seed, bootstrap_samples, "people", len(people),
                              "Percentile bootstrap resampling both trial endpoints by person", trial_counts)


def window_quality_intervals(labels, log_lrs, window_ids, prior=0.5, *, seed=42, bootstrap_samples=300):
    """Resample whole held-out recording windows and retain all their trials together."""
    evidence, y = _labeled_evidence(labels, log_lrs)
    windows = list(window_ids)
    if len(windows) != len(y):
        raise ValueError("each trial needs one window identifier")
    for value in windows:
        if isinstance(value, str) and value.strip():
            continue
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            continue
        raise ValueError("a window identifier must be a nonempty string or integer")
    unique = sorted(set(windows), key=lambda w: (isinstance(w, str), w))
    index = {w: i for i, w in enumerate(unique)}
    group_for_trial = np.array([index[w] for w in windows], dtype=np.intp)

    def trial_counts(multiplicity):
        return multiplicity[group_for_trial]

    return _quality_bootstrap(y, evidence, prior, seed, bootstrap_samples, "windows", len(unique),
                              "Percentile bootstrap resampling complete study windows", trial_counts)


def _quality_bootstrap(y, evidence, prior, seed, bootstrap_samples, unit, group_count, method, trial_counts):
    prior = _prior(prior)
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a non-negative integer")
    result = {
        "method": method,
        "note": (
            f"Approximate 95% intervals conditional on the fitted model, observed {unit} and trial design. "
            "Each draw uses the declared prior. Positive improvement means lower loss than predicting "
            "the prior for every pair. Degenerate intervals and intervals with too few usable draws are omitted."
        ),
        "seed": seed, "prior": prior, unit: group_count, "pairs": len(y),
        "repetitions": bootstrap_samples, "usable_repetitions": 0, "intervals_95": {},
    }
    if bootstrap_samples < 2 or group_count < 2 or not y.any() or y.all():
        return result

    probs = posterior(evidence, prior)
    squared_errors = (probs - y) ** 2
    log_losses = np.logaddexp(0, np.where(y, -1, 1) * (evidence + logit(prior)))
    baseline_brier = prior * (1 - prior)
    baseline_log_loss = -prior * log(prior) - (1 - prior) * log1p(-prior)
    # The rank groups stay fixed. Accumulate tie-correct AUC in linear time
    # per resample rather than sorting a new weighted ROC curve on every draw.
    levels, group = np.unique(evidence, return_inverse=True)
    pos_groups, neg_groups = group[y], group[~y]
    estimates = {"brier_improvement": [], "log_loss_improvement": [], "roc_auc": []}
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_samples):
        draw = rng.integers(group_count, size=group_count)
        counts = trial_counts(np.bincount(draw, minlength=group_count))
        pos, neg = counts[y], counts[~y]
        n_pos, n_neg = pos.sum(), neg.sum()
        if n_pos == 0 or n_neg == 0:
            continue
        weights = counts * np.where(y, prior / n_pos, (1 - prior) / n_neg)
        estimates["brier_improvement"].append(baseline_brier - weights @ squared_errors)
        estimates["log_loss_improvement"].append(baseline_log_loss - weights @ log_losses)
        wp = np.bincount(pos_groups, weights=pos, minlength=len(levels))
        wn = np.bincount(neg_groups, weights=neg, minlength=len(levels))
        estimates["roc_auc"].append(float(wp @ (np.cumsum(wn) - 0.5 * wn)) / (float(n_pos) * n_neg))

    usable = len(estimates["roc_auc"])
    result["usable_repetitions"] = usable
    if usable >= max(2, bootstrap_samples * 0.8):
        for name, values in estimates.items():
            lo, hi = np.quantile(values, [0.025, 0.975])
            if not np.isclose(lo, hi, rtol=1e-12, atol=1e-14):
                result["intervals_95"][name] = [float(lo), float(hi)]
    return result
