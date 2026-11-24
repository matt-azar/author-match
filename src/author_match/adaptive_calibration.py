"""Continuous, writer-balanced score calibration across unequal account histories."""

from __future__ import annotations

from math import isfinite, log1p

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .profile_probability import posterior


def _sizes(values):
    a = np.asarray(values)
    if a.ndim != 2 or a.shape[1] != 4 or a.dtype.kind not in "iu" or (a < 1).any():
        raise ValueError("history sizes must contain four positive integer counts per pair")
    # Axes are trial, endpoint, measurement (posts or words).
    return np.stack((a[:, :2], a[:, 2:]), axis=2)


def _rows(values):
    a = np.asarray(values)
    if (a.ndim != 2 or a.shape[1] != 7 or a.dtype.kind not in "iu"
            or not len(a) or (a[:, 5:] < 0).any()):
        raise ValueError("expected calibration rows with sizes, label and two integer writer identifiers")
    sizes = _sizes(a[:, :4])
    y, people = a[:, 4], a[:, 5:]
    if set(y.tolist()) != {0, 1} or np.any((people[:, 0] == people[:, 1]) != y):
        raise ValueError("calibration requires both classes and consistent writer identifiers")
    return sizes, y.astype(bool), people


def _contributions(y, people):
    """Budget one unit of calibration weight per writer in each class."""
    pos, pos_inverse, pos_counts = np.unique(people[y, 0], return_inverse=True, return_counts=True)
    # Count each undirected comparison once. Split a writer's budget across
    # distinct partners; either endpoint can be the more heavily reused one.
    edges, neg_inverse, edge_counts = np.unique(np.sort(people[~y], axis=1), axis=0,
                                                return_inverse=True, return_counts=True)
    neg_inverse = neg_inverse.reshape(-1)
    _, endpoint_ids, degrees = np.unique(edges.ravel(), return_inverse=True, return_counts=True)
    endpoint_ids = endpoint_ids.reshape(-1, 2)
    edge_weight = 1. / degrees[endpoint_ids].max(axis=1)
    negative_mass = edge_weight.sum()
    effective = min(len(pos), negative_mass)
    w = np.empty(len(y))
    w[y] = effective / (len(pos) * pos_counts[pos_inverse])
    w[~y] = effective * edge_weight[neg_inverse] / (negative_mass * edge_counts[neg_inverse])
    return w, effective, pos_inverse, neg_inverse, edge_weight, endpoint_ids


def _noise(sizes, bandwidth):
    # Sampling variation in rates decreases approximately as 1/sqrt(n). This
    # compresses large histories while distinguishing small changes in short ones.
    return 1 / (np.sqrt(sizes) * bandwidth)


def _precision(noise):
    return 1 / (1 + np.mean(noise ** 2, axis=(1, 2)))


def fit_adaptive_calibration(raw_scores, support_rows):
    """Fit only on a writer-disjoint calibration set after freezing the ranker.

    Rows use [posts A, posts B, words A, words B, same, writer A, writer B].
    Returns a portable dict suitable for AdaptiveCalibration.from_dict().
    """
    sizes, y, people = _rows(support_rows)
    raw = np.asarray(raw_scores, dtype=float)
    if raw.shape != y.shape or not np.isfinite(raw).all():
        raise ValueError("each calibration pair needs one finite ranking score")

    weights, effective, *_ = _contributions(y, people)
    # Class- and writer-balanced reference scales. Duplicating trials a uniform
    # number of times cannot increase evidence or change the calibration map.
    bandwidth = np.average((1 / np.sqrt(sizes)).mean(axis=1), axis=0, weights=weights)
    center = np.average(raw, weights=weights)
    scale = np.sqrt(np.average((raw - center) ** 2, weights=weights))
    lower, upper = float(raw.min()), float(raw.max())
    theta = np.zeros(3)
    status = "fitted"
    if scale <= np.finfo(float).eps * max(1, abs(center)):
        scale, status = 1.0, "no_score_variation"
    elif np.average(raw[y], weights=weights[y]) <= np.average(raw[~y], weights=weights[~y]):
        status = "nonpositive_association"
    else:
        # Interpolate the slopes learned for low and high sampling precision.
        # Nonnegative endpoint slopes preserve monotonicity at every quantity.
        x = (raw - center) / scale
        precision = _precision(_noise(sizes, bandwidth))
        design = np.column_stack((x * (1 - precision), x * precision, np.ones(len(x))))
        smoothing = 1 / (effective + 2)
        target = np.where(y, 1 - smoothing, smoothing)
        # Penalize the average slope and deviations from a common slope.
        # A small calibration sample therefore favors one global relationship.
        penalty = np.array([[1.25, -0.75, 0.], [-0.75, 1.25, 0.], [0., 0., 0.]])

        def objective(parameters):
            z = design @ parameters
            cost = weights @ (np.logaddexp(0., z) - target * z) + 0.5 * parameters @ penalty @ parameters
            gradient = design.T @ (weights * (expit(z) - target)) + penalty @ parameters
            return cost, gradient

        result = minimize(objective, theta, jac=True, method="L-BFGS-B",
                          bounds=[(0., None), (0., None), (None, None)])
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("adaptive score calibration failed to converge")
        theta = result.x
    return {
        "version": 1, "status": status, "rows": np.asarray(support_rows).tolist(),
        "bandwidth": bandwidth.tolist(), "center": float(center), "scale": float(scale),
        "raw_min": lower, "raw_max": upper, "slopes": theta[:2].tolist(), "intercept": float(theta[2]),
    }


class AdaptiveCalibration:
    """Apply a monotone held-out score map with smoothly varying writer support."""

    @classmethod
    def from_dict(cls, payload):
        return cls(payload)

    def __init__(self, payload):
        fields = {"version", "status", "rows", "bandwidth", "center", "scale", "raw_min",
                  "raw_max", "slopes", "intercept"}
        if (not isinstance(payload, dict) or set(payload) != fields
                or type(payload["version"]) is not int or payload["version"] != 1
                or payload["status"] not in ("fitted", "no_score_variation", "nonpositive_association")):
            raise ValueError("invalid adaptive calibration format")
        self.status = payload["status"]
        for name in ("center", "scale", "raw_min", "raw_max", "intercept"):
            value = payload[name]
            if type(value) not in (int, float) or not isfinite(value):
                raise ValueError(f"invalid adaptive calibration parameter {name}")
            setattr(self, name, value)
        for name in ("bandwidth", "slopes"):
            value = payload[name]
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(v) not in (int, float) or not isfinite(v) for v in value)):
                raise ValueError(f"invalid adaptive calibration vector {name}")
            setattr(self, name, np.asarray(value))
        if (self.scale <= 0 or self.raw_min > self.raw_max or np.min(self.bandwidth) <= 0
                or np.min(self.slopes) < 0 or (self.status != "fitted" and
                                               (self.intercept != 0 or np.any(self.slopes != 0)))):
            raise ValueError("inconsistent adaptive calibration parameters")
        sizes, self._y, people = _rows(payload["rows"])
        self._reference = _noise(sizes, self.bandwidth)
        (_, _, self._pos_ids, self._neg_ids, self._edge_weights,
         self._edge_endpoints) = _contributions(self._y, people)
        self._npos = int(self._pos_ids.max()) + 1
        self._nneg = int(self._edge_endpoints.max()) + 1

    def _support(self, noise):
        # Minimum assignment cost makes the neighborhood invariant to swapping
        # accounts. For each writer count only their closest calibration pair,
        # so repeated trials by one person cannot manufacture local support.
        forward = np.sum((self._reference - noise) ** 2, axis=(1, 2))
        reverse = np.sum((self._reference - noise[::-1]) ** 2, axis=(1, 2))
        kernel = np.exp(-0.5 * np.minimum(forward, reverse))
        pos = np.zeros(self._npos)
        edges = np.zeros(len(self._edge_weights))
        neg = np.zeros(self._nneg)
        np.maximum.at(pos, self._pos_ids, kernel[self._y])
        np.maximum.at(edges, self._neg_ids, kernel[~self._y])
        np.maximum.at(neg, self._edge_endpoints.ravel(), np.repeat(edges, 2))
        return float(pos.sum()), float(neg.sum()), float(edges @ self._edge_weights)

    def estimate(self, raw_score, sizes, prior):
        """Return a posterior for a declared prior plus local support diagnostics."""
        if (isinstance(raw_score, (bool, np.bool_)) or not isinstance(raw_score, (int, float, np.number))
                or not np.isrealobj(raw_score) or not isfinite(raw_score)):
            raise ValueError("expected one finite ranking score")
        noise = _noise(_sizes([sizes]), self.bandwidth)
        precision = float(_precision(noise)[0])
        positive, negative, negative_pairs = self._support(noise[0])
        effective = min(positive, negative_pairs)
        # One neutral pseudo-observation regularizes the log evidence in an
        # unfamiliar neighborhood; the finite-writer bound also tends to zero.
        weight = effective / (1 + effective)
        bound = log1p(effective)
        supported = np.clip(raw_score, self.raw_min, self.raw_max)
        slope = (1 - precision) * self.slopes[0] + precision * self.slopes[1]
        z = slope * ((supported - self.center) / self.scale) + self.intercept
        log_lr = float(np.clip(z * weight, -bound, bound))
        return {
            "probability": posterior(log_lr, prior), "log_likelihood_ratio": log_lr,
            "extrapolated": bool(raw_score != supported), "support_weight": weight,
            "effective_calibration_pairs": effective, "positive_writer_support": positive,
            "negative_writer_support": negative, "sampling_precision": precision,
        }
