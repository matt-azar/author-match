"""A linear stylometry score combined with shallow, learned feature interactions."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .profile_probability import _labels, _matrix


@dataclass
class AdaptiveRanker:
    # Coefficients already include standardization and the equal blend weight.
    coefficients: list
    intercept: float
    trees: list

    def __post_init__(self):
        def finite(v):
            return type(v) in (float, int) and isfinite(v)
        if (not isinstance(self.coefficients, list) or not self.coefficients
                or not all(finite(v) for v in self.coefficients) or not finite(self.intercept)):
            raise ValueError("invalid adaptive ranker coefficients")
        if not isinstance(self.trees, list) or not self.trees:
            raise ValueError("adaptive ranker requires fitted trees")
        for tree in self.trees:
            if not isinstance(tree, list) or not 1 <= len(tree) <= 7:
                raise ValueError("invalid depth-two ranking tree")
            reachable = {0}
            for i, node in enumerate(tree):
                if i not in reachable or not isinstance(node, list):
                    raise ValueError("invalid ranking tree topology")
                if len(node) == 1 and finite(node[0]):
                    continue
                if len(node) != 4:
                    raise ValueError("invalid ranking tree node")
                feature, threshold, left, right = node
                if (type(feature) is not int or not 0 <= feature < self.feature_count
                        or not finite(threshold) or type(left) is not int or type(right) is not int
                        or not i < left < right < len(tree) or left in reachable or right in reachable):
                    raise ValueError("invalid ranking tree split")
                reachable.update((left, right))

    @property
    def feature_count(self):
        return len(self.coefficients)

    def margins(self, features):
        x = _matrix(features, self.feature_count)
        scores = x @ self.coefficients + self.intercept
        # sklearn evaluates tree thresholds on float32 inputs.
        tree_x = x.astype(np.float32)
        def add(tree, index, rows):
            node = tree[index]
            if len(node) == 1:
                scores[rows] += node[0]
            else:
                col, cut, left, right = node
                mask = tree_x[rows, col].astype(np.float64) <= cut
                if mask.any():
                    add(tree, left, rows[mask])
                if not mask.all():
                    add(tree, right, rows[~mask])
        rows = np.arange(len(x))
        for tree in self.trees:
            add(tree, 0, rows)
        return scores

    def margin(self, features):
        x = np.asarray(features, dtype=float)
        if x.shape != (self.feature_count,) or not np.isfinite(x).all():
            raise ValueError("one comparison needs one finite feature vector")
        score = float(x @ self.coefficients + self.intercept)
        tree_x = x.astype(np.float32)
        for tree in self.trees:
            node = tree[0]
            while len(node) == 4:
                col, cut, left, right = node
                node = tree[left if float(tree_x[col]) <= cut else right]
            score += node[0]
        return score


def fit_ranker(features, labels, seed):
    """Fit only on training writers. Neither branch sees calibration outcomes."""
    x = _matrix(features)
    y = _labels(labels, len(x))
    if len(np.unique(y)) != 2:
        raise ValueError("ranking requires both pair classes")
    scaler = StandardScaler().fit(x)
    linear = LogisticRegression(C=0.5, class_weight="balanced", max_iter=600).fit(scaler.transform(x), y)
    beta = 0.5 * linear.coef_[0] / scaler.scale_
    bias = 0.5 * linear.intercept_[0] - beta @ scaler.mean_
    # A zero initial tree margin expresses equal class odds. Depth two allows
    # recurring habits to interact, without memorizing high-order combinations.
    boosted = GradientBoostingClassifier(
        n_estimators=64, learning_rate=0.06, max_depth=2, min_samples_leaf=max(2, min(32, len(y) // 40)),
        init="zero", random_state=seed,
    ).fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    trees = []
    for estimator in boosted.estimators_[:, 0]:
        tree = estimator.tree_
        nodes = []
        for i, child in enumerate(tree.children_left):
            if child < 0:
                nodes.append([float(0.5 * boosted.learning_rate * tree.value[i, 0, 0])])
            else:
                nodes.append([int(tree.feature[i]), float(tree.threshold[i]), int(child),
                              int(tree.children_right[i])])
        trees.append(nodes)
    return AdaptiveRanker(beta.tolist(), float(bias), trees)
