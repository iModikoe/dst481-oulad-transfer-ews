"""Source-target pair construction with person-level disjointness (§3.8.1).

A student may hold several registrations, so without an explicit rule the same
person can appear in both source and target. Any performance arising from that
partly reflects prior exposure to the individual rather than transfer across
contexts. Every pair built here removes, from the *training* side, all
registrations belonging to students who also appear in the target.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Pair:
    tier: int
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    n_train_before_rule: int
    n_removed_overlap: int
    differs_by: str
    extra: dict = field(default_factory=dict)

    @property
    def pct_removed(self) -> float:
        if self.n_train_before_rule == 0:
            return 0.0
        return 100.0 * self.n_removed_overlap / self.n_train_before_rule


def _drop_overlap(frame, train_mask, test_mask, enforce=True):
    """Remove training rows whose student also appears in the target."""
    before = int(train_mask.sum())
    if not enforce:
        return train_mask, before, 0
    target_students = set(frame.loc[test_mask, "id_student"].unique())
    keep = train_mask & ~frame["id_student"].isin(target_students)
    return keep, before, before - int(keep.sum())


def tier2_pairs(frame: pd.DataFrame, enforce_disjoint: bool = True) -> list[Pair]:
    """Train on 2013 presentations, test on 2014 presentations of the same module."""
    pairs = []
    for module, g in frame.groupby("code_module", observed=True):
        years = set(g["year"].unique())
        if not ({2013, 2014} <= years):
            continue
        train_mask = (frame["code_module"] == module) & (frame["year"] == 2013)
        test_mask = (frame["code_module"] == module) & (frame["year"] == 2014)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        keep, before, removed = _drop_overlap(frame, train_mask, test_mask, enforce_disjoint)
        if keep.sum() < 50 or frame.loc[test_mask, "y"].nunique() < 2:
            continue
        pairs.append(Pair(2, f"{module} 2013->2014",
                          np.flatnonzero(keep.to_numpy()),
                          np.flatnonzero(test_mask.to_numpy()),
                          before, removed, "presentation",
                          {"module": module}))
    return pairs


def tier3_pairs(frame: pd.DataFrame, enforce_disjoint: bool = True) -> list[Pair]:
    """Leave-one-module-out: train on all presentations of the other modules."""
    pairs = []
    modules = sorted(frame["code_module"].unique())
    for held in modules:
        test_mask = frame["code_module"] == held
        train_mask = ~test_mask
        if test_mask.sum() == 0:
            continue
        keep, before, removed = _drop_overlap(frame, train_mask, test_mask, enforce_disjoint)
        if keep.sum() < 50 or frame.loc[test_mask, "y"].nunique() < 2:
            continue
        pairs.append(Pair(3, f"six modules -> {held}",
                          np.flatnonzero(keep.to_numpy()),
                          np.flatnonzero(test_mask.to_numpy()),
                          before, removed, "module",
                          {"held_out_module": held}))
    return pairs


def tier1_folds(frame: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Student-grouped stratified folds within the pooled data (§3.8.1).

    Grouping by student prevents the same person appearing on both sides of a
    fold, which is the within-tier analogue of the person-disjointness rule.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    y = frame["y"].to_numpy()
    groups = frame["id_student"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))
