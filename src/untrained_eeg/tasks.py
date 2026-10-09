"""The ds003478 tasks and their probe, shared by phase 1 (single models) and phase 2 (ensembles).

Tasks (same probe everywhere: standardize + logistic regression, stratified 5-fold CV):
  ec_eo        eyes closed vs open, one sample per 1-minute block; folds grouped by
               participant, so no participant is in both training and test
  bdi_closed   high vs low BDI, one sample per participant (mean of its eyes-closed blocks)
  bdi_open     the same with eyes-open blocks
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = ("ec_eo", "bdi_closed", "bdi_open")


def task_data(feats, cond, pid, task, labels):
    """(X, y, groups, sample ids) for one task from block features (one row per block)."""
    if task == "ec_eo":
        keep = np.isin(pid, labels.index)  # the same 113 participants as the BDI tasks
        block = pd.Series(pid[keep]).groupby(pid[keep]).cumcount().to_numpy()
        ids = [f"{p}/block{b}" for p, b in zip(pid[keep], block)]
        return feats[keep], (cond[keep] == "closed").astype(int), pid[keep], ids
    condition = task.removeprefix("bdi_")
    people = [p for p in labels.index if ((pid == p) & (cond == condition)).any()]
    X = np.stack([feats[(pid == p) & (cond == condition)].mean(axis=0) for p in people])
    return X, labels[people].to_numpy(), np.array(people), people


def probe(X, y, groups, task, cv_seed, folds):
    """Out-of-fold P(class 1), AUC and balanced accuracy."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if task == "ec_eo":  # blocks: keep each participant's blocks in one fold
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=cv_seed)
    else:  # one sample per participant: no groups needed
        cv, groups = StratifiedKFold(folds, shuffle=True, random_state=cv_seed), None
    prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return prob, roc_auc_score(y, prob), balanced_accuracy_score(y, prob > 0.5)
