"""Phase 1, step 2: probe every instance's features on the ds003478 tasks.

Tasks (same probe everywhere: standardize + logistic regression, stratified 5-fold CV):
  ec_eo        eyes closed vs open, one sample per 1-minute block; folds grouped by
               participant, so no participant is in both training and test
  bdi_closed   high vs low BDI, one sample per participant (mean of its eyes-closed blocks)
  bdi_open     the same with eyes-open blocks

Every instance (seed or checkpoint) is probed with the config's CV split. The first
instance of each extractor is also probed with --cv-seeds, the CV-split variability.

Reads results/phase1/ds003478/features (phase1_features.py). Writes, next to it:
  metrics.csv            extractor, group, instance, task, cv_seed, n, auc, balanced_accuracy
  predictions.csv.gz     out-of-fold P(class 1) per sample, for the ensembles of phase 2

    python scripts/phase1_probe.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config, load_participants

TASKS = ("ec_eo", "bdi_closed", "bdi_open")


def task_data(f, task, labels):
    """(X, y, groups, sample ids) for one task from an instance's block features."""
    feats, cond, pid = f["feats"], f["cond"], f["participant_id"]
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
    if task == "ec_eo":
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=cv_seed)
    else:
        cv = StratifiedKFold(folds, shuffle=True, random_state=cv_seed)
    prob = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    return prob, roc_auc_score(y, prob), balanced_accuracy_score(y, prob > 0.5)


def seed_range(text):
    a, _, b = text.partition("-")
    return list(range(int(a), int(b or a) + 1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--cv-seeds", type=seed_range, default=list(range(30)),
                    help="CV splits for the first instance of each extractor, e.g. 0-29")
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
root = Path(cfg["results_dir"]) / "phase1" / "ds003478"
labels = load_participants(cfg).set_index("participant_id")["label"]
metrics, preds = [], []
for ext_dir in sorted(p for p in (root / "features").iterdir() if p.is_dir()):
    for i, path in enumerate(sorted(ext_dir.glob("*.npz"))):
        f = np.load(path)
        cv_seeds = args.cv_seeds if i == 0 else [cfg["cv_seed"]]
        for task in TASKS:
            X, y, groups, ids = task_data(f, task, labels)
            for cv_seed in cv_seeds:
                prob, auc, bacc = probe(X, y, groups, task, cv_seed, cfg["cv_folds"])
                metrics.append(dict(extractor=ext_dir.name, group=str(f["group"]), instance=path.stem,
                                    task=task, cv_seed=cv_seed, n=len(y), auc=auc, balanced_accuracy=bacc))
                preds.append(pd.DataFrame(dict(extractor=ext_dir.name, instance=path.stem, task=task,
                                               cv_seed=cv_seed, sample=ids, label=y, prob=prob)))
    m = pd.DataFrame(metrics)
    m = m[(m.extractor == ext_dir.name) & (m.cv_seed == cfg["cv_seed"])]
    for task, g in m.groupby("task"):
        print(f"{ext_dir.name:18s} {task:10s} {len(g):3d} instances: AUC {g.auc.mean():.3f} ± {g.auc.std():.3f}",
              flush=True)

pd.DataFrame(metrics).to_csv(root / "metrics.csv", index=False, float_format="%.6g")
pd.concat(preds).to_csv(root / "predictions.csv.gz", index=False, float_format="%.6g")
print(f"wrote {root / 'metrics.csv'}")
