"""Permutation test of every model on every task of one dataset (refitting the probe).

For each extractor and task: the observed AUC of its first instance (seed000, the first
checkpoint, scattering's J3 or the single foundation model, fixed in advance), against the
AUCs obtained after shuffling the labels and refitting the same probe with the same CV split
(Ojala and Garriga 2010; scikit-learn's permutation_test_score). Labels are shuffled within
each participant where a participant has several samples (eyes closed vs open, motor imagery,
sleep staging), freely otherwise. p = (1 + #null AUC >= observed) / (1 + n_permutations),
Holm-corrected over all extractor x task tests of the dataset.

Writes results/stats/permutation_<dataset>.csv. Run it as a job (one probe per permutation):
    CONFIG=configs/ds004504.yaml sbatch slurm/permutation.sh
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, probe, task_data, task_labels, task_names


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def shuffled(y, groups, rng):
    """y with its labels permuted, within each group if groups is given."""
    if groups is None:
        return rng.permutation(y)
    out = y.copy()
    for g in np.unique(groups):
        i = np.flatnonzero(groups == g)
        out[i] = rng.permutation(y[i])
    return out


def null_auc(X, y, groups, task, cfg, seed):
    rng = np.random.default_rng(seed)
    clustered = task == "ec_eo" or task in WITHIN_SUBJECT or task in MULTICLASS
    return probe(X, shuffled(y, groups if clustered else None, rng), groups, task, cfg["cv_seed"], cfg["cv_folds"])[1]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--permutations", type=int, default=None, help="default: config n_permutations, else 1000")
parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
n_perm = args.permutations or cfg.get("n_permutations", 1000)
ds = cfg["openneuro"]["id"]
src = Path(cfg["results_dir"]) / "phase1" / ds / "features"
labels = task_labels(cfg)
rows = []
for ext_dir in sorted(p for p in src.iterdir() if p.is_dir()):
    path = sorted(ext_dir.glob("*.npz"))[0]  # the first instance, as in phase1_probe's CV-split runs
    f = np.load(path)
    for task in task_names(cfg):
        X, y, groups, _ = task_data(f["feats"], f["cond"], f["participant_id"], task, labels)
        observed = probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[1]
        null = np.array(Parallel(n_jobs=args.workers)(
            delayed(null_auc)(X, y, groups, task, cfg, seed) for seed in range(n_perm)))
        p = (1 + (null >= observed).sum()) / (1 + n_perm)
        rows.append(dict(dataset=ds, task=task, extractor=ext_dir.name, instance=path.stem, auc=observed,
                         null_mean=null.mean(), null_p95=np.quantile(null, 0.95), n_permutations=n_perm, p=p))
        print(f"{ext_dir.name:18s} {task:15s} AUC {observed:.3f}  null {null.mean():.3f} "
              f"(95% {np.quantile(null, 0.95):.3f})  p {p:.4f}", flush=True)

out = pd.DataFrame(rows)
out["p_holm"] = holm(out.p)
out["significant"] = out.p_holm < 0.05
dest = Path(cfg["results_dir"]) / "stats" / f"permutation_{ds}.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False, float_format="%.6g")
print(f"wrote {dest}: {out.significant.sum()} of {len(out)} significant after Holm")
