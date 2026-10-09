"""Phase 2: ensembles of K model instances on one dataset, and how they change with K.

A pool is a set of instances: the 30 seeds of one untrained extractor, the pretrained
checkpoints, or a mix of extractors. For each K (1 .. pool size) and up to --draws distinct
sets of K instances (all of them when fewer exist, fixed RNG), the K instances are combined:
  features   average their block features, then one probe (same extractor only)
  average    average their out-of-fold P(class 1) (soft voting)
  vote       majority vote of their predicted classes; the score is the share voting
             class 1, ties (even K) go to the average P
All instances share the same fixed CV split (cv_seed of the config), so their out-of-fold
predictions can be combined sample by sample.

Reads results/phase1/<dataset>/ (features/ and predictions.csv.gz). Writes results/phase2/<dataset>/:
  draws.csv     pool, method, task, k, draw, instances, auc
  summary.csv   per pool x method x task x k: AUC mean and SD over draws, median per-sample
                SD of the score over draws, flip rate (share of samples whose class differs
                between draws)
  curves.png    AUC and spread against K

    python scripts/phase2_ensembles.py --workers 8                   # ds003478
    python scripts/phase2_ensembles.py --config configs/ds004504.yaml --workers 8
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import load_config
from untrained_eeg.tasks import probe, task_data, task_labels, task_names

UNTRAINED = ["cnn1d", "shallow", "eegnet", "eegnet_3ch", "shallow_bcic2a"]
PRETRAINED = ["eegnetv4", "shallow_pretrained"]
POOLS = {**{e: [e] for e in UNTRAINED + PRETRAINED},  # pool name: extractors in it
         "untrained_60ch_mix": ["cnn1d", "shallow", "eegnet"],  # different architectures
         "pretrained_mix": PRETRAINED}


def draws_of(n, k, n_draws, rng):
    """Up to n_draws distinct sets of k indices out of n: all of them when there are few."""
    if math.comb(n, k) <= n_draws:
        return [list(c) for c in itertools.combinations(range(n), k)]
    out = set()
    while len(out) < n_draws:
        out.add(tuple(sorted(rng.choice(n, size=k, replace=False))))
    return [list(d) for d in sorted(out)]


def vote(probs):
    """Majority vote of the rows' classes. Score = share voting 1; ties go to the mean P."""
    share = (probs > 0.5).mean(axis=0)
    tie = share == 0.5
    share[tie] = np.where(probs[:, tie].mean(axis=0) > 0.5, 0.5 + 1e-9, 0.5)
    return share


def feature_ensemble(X, y, groups, task, cfg):
    """Probe one draw's averaged features (computed by the caller: workers get only X)."""
    return probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[0]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--draws", type=int, default=50, help="draws per K for average and vote")
parser.add_argument("--feature-draws", type=int, default=20, help="draws per K for features (each a probe)")
parser.add_argument("--max-k", type=int, default=30)
parser.add_argument("--workers", type=int, default=1)
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
src = Path(cfg["results_dir"]) / "phase1" / cfg["openneuro"]["id"]
out = Path(cfg["results_dir"]) / "phase2" / cfg["openneuro"]["id"]
out.mkdir(parents=True, exist_ok=True)
labels = task_labels(cfg)
TASKS = task_names(cfg)
preds = pd.read_csv(src / "predictions.csv.gz")
preds = preds[preds.cv_seed == cfg["cv_seed"]]
rng = np.random.default_rng(0)
rows, summary = [], []

for pool, extractors in POOLS.items():
    if not set(extractors) <= set(preds.extractor):  # e.g. no BCI IV 2a input on this dataset
        continue
    p = preds[preds.extractor.isin(extractors)]
    insts = sorted(set(zip(p.extractor, p.instance)))  # (extractor, instance) members
    for task in TASKS:
        t = p[p.task == task]
        wide = t.pivot_table(index="sample", columns=["extractor", "instance"], values="prob")[insts]
        y = t.drop_duplicates("sample").set_index("sample").label.loc[wide.index].to_numpy()
        P = wide.to_numpy().T  # instances x samples
        methods = ["average", "vote"] + (["features"] if len(extractors) == 1 else [])
        if len(extractors) == 1:
            files = [src / "features" / e / f"{i}.npz" for e, i in insts]
            f0 = {k: np.load(files[0])[k] for k in ("cond", "participant_id")}  # arrays: workers get copies
            feats = [np.load(f)["feats"] for f in files]
        for k in range(1, min(len(insts), args.max_k) + 1):
            for method in methods:
                draws = draws_of(len(insts), k, args.feature_draws if method == "features" else args.draws, rng)
                if method == "features":
                    inputs = [task_data(np.mean([feats[m] for m in d], axis=0), f0["cond"],
                                        f0["participant_id"], task, labels) for d in draws]
                    scores = Parallel(n_jobs=args.workers)(
                        delayed(feature_ensemble)(X, yy, g, task, cfg) for X, yy, g, _ in inputs)
                    ids = inputs[0][3]
                    scores = pd.DataFrame(np.array(scores), columns=ids)[list(wide.index)].to_numpy()  # same order as P
                elif method == "average":
                    scores = np.array([P[d].mean(axis=0) for d in draws])
                else:
                    scores = np.array([vote(P[d]) for d in draws])
                aucs = np.array([roc_auc_score(y, s) for s in scores])
                cls = scores > 0.5
                for i, (d, a) in enumerate(zip(draws, aucs)):
                    rows.append(dict(pool=pool, method=method, task=task, k=k, draw=i, auc=a,
                                     instances=" ".join(f"{insts[j][0]}:{insts[j][1]}" for j in d)))
                many = len(draws) > 1
                summary.append(dict(pool=pool, method=method, task=task, k=k, draws=len(draws),
                                    auc_mean=aucs.mean(), auc_sd=aucs.std(ddof=1) if many else np.nan,
                                    median_score_sd=np.median(scores.std(axis=0, ddof=1)) if many else np.nan,
                                    flip_rate=(cls.any(axis=0) & ~cls.all(axis=0)).mean()))
        s = pd.DataFrame(summary)
        s = s[(s.pool == pool) & (s.task == task) & (s.method == "average")]
        print(f"{pool:20s} {task:10s} average: AUC K=1 {s.auc_mean.iloc[0]:.3f} -> "
              f"K={s.k.iloc[-1]} {s.auc_mean.iloc[-1]:.3f}", flush=True)

pd.DataFrame(rows).to_csv(out / "draws.csv", index=False, float_format="%.6g")
summary = pd.DataFrame(summary)
summary.to_csv(out / "summary.csv", index=False, float_format="%.6g")

# AUC against K, one panel per task, one line per pool (averaged predictions).
fig, axes = plt.subplots(1, len(TASKS), figsize=(4.4 * len(TASKS), 4), sharex=True, squeeze=False)
axes = axes[0]
for ax, task in zip(axes, TASKS):
    for pool, g in summary[(summary.task == task) & (summary.method == "average")].groupby("pool"):
        ax.plot(g.k, g.auc_mean, label=pool, lw=2 if pool in PRETRAINED + ["pretrained_mix"] else 1)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_title(task)
    ax.set_xlabel("K (instances in the ensemble)")
axes[0].set_ylabel("AUC (mean over draws), averaged predictions")
axes[-1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "curves.png", dpi=150)
print(f"wrote {out}")
