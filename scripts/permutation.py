"""Permutation test of every model on every task of one dataset (refitting the probe).

For each extractor and task: the observed AUC against the AUCs obtained after shuffling the
labels and refitting the same probe with the same CV split (Ojala and Garriga 2010;
scikit-learn's permutation_test_score). Labels are shuffled within each participant where a
participant has several samples (eyes closed vs open, motor imagery, sleep staging), freely
otherwise. Holm-corrected over all extractor x task tests of the dataset.

Two modes:
  default     the first instance only (seed000, the first checkpoint, scattering's J3 or the
              single foundation model, fixed in advance); p = (1 + #null >= observed) / (1 + n)
  --ensemble  the ensemble of ALL instances of the extractor (averaged out-of-fold predictions,
              as in phase 2 and the above-chance test of stats.py). Every instance's probe is
              refit on the same shuffled labels, then their predictions are averaged. Single-
              instance extractors (foundation models) are tested the same way. Permutations stop
              early once 20 null AUCs reach the observed one (Besag and Clifford 1991: p = 20 / n),
              so only tests that may be significant run all --permutations.

Writes results/stats/permutation_<dataset>.csv (default) or permutation_ensemble_<dataset>.csv;
the ensemble mode saves each finished test, so a job that times out can be resubmitted.
Run it as a job (one probe per instance and permutation):
    CONFIG=configs/ds004504.yaml sbatch slurm/permutation.sh
    CONFIG=configs/ds004504.yaml ARGS="--ensemble --permutations 10000" sbatch slurm/permutation.sh
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
from untrained_eeg.tasks import MULTICLASS, WITHIN_SUBJECT, auc_score, probe, task_data, task_labels, task_names

STOP_AFTER = 20  # ensemble mode: null AUCs >= observed before stopping early


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


def ensemble_auc(Xs, y, groups, task, cfg):
    """AUC of the averaged out-of-fold predictions of one probe per instance (Xs)."""
    probs = [probe(X, y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[0] for X in Xs]
    return auc_score(y, np.mean(probs, axis=0))


def null_aucs(Xs, y, groups, task, cfg, seeds):
    """Ensemble AUCs after shuffling the labels with each seed (the same labels for every instance)."""
    clustered = task == "ec_eo" or task in WITHIN_SUBJECT or task in MULTICLASS
    return [ensemble_auc(Xs, shuffled(y, groups if clustered else None, np.random.default_rng(s)), groups, task, cfg)
            for s in seeds]


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--config", default=None)
parser.add_argument("--permutations", type=int, default=None, help="default: config n_permutations, else 1000")
parser.add_argument("--ensemble", action="store_true", help="test the ensemble of all instances (see above)")
parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
args = parser.parse_args()

cfg = load_config(args.config) if args.config else load_config()
n_perm = args.permutations or cfg.get("n_permutations", 1000)
ds = cfg["openneuro"]["id"]
src = Path(cfg["results_dir"]) / "phase1" / ds / "features"
dest = Path(cfg["results_dir"]) / "stats" / f"permutation_{'ensemble_' if args.ensemble else ''}{ds}.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
labels = task_labels(cfg)
rows = []
partial = dest.with_suffix(".partial.csv")  # ensemble mode: tests finished so far
if args.ensemble and partial.exists():
    rows = pd.read_csv(partial).to_dict("records")
done = {(r["extractor"], r["task"]) for r in rows}

for ext_dir in sorted(p for p in src.iterdir() if p.is_dir()):
    paths = sorted(ext_dir.glob("*.npz"))
    if not args.ensemble:
        paths = paths[:1]  # the first instance, as in phase1_probe's CV-split runs
    files = [np.load(p) for p in paths]
    for task in task_names(cfg):
        if (ext_dir.name, task) in done:
            continue
        data = [task_data(f["feats"], f["cond"], f["participant_id"], task, labels) for f in files]
        Xs, (_, y, groups, ids) = [d[0] for d in data], data[0]
        assert all(d[3] == ids for d in data), "instances must share the same samples"
        if not args.ensemble:
            observed = probe(Xs[0], y, groups, task, cfg["cv_seed"], cfg["cv_folds"])[1]
            null = np.concatenate(Parallel(n_jobs=args.workers)(
                delayed(null_aucs)(Xs, y, groups, task, cfg, [seed]) for seed in range(n_perm)))
            p = (1 + (null >= observed).sum()) / (1 + n_perm)
        else:  # rounds of one chunk of permutations per worker, until done or STOP_AFTER exceedances
            observed = ensemble_auc(Xs, y, groups, task, cfg)
            null, chunk = np.array([]), max(1, min(50, n_perm // (10 * args.workers)))
            while len(null) < n_perm and (null >= observed).sum() < STOP_AFTER:
                start = len(null)
                seeds = list(range(start, min(n_perm, start + chunk * args.workers)))
                parts = Parallel(n_jobs=args.workers)(
                    delayed(null_aucs)(Xs, y, groups, task, cfg, seeds[i : i + chunk])
                    for i in range(0, len(seeds), chunk))
                null = np.concatenate([null, *map(np.array, parts)])
            hits = (null >= observed).sum()
            p = hits / len(null) if hits >= STOP_AFTER else (1 + hits) / (1 + len(null))
        rows.append(dict(dataset=ds, task=task, extractor=ext_dir.name,
                         instance=f"{len(paths)} instances" if args.ensemble else paths[0].stem, auc=observed,
                         null_mean=null.mean(), null_p95=np.quantile(null, 0.95), n_permutations=len(null), p=p))
        if args.ensemble:
            pd.DataFrame(rows).to_csv(partial, index=False, float_format="%.6g")
        print(f"{ext_dir.name:18s} {task:15s} AUC {observed:.3f}  null {null.mean():.3f} "
              f"(95% {np.quantile(null, 0.95):.3f}, n {len(null)})  p {p:.4f}", flush=True)

out = pd.DataFrame(rows)
out["p_holm"] = holm(out.p)
out["significant"] = out.p_holm < 0.05
out.to_csv(dest, index=False, float_format="%.6g")
partial.unlink(missing_ok=True)
print(f"wrote {dest}: {out.significant.sum()} of {len(out)} significant after Holm")
