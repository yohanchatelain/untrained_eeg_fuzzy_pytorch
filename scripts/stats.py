"""Significance tests for the model comparisons, from saved phase 1 and phase 2 results.

For every dataset and task:
  above_chance   is the extractor's ensemble (averaged predictions of all its instances)
                 better than chance? Label permutation test of its AUC (10,000 permutations;
                 for eyes closed vs open, labels are permuted within each participant)
  vs_matched     pretrained checkpoints vs the untrained network of the same architecture and
                 input: two-sided Mann-Whitney U test between the checkpoints' AUCs and the
                 seeds' AUCs (is the checkpoint distinguishable from a random draw?)
  vs_scattering  wavelet scattering (one AUC per scale J) vs each other extractor (one AUC per
                 seed or checkpoint), Mann-Whitney U

Holm correction within each family (test x dataset). Instance AUCs share the same samples, so
these tests ask whether weights (random, pretrained or wavelets) change the result beyond the
weight-to-weight spread; they do not generalize over new participants.

Writes results/stats/stats.csv.

    python scripts/stats.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from untrained_eeg.data import REPO

MATCHED = [("eegnetv4", "eegnet_3ch"), ("shallow_pretrained", "shallow_bcic2a")]
N_PERM = 10_000


def holm(p):
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def permutation_p(y, score, groups, rng):
    """One-sided p-value of AUC(y, score) under label permutations (within groups if given)."""
    obs = roc_auc_score(y, score)
    count = 0
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)] if groups is not None else None
    for _ in range(N_PERM):
        if idx is None:
            yp = rng.permutation(y)
        else:
            yp = y.copy()
            for i in idx:
                yp[i] = rng.permutation(y[i])
        count += roc_auc_score(yp, score) >= obs
    return obs, (count + 1) / (N_PERM + 1)


rng = np.random.default_rng(0)
rows = []
for d in sorted((REPO / "results" / "phase1").iterdir()):
    if not (d / "metrics.csv").exists():
        continue
    ds = d.name
    m = pd.read_csv(d / "metrics.csv")
    m = m[m.cv_seed == m.cv_seed.min()]  # the fixed split of the config
    preds = pd.read_csv(d / "predictions.csv.gz")
    preds = preds[preds.cv_seed == preds.cv_seed.min()]
    for task, mt in m.groupby("task"):
        auc = {e: g.auc.to_numpy() for e, g in mt.groupby("extractor")}
        pt = preds[preds.task == task]
        for e, g in pt.groupby("extractor"):  # ensemble of all instances vs chance
            w = g.pivot_table(index="sample", columns="instance", values="prob")
            y = g.drop_duplicates("sample").set_index("sample").label.loc[w.index].to_numpy()
            groups = np.array([s.split("/")[0] for s in w.index]) if task == "ec_eo" else None
            obs, p = permutation_p(y, w.mean(axis=1).to_numpy(), groups, rng)
            rows.append(dict(test="above_chance", dataset=ds, task=task, a=e, b="chance",
                             auc_a=obs, auc_b=0.5, n_a=w.shape[1], n_b=0, p=p))
        for pre, unt in MATCHED:
            if pre in auc and unt in auc:
                p = mannwhitneyu(auc[pre], auc[unt], alternative="two-sided").pvalue
                rows.append(dict(test="vs_matched", dataset=ds, task=task, a=pre, b=unt, auc_a=auc[pre].mean(),
                                 auc_b=auc[unt].mean(), n_a=len(auc[pre]), n_b=len(auc[unt]), p=p))
        if "scattering" in auc:
            for e, v in auc.items():
                if e != "scattering":
                    p = mannwhitneyu(auc["scattering"], v, alternative="two-sided").pvalue
                    rows.append(dict(test="vs_scattering", dataset=ds, task=task, a="scattering", b=e,
                                     auc_a=auc["scattering"].mean(), auc_b=v.mean(),
                                     n_a=len(auc["scattering"]), n_b=len(v), p=p))
    print(f"{ds}: {sum(r['dataset'] == ds for r in rows)} tests", flush=True)

s = pd.DataFrame(rows)
s["p_holm"] = s.groupby(["test", "dataset"]).p.transform(holm)
s["significant"] = s.p_holm < 0.05
s["key"] = s.dataset + "|" + s.test + "|" + s.task + "|" + s.a + "|" + s.b
out = REPO / "results" / "stats"
out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / "stats.csv", index=False, float_format="%.6g")
print(s.groupby(["test", "dataset"]).significant.agg(["sum", "count"]).to_string())
print(f"wrote {out / 'stats.csv'}")
