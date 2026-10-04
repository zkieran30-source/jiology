#!/usr/bin/env python3
"""Jiology multi-agent resource allocation experiment.

Python 3.11 compatible. The experiment is intentionally fixed to the requested
protocol: six conditions, seeds 0..199, T=200, shared initial state/noise/
replenishment, and the three Jiology cap rules plus three baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, t

N = 40
T = 200
SEEDS = range(200)
NOISE_SD = 0.015
FLOOR = 0.02
UNIFORM_INCREMENT = 0.004
FIXED_CAP = 0.01
UNCONSTRAINED_CAP = 0.20
CONDITIONS = ["fixed", "unconstrained", "adaptive", "random", "uniform", "greedy"]


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.sort(np.maximum(x, 0.0))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2.0 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def select_initial_resources(data_path: Path, cache_path: Path) -> tuple[np.ndarray, dict]:
    """Use the 40 most frequent real PULocationID subjects and normalize mean to 1."""
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return np.asarray(cached["initial_resources"], dtype=float), cached
    df = pd.read_parquet(data_path, columns=["PULocationID"])
    counts = df["PULocationID"].dropna().astype(int).value_counts().sort_index()
    top = counts.sort_values(ascending=False).head(N)
    raw = top.to_numpy(dtype=float)
    initial = raw / raw.mean()
    meta = {
        "source": "NYC TLC Green Taxi Trip Records, January 2015",
        "source_url": "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet",
        "selection_rule": "40 most frequent PULocationID values in the downloaded real dataset",
        "n_rows": int(len(df)),
        "subject_ids": [int(i) for i in top.index.tolist()],
        "subject_trip_counts": [int(v) for v in top.tolist()],
        "initial_resources": initial.tolist(),
        "normalization": "divide by the mean of the selected 40 counts, yielding mean resource 1",
    }
    cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return initial, meta


def pair_transfers(r: np.ndarray, cap: float, rng: np.random.Generator) -> None:
    """Random pairing; rich-to-poor transfer min(0.8*gap, cap), in place."""
    order = rng.permutation(N)
    for i in range(0, N - 1, 2):
        a, b = int(order[i]), int(order[i + 1])
        gap = abs(float(r[a] - r[b]))
        amount = min(0.8 * gap, cap)
        if r[a] > r[b]:
            r[a] -= amount; r[b] += amount
        else:
            r[b] -= amount; r[a] += amount


def one_step(r: np.ndarray, condition: str, rng: np.random.Generator) -> None:
    if condition == "fixed":
        pair_transfers(r, FIXED_CAP, rng)
    elif condition == "unconstrained":
        pair_transfers(r, UNCONSTRAINED_CAP, rng)
    elif condition == "adaptive":
        cap = min(0.08, 0.01 + 0.25 * gini(r))
        pair_transfers(r, cap, rng)
    elif condition == "random":
        order = rng.permutation(N)
        for i in range(0, N - 1, 2):
            a, b = int(order[i]), int(order[i + 1])
            max_amount = min(0.8 * abs(float(r[a] - r[b])), UNCONSTRAINED_CAP)
            amount = rng.uniform(0.0, max_amount)
            if rng.random() < 0.5:
                r[a] -= amount; r[b] += amount
            else:
                r[b] -= amount; r[a] += amount
    elif condition == "uniform":
        mean = float(r.mean())
        r += 0.01 * (mean - r)
    elif condition == "greedy":
        rich = int(np.argmax(r)); poor = int(np.argmin(r))
        amount = min(0.8 * abs(float(r[rich] - r[poor])), UNCONSTRAINED_CAP)
        r[rich] -= amount; r[poor] += amount
    else:
        raise ValueError(condition)


def simulate(initial: np.ndarray, condition: str, seed: int, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one condition with common noise and common +0.004 increment."""
    # Separate deterministic condition RNG from the shared perturbation stream.
    rng = np.random.default_rng(seed + 10_000_000)
    r = initial.copy()
    trajectory = np.empty(T + 1, dtype=float)
    trajectory[0] = gini(r)
    for t_idx in range(T):
        one_step(r, condition, rng)
        r += noise[t_idx]
        r = np.maximum(r, FLOOR)
        r += UNIFORM_INCREMENT
        trajectory[t_idx + 1] = gini(r)
    return r, trajectory


def ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values)); se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    q = float(t.ppf(0.975, len(values) - 1))
    return mean - q * se, mean + q * se


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/green_tripdata_2015-01.parquet"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    initial, data_meta = select_initial_resources(args.data, args.out / "data_selection.json")
    finals = {c: [] for c in CONDITIONS}; trajectories = {c: [] for c in CONDITIONS}
    for seed in SEEDS:
        noise_rng = np.random.default_rng(seed)
        shared_noise = noise_rng.normal(0.0, NOISE_SD, size=(T, N))
        for c in CONDITIONS:
            final, traj = simulate(initial, c, seed, shared_noise)
            finals[c].append(gini(final)); trajectories[c].append(traj)
    final_df = pd.DataFrame(finals)
    summary_rows=[]
    for c in CONDITIONS:
        vals=final_df[c].to_numpy(); lo,hi=ci95(vals)
        summary_rows.append({"condition":c,"n_seeds":len(vals),"final_gini_mean":vals.mean(),"final_gini_std":vals.std(ddof=1),"ci95_low":lo,"ci95_high":hi})
    summary=pd.DataFrame(summary_rows); summary.to_csv(args.out/"final_summary.csv",index=False)
    fried=friedmanchisquare(*[final_df[c].to_numpy() for c in CONDITIONS])
    pairs=[("fixed","unconstrained"),("fixed","adaptive"),("adaptive","unconstrained"),("fixed","random"),("fixed","uniform"),("fixed","greedy")]
    tests=[{"test":"Friedman","condition_a":"all_six","condition_b":"","statistic":float(fried.statistic),"p_value":float(fried.pvalue),"cohens_dz":np.nan}]
    for a,b in pairs:
        x=final_df[a].to_numpy(); y=final_df[b].to_numpy(); diff=x-y
        w=wilcoxon(x,y,alternative="two-sided",zero_method="wilcox",method="auto")
        dz=float(diff.mean()/diff.std(ddof=1)) if diff.std(ddof=1)>0 else np.nan
        tests.append({"test":"paired_Wilcoxon","condition_a":a,"condition_b":b,"statistic":float(w.statistic),"p_value":float(w.pvalue),"cohens_dz":dz})
    pd.DataFrame(tests).to_csv(args.out/"statistical_tests.csv",index=False)
    # Figure 1: mean trajectories and final boxplot.
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
    colors=dict(zip(CONDITIONS,["#d73027","#1a9850","#66bd63","#4575b4","#fee08b","#762a83"]))
    for c in CONDITIONS:
        arr=np.asarray(trajectories[c]); ax1.plot(np.arange(T+1),arr.mean(axis=0),label=c,color=colors[c],lw=1.8)
    ax1.set(xlabel="Time step",ylabel="Gini",title="Mean Gini trajectories (200 paired seeds)"); ax1.legend(fontsize=8); ax1.grid(alpha=.2)
    ax2.boxplot([final_df[c] for c in CONDITIONS],tick_labels=CONDITIONS,showfliers=False); ax2.set(ylabel="Final Gini",title="Final Gini by condition"); ax2.tick_params(axis='x',rotation=35); ax2.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(args.out/"figure1_trajectories_and_boxplot.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); means=summary.final_gini_mean.to_numpy(); lows=means-summary.ci95_low.to_numpy(); highs=summary.ci95_high.to_numpy()-means
    ax.bar(CONDITIONS,means,yerr=[lows,highs],capsize=4,color=[colors[c] for c in CONDITIONS]); ax.set_ylabel("Final Gini"); ax.set_title("Final Gini comparison with 95% CI"); ax.tick_params(axis='x',rotation=35); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(args.out/"figure2_bar_comparison.png",dpi=220); plt.close(fig)
    manifest={"protocol":"Jiology multi-agent resource allocation experiment","python":"3.11+","N":N,"T":T,"seeds":"0..199","noise_mean":0.0,"noise_sd":NOISE_SD,"floor":FLOOR,"uniform_increment":UNIFORM_INCREMENT,"fixed_cap":FIXED_CAP,"unconstrained_cap":UNCONSTRAINED_CAP,"adaptive_cap":"min(0.08,0.01+0.25*Gini)","conditions":CONDITIONS,"shared_inputs":"initial resources, Gaussian perturbation stream, and uniform increment are shared within each seed","data":data_meta,"outputs":["final_summary.csv","statistical_tests.csv","figure1_trajectories_and_boxplot.png","figure2_bar_comparison.png"]}
    (args.out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(pd.DataFrame(tests).to_string(index=False))

if __name__ == "__main__": main()
