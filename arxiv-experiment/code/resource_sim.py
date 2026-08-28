import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/home/ubuntu/work_jixue/experiment')
OUT.mkdir(exist_ok=True)

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x - x.min() + 1e-9
    n = len(x)
    return np.sum((2*np.arange(1,n+1)-n-1)*np.sort(x))/(n*np.sum(x))

def run(seed, regime, n=40, steps=200):
    rng = np.random.default_rng(seed)
    r = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    r = r / r.mean()
    target = 1.0
    rows = []
    for t in range(steps+1):
        gi = gini(r)
        rows.append({'seed':seed,'regime':regime,'step':t,'gini':gi,'mean_abs_gap':np.mean(np.abs(r-target)),'below_target':np.mean(r<target),'total':r.sum()})
        if t == steps: break
        pairs = rng.permutation(n).reshape(-1,2)
        if regime == 'none': cap = 0.20
        elif regime == 'fixed': cap = 0.01
        else: cap = min(0.08, 0.01 + 0.25*gi)
        for i,j in pairs:
            d = r[i]-r[j]
            # interaction: partial equalizing transfer, limited by constraint cap
            amount = np.sign(d) * min(abs(d)*0.80, cap)
            if amount > 0:
                amount = min(amount, r[i])
                r[i] -= amount; r[j] += amount
            elif amount < 0:
                amount = min(-amount, r[j])
                r[j] -= amount; r[i] += amount
        # exogenous resource shock and replenishment, explicitly part of the model
        shock = rng.normal(0, 0.015, size=n)
        r = np.maximum(r + shock, 0.02)
        r += 0.004
    return pd.DataFrame(rows)

all_rows = []
for regime in ['none','fixed','adaptive']:
    for seed in range(200):
        all_rows.append(run(seed, regime))
df = pd.concat(all_rows, ignore_index=True)
df.to_csv(OUT/'trajectories.csv', index=False)
final = df[df.step==200].groupby('regime').agg(
    gini_mean=('gini','mean'), gini_sd=('gini','std'),
    gap_mean=('mean_abs_gap','mean'), gap_sd=('mean_abs_gap','std'),
    below_mean=('below_target','mean'), below_sd=('below_target','std'),
    total_mean=('total','mean'), total_sd=('total','std')).reset_index()
final.to_csv(OUT/'final_summary.csv', index=False)
# 95% CI for final gini and gap
for metric in ['gini','mean_abs_gap','below_target']:
    sub = df[df.step==200].groupby('regime')[metric].agg(['mean','std','count']).reset_index()
    sub['ci95'] = 1.96*sub['std']/np.sqrt(sub['count'])
    sub.to_csv(OUT/f'{metric}_ci95.csv', index=False)

plt.figure(figsize=(9,5.2))
colors={'none':'#777777','fixed':'#2D00F7','adaptive':'#F20089'}
for regime in ['none','fixed','adaptive']:
    sub=df[df.regime==regime].groupby('step')['gini'].agg(['mean','std']).reset_index()
    plt.plot(sub.step, sub['mean'], label=regime, color=colors[regime], linewidth=2)
    plt.fill_between(sub.step, sub['mean']-1.96*sub['std']/np.sqrt(200), sub['mean']+1.96*sub['std']/np.sqrt(200), color=colors[regime], alpha=.12)
plt.xlabel('Time step'); plt.ylabel('Gini coefficient'); plt.title('Resource inequality under three constraint regimes')
plt.legend(frameon=False); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(OUT/'gini_trajectories.png', dpi=180); plt.close()
print(final.to_string(index=False))
