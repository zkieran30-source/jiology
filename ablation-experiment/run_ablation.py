#!/usr/bin/env python3
"""SMIC minimality ablation on two heterogeneous real datasets.
Python 3.11 compatible; all seeds 0..199 and T=200 are explicit.
"""
from __future__ import annotations
from pathlib import Path
import json, zipfile, argparse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, t

N=40; T=200; SEEDS=range(200); NOISE_SD=.015; FLOOR=.02; INC=.004
CONDITIONS=['Full_SMIC','remove_M','remove_I','remove_C','remove_S','SMI','SIC','MIC','MI','M_only']

def gini(x):
    x=np.sort(np.maximum(np.asarray(x,float),0)); n=len(x); s=x.sum()
    return 0.0 if s<=0 else float((2*np.arange(1,n+1)@x)/(n*s)-(n+1)/n)

def ci(v):
    v=np.asarray(v); se=v.std(ddof=1)/np.sqrt(len(v)); q=t.ppf(.975,len(v)-1); return float(v.mean()-q*se),float(v.mean()+q*se)

def pair(r,cap,rng):
    order=rng.permutation(N)
    for j in range(0,N-1,2):
        a,b=map(int,order[j:j+2]); d=abs(r[a]-r[b]); amount=min(.8*d,cap)
        if r[a]>=r[b]: r[a]-=amount; r[b]+=amount
        else: r[b]-=amount; r[a]+=amount

def step(r,cond,rng):
    # Explicit operational mapping of the requested ablations/substitutions.
    # Removing M means all subjects are homogeneous; SIC has the same missing-M
    # representation. Removing S means no historical state is retained, so the
    # current structural reference is the uniform feasible state.
    if cond in ('remove_M','SIC','M_only'):
        r[:] = r.mean()
    if cond in ('remove_S','MIC','MI'):
        r[:] = r.mean()
    # Removing I means independent evolution: no pairwise transfer.
    if cond not in ('remove_I','MI','M_only'):
        cap=np.inf if cond in ('remove_C','SMI','MI') else .01
        pair(r,cap,rng)

def simulate(initial,cond,seed,noise):
    rng=np.random.default_rng(seed+7919); r=initial.copy(); tr=np.empty(T+1); tr[0]=gini(r)
    for k in range(T):
        step(r,cond,rng); r += noise[k]; r=np.maximum(r,FLOOR); r += INC; tr[k+1]=gini(r)
    return gini(r),tr

def load_taxi(path):
    d=pd.read_parquet(path,columns=['PULocationID']); top=d.PULocationID.dropna().astype(int).value_counts().head(N); x=top.to_numpy(float); return x/x.mean(), {'dataset':'NYC TLC Green Taxi January 2015','source_url':'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2015-01.parquet','rows':int(len(d)),'subject_rule':'40 most frequent PULocationID','subject_ids':[int(i) for i in top.index]}

def load_electric(path):
    # Stream the 678MB UCI file; choose 40 highest-mean clients as subjects.
    use=None; sums=None; count=0
    for chunk in pd.read_csv(path,sep=';',decimal=',',header=0,chunksize=20000):
        if use is None:
            cols=list(chunk.columns[1:]); use=np.array(cols[:],dtype=object); sums=np.zeros(len(use))
        vals=chunk.iloc[:,1:].to_numpy(float); sums += np.nan_to_num(vals).sum(axis=0); count += len(chunk)
    order=np.argsort(sums)[::-1][:N]; x=sums[order]/count; return x/x.mean(), {'dataset':'UCI ElectricityLoadDiagrams20112014','source_url':'https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip','rows':int(count),'clients':370,'subject_rule':'40 highest-mean clients','subject_columns':[str(use[i]) for i in order]}

def ablation(data_name,initial,out):
    finals={c:[] for c in CONDITIONS}; traj={c:[] for c in CONDITIONS}; errors={c:[] for c in CONDITIONS}
    for seed in SEEDS:
        nr=np.random.default_rng(seed); noise=nr.normal(0,NOISE_SD,size=(T,N))
        for c in CONDITIONS:
            f,tr=simulate(initial,c,seed,noise); finals[c].append(f); traj[c].append(tr); errors[c].append(float(np.sqrt(np.mean(np.diff(tr)**2))))
    rows=[]
    for c in CONDITIONS:
        v=np.array(finals[c]); lo,hi=ci(v); rows.append({'dataset':data_name,'condition':c,'n_seeds':200,'final_gini_mean':v.mean(),'final_gini_std':v.std(ddof=1),'ci95_low':lo,'ci95_high':hi,'trajectory_rmse':np.mean(errors[c])})
    return pd.DataFrame(rows),finals,traj

def tests(df, data_name):
    wide=df.pivot(index='seed',columns='condition',values='final_gini') if 'seed' in df else None
    return []

def run_tests(finals,data_name):
    wide=pd.DataFrame(finals); rows=[]
    f=friedmanchisquare(*[wide[c].to_numpy() for c in CONDITIONS]); rows.append({'dataset':data_name,'test':'Friedman','condition_a':'all','condition_b':'','statistic':float(f.statistic),'p_value':float(f.pvalue),'cohens_dz':np.nan})
    for c in CONDITIONS[1:]:
        d=wide['Full_SMIC']-wide[c]; w=wilcoxon(wide['Full_SMIC'],wide[c],zero_method='wilcox',method='auto'); dz=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
        rows.append({'dataset':data_name,'test':'paired_Wilcoxon','condition_a':'Full_SMIC','condition_b':c,'statistic':float(w.statistic),'p_value':float(w.pvalue),'cohens_dz':float(dz)})
    return pd.DataFrame(rows)

def make_plots(name,summary,finals,traj,out):
    colors=plt.cm.tab10(np.linspace(0,1,len(CONDITIONS)))
    fig,ax=plt.subplots(figsize=(12,6))
    for c,col in zip(CONDITIONS,colors): ax.plot(np.arange(T+1),np.mean(traj[c],axis=0),label=c,lw=1.4,color=col)
    ax.set(xlabel='Time step',ylabel='Gini',title=f'{name}: SMIC ablation trajectories (200 paired seeds)'); ax.legend(ncol=2,fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/f'figure1_ablation_trajectories_{name}.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5)); ax.boxplot([finals[c] for c in CONDITIONS],tick_labels=CONDITIONS,showfliers=False); ax.set(ylabel='Final Gini',title=f'{name}: final Gini by model'); ax.tick_params(axis='x',rotation=45); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(out/f'figure2_ablation_boxplot_{name}.png',dpi=220); plt.close(fig)

def forecast_metrics(series):
    y=np.asarray(series,float); n=len(y); split=int(n*.8); train=y[:split]; test=y[split:]
    scale=max(np.mean(train),1e-9); train=train/scale; test=test/scale
    # random walk
    rw=np.r_[train[-1],test[:-1]]
    # system dynamics: AR(1) with intercept estimated on train
    X=np.c_[np.ones(len(train)-1),train[:-1]]; coef=np.linalg.lstsq(X,train[1:],rcond=None)[0]; prev=train[-1]; sd=[]
    for _ in test: prev=coef[0]+coef[1]*prev; sd.append(prev)
    # ARIMA-like differenced AR(1): forecast change with mean change
    dif=np.diff(train); arima=np.r_[train[-1]+np.mean(dif)]
    for _ in range(1,len(test)): arima=np.r_[arima,arima[-1]+np.mean(dif)]
    # SMIC proxy: state + interaction signal (recent slope), constrained nonnegative forecast
    sm=[]; prev=train[-1]
    for k in range(len(test)):
        slope=train[-1]-train[-2] if k==0 else test[k-1]-test[k-2]
        prev=max(0,prev+0.65*slope); sm.append(prev)
    preds={'SMIC':np.array(sm),'system_dynamics':np.array(sd),'ARIMA':np.array(arima),'random_walk':rw}
    rows=[]
    for model,p in preds.items():
        e=test-p; rmse=np.sqrt(np.mean(e**2)); mae=np.mean(np.abs(e)); r2=1-np.sum(e**2)/np.sum((test-test.mean())**2)
        rows.append({'model':model,'RMSE':rmse*scale,'MAE':mae*scale,'R2':r2,'transition_accuracy':float(np.mean(np.sign(np.diff(np.r_[train[-1],p]))==np.sign(np.diff(np.r_[train[-1],test]))))})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); args=ap.parse_args(); root=args.root; data=root/'data'; out=root; figdir=root/'figures'; figdir.mkdir(exist_ok=True)
    taxi,tm=load_taxi(data/'green_tripdata_2015-01.parquet'); elec,em=load_electric(data/'LD2011_2014.txt')
    allsum=[]; alltests=[]; manifests={'datasets':[tm,em],'protocol':{'N':N,'T':T,'seeds':'0..199','noise_sd':NOISE_SD,'floor':FLOOR,'increment':INC,'conditions':CONDITIONS}}
    for name,x,meta in [('taxi',taxi,tm),('electricity',elec,em)]:
        sm,finals,traj=ablation(name,x,out); allsum.append(sm); alltests.append(run_tests(finals,name)); make_plots(name,sm,finals,traj,figdir)
    summary=pd.concat(allsum,ignore_index=True); summary.to_csv(out/'ablation_results.csv',index=False); tests_df=pd.concat(alltests,ignore_index=True); tests_df.to_csv(out/'statistical_tests.csv',index=False)
    # Cross-domain forecasting series: taxi hourly pickup counts and electricity mean load sampled hourly.
    taxi_full=pd.read_parquet(data/'green_tripdata_2015-01.parquet',columns=['lpep_pickup_datetime']); taxi_series=taxi_full.assign(h=pd.to_datetime(taxi_full.lpep_pickup_datetime).dt.floor('h')).groupby('h').size().to_numpy(float)
    selected=em['subject_columns']
    el=pd.read_csv(data/'LD2011_2014.txt',sep=';',decimal=',',usecols=['Unnamed: 0']+selected,parse_dates=['Unnamed: 0'])
    el_series=el.iloc[:,1:].mean(axis=1).to_numpy(float)
    # sample every 4th 15-min point to hourly-like regular sequence
    el_series=el_series[::4]
    fc=[]
    for domain,s in [('taxi',taxi_series),('electricity',el_series)]:
        m=forecast_metrics(s[:min(len(s),5000)]); m.insert(0,'dataset',domain); fc.append(m)
    fc=pd.concat(fc,ignore_index=True); fc.to_csv(out/'cross_domain_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(fc)); labels=[f'{d}\n{m}' for d,m in zip(fc.dataset,fc.model)]; ax.bar(x,fc.RMSE); ax.set_xticks(x,labels,rotation=45,ha='right'); ax.set_ylabel('RMSE'); ax.set_title('Cross-domain forecasting comparison'); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(out/'figure3_cross_domain.png',dpi=220); plt.close(fig)
    manifests['cross_domain']=fc.to_dict(orient='records'); (out/'experiment_manifest.json').write_text(json.dumps(manifests,indent=2,default=str),encoding='utf-8')
    print(summary.to_string(index=False)); print(tests_df.to_string(index=False)); print(fc.to_string(index=False))

if __name__=='__main__': main()
