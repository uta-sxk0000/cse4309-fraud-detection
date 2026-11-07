"""Reproduce the fraud study. Run from the project root; see README.md."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
SOURCE_URL = 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'

def make_estimator(name, weight=None):
    """One definition shared by the full experiment and saved-model preparation."""
    if name == 'Logistic Regression':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0,
            solver='lbfgs', max_iter=2000, class_weight=weight, random_state=SEED))
    if name == 'Decision Tree':
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=10,
            class_weight=weight, random_state=SEED)
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=150, max_depth=16,
            min_samples_leaf=2, max_features='sqrt', class_weight=weight,
            n_jobs=4, random_state=SEED)
    raise ValueError(f'Unknown algorithm: {name}')

def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), accuracy=float(accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y, scores)),
        average_precision=float(average_precision_score(y, scores)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def threshold_for_f1(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Deterministic tie rule: highest threshold among equal maximum F1 values.
    index = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[index])

def wilson(success, total):
    z = 1.959963984540054
    p = success / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1+z*z/total)
    return [float(center-half), float(center+half)]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('.'))
    args = parser.parse_args()
    out = args.output
    for name in ['results', 'figures']:
        (out/name).mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        print('Downloading the public dataset mirror...', flush=True)
        urllib.request.urlretrieve(DATA_URL, args.data)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    raw = pd.read_csv(args.data)
    expected = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount', 'Class']
    assert raw.columns.tolist() == expected, 'Unexpected data schema'
    assert set(raw.Class.unique()) == {0, 1}
    assert np.isfinite(raw.to_numpy()).all(), 'Missing or non-finite values need a new policy'
    assert (raw.Amount >= 0).all() and (raw.Time >= 0).all()
    duplicate_count = int(raw.duplicated().sum())
    # Exact rows can otherwise appear in different splits. Keep their first occurrence.
    data = raw.drop_duplicates().copy()
    data.index.name = 'source_row'
    X, y = data.drop(columns='Class'), data.Class
    trainval_idx, test_idx = train_test_split(data.index, test_size=.2,
        random_state=SEED, stratify=y)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=.25,
        random_state=SEED, stratify=y.loc[trainval_idx])
    assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))
    Xtr, Xv, Xt = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
    ytr, yv, yt = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]
    splits = []
    for name, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        counts = y.loc[idx].value_counts()
        splits.append(dict(split=name, rows=len(idx), legitimate=int(counts[0]), fraud=int(counts[1])))
    pd.DataFrame(splits).to_csv(out/'results/splits.csv', index=False)
    pd.concat([pd.DataFrame({'source_row':idx, 'split':name}) for name,idx in
        [('train',train_idx),('validation',val_idx),('test',test_idx)]]).to_csv(out/'results/split_assignments.csv',index=False)
    feature_summary = data.describe().T
    feature_summary['dtype'] = data.dtypes.astype(str)
    feature_summary['missing'] = data.isna().sum()
    feature_summary.to_csv(out/'results/data_dictionary.csv')
    summary = dict(seed=SEED, raw_rows=len(raw), cleaned_rows=len(data),
        features=X.shape[1], raw_fraud=int(raw.Class.sum()), cleaned_fraud=int(y.sum()),
        duplicates_removed=duplicate_count, missing_values=int(raw.isna().sum().sum()),
        duplicate_fraud_removed=int(raw.Class.sum()-y.sum()),
        target='Class: 1=fraud; 0=legitimate', sha256=digest,
        original_source=SOURCE_URL, download_url=DATA_URL, splits=splits,
        python=platform.python_version(), versions={p:importlib.metadata.version(p)
        for p in ['numpy','pandas','scikit-learn','matplotlib','scipy','joblib']})
    print(json.dumps(summary, indent=2), flush=True)
    candidates, selected, models = [], [], {}
    for name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
        family = []
        for weight in [None, 'balanced']:
            model = make_estimator(name, weight)
            print(f'Fitting {name}, class_weight={weight}', flush=True)
            start = time.perf_counter()
            model.fit(Xtr, ytr)
            seconds = time.perf_counter() - start
            scores = model.predict_proba(Xv)[:, 1]
            threshold = threshold_for_f1(yv, scores)
            row = dict(model=name, class_weight=str(weight), fit_seconds=seconds,
                **evaluate(yv,scores,threshold))
            candidates.append(row)
            family.append((row,model))
            print(json.dumps(row),flush=True)
        # Select imbalance treatment using validation AP, before opening test outcomes.
        best, model = max(family, key=lambda pair:pair[0]['average_precision'])
        selected.append(best)
        models[name] = model
    recommendation = max(selected, key=lambda row: row['f1'])['model']
    summary['selection_rule'] = 'Within family: highest validation AP; threshold: max validation F1; recommended family: highest validation F1.'
    summary['recommended_model'] = recommendation
    pd.DataFrame(candidates).to_csv(out/'results/validation_candidates.csv',index=False)
    # All model choices and thresholds are now frozen. No refitting or test-driven tuning.
    print('Choices frozen. Evaluating held-out test set.',flush=True)
    results, defaults, curve_data = [], [], {}
    predictions = pd.DataFrame({'source_row':test_idx, 'actual':yt.to_numpy()})
    for selection in selected:
        name = selection['model']
        model = models[name]
        start = time.perf_counter()
        scores = model.predict_proba(Xt)[:,1]
        predict_seconds = time.perf_counter()-start
        row = dict(model=name,class_weight=selection['class_weight'],
            fit_seconds=selection['fit_seconds'],predict_seconds=predict_seconds,
            **evaluate(yt,scores,selection['threshold']))
        row['recall_ci_low'], row['recall_ci_high'] = wilson(row['tp'],row['tp']+row['fn'])
        results.append(row)
        defaults.append(dict(model=name, **evaluate(yt,scores,.5)))
        predictions[name+'_score'] = scores
        predictions[name+'_prediction'] = (scores >= selection['threshold']).astype(int)
        p,r,t = precision_recall_curve(yt,scores)
        fpr,tpr,_ = roc_curve(yt,scores)
        # Full resolution curves are retained for independent plots.
        pd.DataFrame({'precision':p,'recall':r}).to_csv(out/f'results/{name.lower().replace(" ","_")}_pr.csv',index=False)
        curve_data[name] = dict(precision=p,recall=r,fpr=fpr,tpr=tpr)
    baseline = dict(model='Always legitimate', **evaluate(yt,np.zeros(len(yt)),.5))
    summary['baseline'] = baseline
    summary['selected'] = selected
    summary['test_results'] = results
    (out/'results/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    pd.DataFrame(results).to_csv(out/'results/test_metrics.csv',index=False)
    pd.DataFrame(defaults).to_csv(out/'results/test_default_threshold.csv',index=False)
    predictions.to_csv(out/'results/test_predictions.csv',index=False)
    rf = models['Random Forest']
    importance = pd.DataFrame({'feature':X.columns,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
    importance.to_csv(out/'results/random_forest_importance.csv',index=False)
    lr_coef = models['Logistic Regression'].named_steps['logisticregression'].coef_[0]
    pd.DataFrame({'feature':X.columns,'standardized_coefficient':lr_coef}).to_csv(out/'results/logistic_coefficients.csv',index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    colors = ['#147d92','#e5a334','#705db5']
    fig, ax = plt.subplots(figsize=(7.8,3.8))
    counts = [int((y==0).sum()),int(y.sum())]
    bars=ax.bar(['Legitimate','Fraud'],counts,color=['#147d92','#e5a334'])
    ax.set_yscale('log');ax.set_ylim(1,1000000);ax.set_ylabel('Transactions (log scale)')
    for bar,n in zip(bars,counts): ax.text(bar.get_x()+bar.get_width()/2,n*1.2,f'{n:,}',ha='center')
    ax.set_title('Class distribution after exact duplicate removal')
    fig.tight_layout();fig.savefig(out/'figures/class_distribution.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,3.8))
    for c,label,color in [(0,'Legitimate','#147d92'),(1,'Fraud','#e5a334')]:
        ax.hist(np.log1p(Xtr.loc[ytr==c,'Amount']),bins=35,density=True,histtype='step',linewidth=2,label=label,color=color)
    ax.set_xlabel('log(1 + Amount)');ax.set_ylabel('Density');ax.legend();ax.set_title('Training transaction amounts by class')
    fig.tight_layout();fig.savefig(out/'figures/amount_distribution.png');plt.close(fig)
    for kind in ['pr','roc']:
        fig,ax=plt.subplots(figsize=(7.8,4.4))
        for row,color in zip(results,colors):
            name=row['model'];c=curve_data[name]
            if kind=='pr': ax.plot(c['recall'],c['precision'],label=f"{name} (AP {row['average_precision']:.3f})",color=color)
            else: ax.plot(c['fpr'],c['tpr'],label=f"{name} (AUC {row['roc_auc']:.3f})",color=color)
        if kind=='pr':
            ax.axhline(yt.mean(),color='gray',ls='--',label=f'Prevalence {yt.mean():.4f}')
            ax.set(xlabel='Recall',ylabel='Precision',title='Held-out test precision-recall curves')
        else:
            ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out test ROC curves')
        ax.legend(fontsize=9);ax.set_xlim(0,1);ax.set_ylim(0,1.02)
        fig.tight_layout();fig.savefig(out/f'figures/{kind}_curves.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for row,ax in zip(results,axes):
        cm=np.array([[row['tn'],row['fp']],[row['fn'],row['tp']]])
        ConfusionMatrixDisplay(cm,display_labels=['Legit','Fraud']).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d')
        ax.set_title(row['model'],fontsize=11)
    fig.tight_layout();fig.savefig(out/'figures/confusion_matrices.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,4.1))
    top=importance.head(10).iloc[::-1]
    ax.barh(top.feature,top.importance,color='#147d92')
    ax.set(xlabel='Mean decrease in impurity (training model)',title='Random Forest feature importance')
    fig.tight_layout();fig.savefig(out/'figures/feature_importance.png');plt.close(fig)
    # Integrity checks on results and output totals, beyond library call success.
    assert all(r['tn']+r['fp']+r['fn']+r['tp']==len(yt) for r in results)
    assert all(r['tp']+r['fn']==int(yt.sum()) for r in results)
    assert len(data)==sum(s['rows'] for s in splits)
    print(pd.DataFrame(results).to_string(index=False),flush=True)
    print('Analysis complete. All integrity checks passed.',flush=True)

if __name__ == '__main__':
    main()
