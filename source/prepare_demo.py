"""Rebuild selected models from recorded training rows and verify frozen test scores."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from run_analysis import make_estimator

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/creditcard.csv')
    args=parser.parse_args()
    summary=json.loads((ROOT/'results/summary.json').read_text())
    assert hashlib.sha256(args.data.read_bytes()).hexdigest()==summary['sha256'], 'Dataset differs from recorded experiment'
    raw=pd.read_csv(args.data)
    splits=pd.read_csv(ROOT/'results/split_assignments.csv')
    train_idx=splits.loc[splits.split=='train','source_row'].to_numpy()
    predictions=pd.read_csv(ROOT/'results/test_predictions.csv')
    test_idx=predictions.source_row.to_numpy()
    X=raw.drop(columns='Class')
    (ROOT/'models').mkdir(exist_ok=True)
    bundles=[]
    for choice in summary['selected']:
        name=choice['model']
        weight=None if choice['class_weight']=='None' else choice['class_weight']
        model=make_estimator(name,weight)
        print('Preparing '+name,flush=True)
        model.fit(X.loc[train_idx],raw.loc[train_idx,'Class'])
        score=model.predict_proba(X.loc[test_idx])[:,1]
        np.testing.assert_allclose(score,predictions[name+'_score'],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(score>=choice['threshold'],predictions[name+'_prediction'])
        filename=name.lower().replace(' ','_')+'.joblib'
        bundle={'model':model,'threshold':choice['threshold'],'name':name,
            'features':X.columns.tolist(),'dataset_sha256':summary['sha256'],'versions':summary['versions']}
        joblib.dump(bundle,ROOT/'models'/filename,compress=3)
        bundles.append({'model':name,'file':filename,'bytes':(ROOT/'models'/filename).stat().st_size})
    rf='Random Forest_prediction'
    # Deliberately selected examples illustrate correct detection AND errors.
    # They are not a representative sample and never enter performance estimates.
    masks={'true_positive':(predictions.actual==1)&(predictions[rf]==1),
           'false_positive':(predictions.actual==0)&(predictions[rf]==1),
           'false_negative':(predictions.actual==1)&(predictions[rf]==0)}
    chosen=[]
    for kind,mask in masks.items():
        source_row=int(predictions.loc[mask,'source_row'].min())
        chosen.append({'case_type':kind,'source_row':source_row,**raw.loc[source_row].to_dict()})
    pd.DataFrame(chosen).to_csv(ROOT/'results/demo_cases.csv',index=False)
    (ROOT/'results/demo_manifest.json').write_text(json.dumps({'models':bundles,
        'selection':'Earliest source row in each Random Forest category: TP, FP, FN. Illustrative, not random.',
        'verified':'All three saved models reproduce all recorded test scores to absolute tolerance 1e-12 and all binary predictions exactly.'},indent=2))
    print('Saved models verified against all 56,746 test predictions.',flush=True)

if __name__=='__main__':main()
