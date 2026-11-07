"""Fast offline demonstration using saved models and three real test cases."""
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def predict_cases(root=ROOT):
    cases=pd.read_csv(root/'results/demo_cases.csv')
    records=[]
    for filename in ['logistic_regression','decision_tree','random_forest']:
        bundle=joblib.load(root/'models'/f'{filename}.joblib')
        scores=bundle['model'].predict_proba(cases[bundle['features']])[:,1]
        for row,score in zip(cases.itertuples(),scores):
            prediction=int(score>=bundle['threshold'])
            records.append({'source_row':row.source_row,'case_type':row.case_type,
                'actual':int(row.Class),'model':bundle['name'],'score':float(score),
                'threshold':bundle['threshold'],'prediction':prediction,
                'correct':prediction==int(row.Class)})
    return pd.DataFrame(records)

def main():
    result=predict_cases()
    print('Three selected held-out cases: true positive, false positive, false negative for Random Forest.')
    print('Scores are model outputs, not calibrated fraud probabilities. 1=fraud; 0=legitimate.\n')
    print(result.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    result.to_csv(ROOT/'results/demo_output.csv',index=False)

if __name__=='__main__':main()
