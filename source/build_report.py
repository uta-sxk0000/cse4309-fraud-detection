"""Build the PDF report and editable Markdown from saved experiment results."""
from pathlib import Path
import json
from html import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

ROOT=Path(__file__).resolve().parents[1]
S=json.loads((ROOT/'results/summary.json').read_text())
R=S['test_results']; V=S['selected']
chosen=next(r for r in R if r['model']==S['recommended_model'])
best_test=max(R,key=lambda r:r['f1'])
imp=pd.read_csv(ROOT/'results/random_forest_importance.csv')
val=pd.read_csv(ROOT/'results/validation_candidates.csv',keep_default_na=False)
defaults=pd.read_csv(ROOT/'results/test_default_threshold.csv')
TEAM=['Sagar Khadka','Bipin Adhikari','Bardan Karki','Samiksha Rijal','Samin Rijal']
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=31,leading=36,textColor=colors.HexColor('#102d44'),spaceAfter=18))
styles.add(ParagraphStyle(name='ReportSub',fontName='Helvetica',fontSize=14,leading=20,textColor=colors.HexColor('#147d92'),spaceAfter=12))
styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9))
styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8.4,leading=11,spaceAfter=7))
styles['Heading1'].textColor=colors.HexColor('#102d44')
styles['Heading1'].fontSize=20;styles['Heading1'].leading=25
styles['Heading2'].textColor=colors.HexColor('#147d92')
story=[];md=[]

def p(text,small=False):
    story.append(Paragraph(escape(text),styles['SmallCustom' if small else 'BodyCustom']))
    md.append(text+'\n')
def h(text):
    story.append(Paragraph(escape(text),styles['Heading1']));md.append('# '+text+'\n')
def sub(text):
    story.append(Paragraph(escape(text),styles['Heading2']));md.append('## '+text+'\n')
def page():story.append(PageBreak())
def table(rows,widths):
    cells=[[Paragraph(escape(str(c)),styles['SmallCustom']) for c in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebf0')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#147d92')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);story.append(Spacer(1,10))
    md.append('| '+' | '.join(map(str,rows[0]))+' |\n| '+' | '.join(['---']*len(rows[0]))+' |')
    md.extend('| '+' | '.join(map(str,row))+' |' for row in rows[1:]);md.append('')
def fig(name,width=490,height=None,caption=''):
    from PIL import Image as PILImage
    path=ROOT/'figures'/name
    with PILImage.open(path) as im: ratio=im.height/im.width
    story.append(Image(str(path),width=width,height=height or width*ratio))
    if caption:p(caption,True)
    md.append(f'![{caption}](figures/{name})\n')

story.append(Spacer(1,40))
story.append(Paragraph('Fraud Detection',styles['ReportTitle']))
story.append(Paragraph('4309-04 | Machine Learning Team Project',styles['ReportSub']))
p('Comparison of Logistic Regression, Decision Tree, and Random Forest for identifying potentially fraudulent credit card transactions.')
story.append(Spacer(1,16))
for member in TEAM:p(member)
story.append(Spacer(1,22))
sub('Abstract')
p(f'This investigation applies three supervised classification algorithms to the recommended Credit Card Fraud Detection dataset. After removing {S["duplicates_removed"]:,} exact duplicate rows, {S["cleaned_rows"]:,} transactions remain, including {S["cleaned_fraud"]} fraud cases. A stratified 60% training, 20% validation, and 20% test design separates fitting, model selection, and final evaluation. Each algorithm is evaluated with and without balanced class weights. Validation average precision selects the weighting treatment within each algorithm, and validation F1 determines its threshold and the recommended model family.')
p(f'The validation-selected recommendation is {chosen["model"]}. On the held-out test set, it achieves precision {chosen["precision"]:.3f}, recall {chosen["recall"]:.3f}, F1 {chosen["f1"]:.3f}, and average precision {chosen["average_precision"]:.3f}. It detects {chosen["tp"]} of 95 fraud cases with {chosen["fp"]} false alerts. Results support a benchmark comparison, with important limits from the small number of fraud cases, anonymous features, and a random rather than chronological split.')
p('Analysis date: November 7, 2026. Project code supplied by the team. The assignment PDF calls this Topic 5. Contribution records require team confirmation.',True)
page()
h('1. Problem and objectives')
p('The problem is binary classification: distinguish potentially fraudulent transactions from legitimate transactions using the available transaction characteristics. Missing fraud and incorrectly flagging a legitimate purchase are different errors. The evaluation therefore measures both fraud detection and false alerts.')
sub('Research question')
p('Among Logistic Regression, Decision Tree, and Random Forest, which provides the strongest validation F1 after selecting class weighting by average precision and tuning its decision threshold, and how does that frozen choice perform on unseen transactions?')
sub('Objectives')
for text in [
    'Describe the data and inspect quality issues, class imbalance, and relevant distributions.',
    'Implement and compare three appropriate algorithms using reproducible preprocessing and identical data partitions.',
    'Measure the effect of balanced class weights and explain the decision-threshold trade-off.',
    'Evaluate accuracy, precision, recall, F1, confusion matrices, ROC-AUC, and average precision, then justify a recommendation.',
    'Discuss feature importance, uncertainty, dataset limitations, and future improvements.']:p(text)
sub('Scope and baseline')
p('The target is Class, with 1 denoting fraud and 0 denoting legitimate transactions. The project evaluates an offline benchmark, not a live payment-blocking service. F1 is the study objective because no verified fraud cost or investigation capacity was supplied. It balances precision and recall, but it does not minimize a known monetary loss.')
p(f'An always-legitimate classifier achieves {S["baseline"]["accuracy"]*100:.3f}% test accuracy while detecting zero fraud cases. This baseline demonstrates why accuracy alone cannot support a fraud-detection recommendation.')
sub('Distinctive approach')
p('The study combines exact duplicate removal, an explicit weighting comparison, validation-only threshold selection, and honest test evaluation. If another team uses the same dataset, confirm that this question and approach are meaningfully different, as required by the assignment.')
page()
h('2. Dataset and exploratory review')
p('The original source is the ULB Machine Learning Group Credit Card Fraud Detection dataset on Kaggle [1]. The file used here comes from the public mirror referenced in TensorFlow\'s official tutorial [2]. The source describes European credit card transactions from September 2013. V1 through V28 are anonymized principal components, while Time and Amount retain their original meanings [1].')
table([['Property','Observed value'],['Original rows / fraud',f'{S["raw_rows"]:,} / {S["raw_fraud"]}'],['Cleaned rows / fraud',f'{S["cleaned_rows"]:,} / {S["cleaned_fraud"]}'],['Predictors / target','30 predictors / Class'],['Missing values / exact duplicate rows',f'0 / {S["duplicates_removed"]:,}'],['Cleaned fraud prevalence',f'{S["cleaned_fraud"]/S["cleaned_rows"]*100:.4f}%']], [250,250])
table([['Columns','Type and interpretation'],['Time','Numeric: seconds elapsed from the first transaction'],['V1 to V28','Numeric: anonymized PCA components; original meanings unavailable'],['Amount','Numeric: transaction amount'],['Class','Integer binary label: 0 legitimate, 1 fraud']], [100,400])
fig('class_distribution.png',width=450,caption='Figure 1. Class imbalance after exact duplicate removal. A log scale makes the minority class visible.')
p('The full per-column count, mean, standard deviation, quantiles, range, dtype, and missing count are available in results/data_dictionary.csv. All values are numeric and finite. No categorical encoding is needed.',True)
page()
h('3. Preprocessing and data partitions')
p(f'Exact duplicate rows were removed before splitting to prevent identical records from appearing in multiple partitions. This removes {S["duplicates_removed"]:,} rows, including {S["duplicate_fraud_removed"]} fraud-labeled duplicates. Identical rows could represent separate real events, so deduplication is an explicit modeling assumption. No missing values, non-finite values, negative amounts, or negative times were found.')
p('All 30 predictors are retained. No feature selection, synthetic oversampling, undersampling, or outlier deletion is performed. Unusual transactions can carry useful fraud signals. Time is retained as an available benchmark feature, but may encode patterns that do not transfer to a later period. The supplied PCA components are used as released; their original fitting procedure cannot be audited.')
p('Logistic Regression uses a StandardScaler fitted only on training data inside a pipeline. The trees use the unscaled numeric inputs because their splits operate on feature ordering. Scaling effects were not separately ablated. Balanced class weights alter the training objective without changing validation or test prevalence [3].')
table([['Split','Transactions','Legitimate','Fraud']]+[[s['split'].title(),f'{s["rows"]:,}',f'{s["legitimate"]:,}',s['fraud']] for s in S['splits']],[140,120,120,120])
p('The initial split reserves 20% for testing. The remaining 80% is divided 75/25 into training and validation, yielding approximately 60/20/20 overall. Both splits are stratified by Class and use random seed 42. This leaves validation data for threshold choice while preserving a separate final test set. Split row identities are saved for audit.')
fig('amount_distribution.png',width=450,caption='Figure 2. Training-only amount distributions. The log transform is for visualization; models use Amount as documented above.')
p('A random split tests generalization within this historical sample. It does not simulate predicting future transactions or prevent unknown cardholder overlap. A chronological evaluation is an important next experiment.',True)
page()
h('4. Algorithms and experiment design')
table([['Algorithm','Why appropriate','Configuration'],
 ['Logistic Regression','Linear classification reference; standardized coefficients support inspection.','C=1; L2 default; lbfgs; max_iter=2000; StandardScaler'],
 ['Decision Tree','Models nonlinear rules and interactions; controlled complexity reduces overfitting.','max_depth=8; min_samples_leaf=10; Gini default'],
 ['Random Forest','Averages diverse trees to reduce single-tree variance and capture nonlinear interactions.','150 trees; max_depth=16; min_samples_leaf=2; max_features=sqrt; bootstrap default']], [112,194,194])
p('Each algorithm is fit twice: class_weight=None and class_weight="balanced". Balanced weighting gives minority examples greater training influence based on inverse class frequency. All six fits use the same training partition. Random state is 42. Random Forest uses four workers; elapsed times are machine dependent. The full constructor settings are in source/run_analysis.py.')
sub('Selection procedure fixed before test evaluation')
for text in [
 '1. Fit each candidate on the training partition only.',
 '2. Within each algorithm, select the weighting treatment with the highest validation average precision (AP). This evaluates score ranking across thresholds.',
 '3. For that candidate, select the threshold that maximizes validation F1. Predictions use score >= threshold; ties in F1 choose the highest threshold.',
 '4. Recommend the algorithm with the highest selected validation F1.',
 '5. Freeze models and thresholds, then evaluate all three on the test partition. Do not refit on validation data, because that could shift the score scale underlying the chosen threshold.']:p(text)
sub('Metric definitions')
p('Precision = TP/(TP+FP): the fraction of alerts that are fraud. Recall = TP/(TP+FN): the fraction of fraud found. F1 = 2PR/(P+R). Accuracy = (TP+TN)/N. ROC-AUC summarizes ranking against the false-positive rate. AP summarizes precision as recall increases; it is the scikit-learn average_precision_score, not trapezoidal area under the precision-recall curve [4].')
p('No broad hyperparameter search or cross-validation was conducted. Depth, leaf size, and tree count were fixed for a tractable semester baseline. The small validation set can make threshold selection unstable, and its optimized F1 is not an unbiased final performance estimate. Only the separate test results serve that purpose [5, 6].')
page()
h('5. Validation and class weighting')
table([['Algorithm','Weights','AP','Best F1','Threshold']]+[[r.model,r.class_weight,f'{r.average_precision:.4f}',f'{r.f1:.4f}',f'{r.threshold:.8f}'] for r in val.itertuples()],[140,70,72,72,146])
p('Table 1. Validation results for six candidates. Each F1 uses that candidate\'s validation-optimal threshold. Weighting is selected by AP, so the candidate with the highest threshold-specific F1 need not be selected.',True)
for name in ['Logistic Regression','Decision Tree','Random Forest']:
    pair=val[val.model==name].set_index('class_weight')
    delta=pair.loc['balanced','average_precision']-pair.loc['None','average_precision']
    selected=next(v for v in V if v['model']==name)
    p(f'{name}: balanced weighting changes validation AP by {delta:+.4f}. The selected treatment is {selected["class_weight"]}, with threshold {selected["threshold"]:.8f} and validation F1 {selected["f1"]:.4f}.')
sub('Why weighting may not improve performance')
p('Class weighting changes how errors influence fitting, but can also increase false alerts or change score calibration. Its effect is empirical rather than guaranteed. In this experiment, the validation ranking metric determines whether the weighting treatment is retained. Scores from weighted models should not automatically be interpreted as calibrated fraud probabilities.')
sub('Recommendation before opening the test results')
p(f'{S["recommended_model"]} has the highest F1 among the selected validation candidates. That rule establishes the recommendation before test evaluation. Test results are reported for comparison without using them to retune thresholds or replace the preselected model.')
p('The balanced Logistic Regression threshold can lie extremely close to 1. This reflects the score distribution and weighting, not near-certain real-world fraud probabilities. The full-precision thresholds are retained in the results files.',True)
page()
h('6. Held-out test results')
table([['Algorithm','Accuracy','Precision','Recall','F1','AP','ROC-AUC']]+[[r['model'],f'{r["accuracy"]:.5f}',f'{r["precision"]:.4f}',f'{r["recall"]:.4f}',f'{r["f1"]:.4f}',f'{r["average_precision"]:.4f}',f'{r["roc_auc"]:.4f}'] for r in R],[123,65,62,60,60,60,70])
p('Table 2. Final test metrics at validation-selected thresholds. Precision, recall, and F1 refer to fraud (Class=1), not macro or weighted averages.',True)
table([['Algorithm','True negatives','False alerts','Missed fraud','Fraud found']]+[[r['model'],r['tn'],r['fp'],r['fn'],r['tp']] for r in R],[140,90,90,90,90])
p('Table 3. Confusion counts on 56,746 test transactions, including 95 fraud cases. Rows of each confusion matrix below are actual classes; columns are predicted classes.',True)
fig('confusion_matrices.png',width=500,caption='Figure 3. Confusion matrices for the three selected models.')
sub('Measured comparison')
p(f'{best_test["model"]} has the highest observed test F1 ({best_test["f1"]:.4f}). The preselected recommendation, {chosen["model"]}, finds {chosen["tp"]} fraud cases, misses {chosen["fn"]}, and generates {chosen["fp"]} false alerts. Its {chosen["tp"]+chosen["fp"]} alerts correspond to {(chosen["tp"]+chosen["fp"])/56746*10000:.1f} alerts per 10,000 transactions.')
p(f'The recommended model\'s recall is {chosen["recall"]*100:.1f}%, with an approximate 95% Wilson interval of {chosen["recall_ci_low"]*100:.1f}% to {chosen["recall_ci_high"]*100:.1f}%. This interval conditions on the fixed model and treats the 95 fraud outcomes as independent trials. It does not include model-selection uncertainty or temporal dependence.')
page()
h('7. Ranking performance and thresholds')
fig('pr_curves.png',width=475,caption='Figure 4. Precision-recall curves computed from test scores. AP measures ranking independently of the chosen threshold.')
fig('roc_curves.png',width=475,caption='Figure 5. ROC curves. High ROC-AUC alone can hide an operationally costly number of false alerts in a large legitimate class.')
p('The precision-recall view directly emphasizes minority-class detection. The dashed prevalence line is a reference for uninformative ranking. These curves describe the frozen models on the test data; they are not used to choose new operating points.',True)
page()
h('8. Interpretation and limitations')
sub('Default versus selected thresholds')
table([['Algorithm','Test F1 at 0.5','Test F1 at selected threshold']]+[[r['model'],f'{defaults.loc[defaults.model==r["model"],"f1"].iloc[0]:.4f}',f'{r["f1"]:.4f}'] for r in R],[170,160,170])
p('Table 4. A descriptive threshold comparison. Both the default threshold and the selection procedure were defined before test evaluation. Threshold choice changes alert volume and error balance; it does not change the model\'s score ranking or AP.',True)
p('Random Forest has slightly higher test F1 at 0.5 (0.8263) than at the selected threshold (0.8208). Lowering its threshold finds two additional fraud cases but adds four false alerts. This illustrates that validation tuning need not improve test F1. The recommendation retains the validation-selected threshold rather than choosing a new one using test results.',True)
sub('Feature importance')
p('The five largest Random Forest impurity importances are '+', '.join(f'{row.feature} ({row.importance:.3f})' for row in imp.head(5).itertuples())+'. These describe how the fitted trees reduce training impurity. They are not causal effects and can favor features with more possible splits. The anonymized component names prevent explanations in terms of original customer characteristics.')
fig('feature_importance.png',width=330,caption='Figure 6. Top ten Random Forest feature importances. Coefficients for standardized Logistic Regression predictors are also supplied.')
p('An ensemble can outperform a single tree by averaging unstable decisions, while a linear model may miss interactions. These are plausible explanations, not established causal findings. Fixed hyperparameters, weighting treatment, and limited validation fraud examples can all affect the observed ranking.')
p('Limits: one historical dataset; only 95 test fraud cases; one random split; unknown customer grouping; released PCA transformation; no probability calibration; no chronological evaluation; no measured review cost or monetary-loss model. Exact duplicate removal changes the population. Results do not demonstrate current banking performance or fairness across demographic groups.')
page()
h('9. Conclusion and reproducibility')
p(f'This project implements three algorithms for credit card fraud classification and evaluates them beyond accuracy. The validation procedure recommends {chosen["model"]}. Its test precision of {chosen["precision"]:.4f}, recall of {chosen["recall"]:.4f}, and F1 of {chosen["f1"]:.4f} support its use as the strongest candidate under this study\'s selection rule. The result is a benchmark recommendation, not a production deployment decision.')
p('The main finding is that meaningful fraud evaluation requires class-specific metrics and an explicit threshold policy. Class weighting must be measured rather than assumed beneficial. A model with nearly perfect accuracy can still miss many fraud cases, while higher recall may increase unnecessary investigations.')
p('Future work should evaluate chronological splits, repeated or nested validation, threshold stability, permutation importance, calibration, and decision thresholds based on documented review capacity or loss costs. Additional data spanning later periods would test drift and external validity. No unperformed experiment is presented as a completed result.')
sub('Reproduction instructions')
p('Use Python 3.12. Install requirements.txt, then run: python source/run_analysis.py. The script downloads the dataset if data/creditcard.csv is absent. For a local file, use --data PATH. Use python source/build_report.py to regenerate this report after results exist. README.md documents the full workflow, and the notebook provides an interactive entry point.')
p('The delivered metrics and figures were produced by the included script. results/test_predictions.csv supports independent metric recomputation, results/split_assignments.csv identifies the exact partitions, and results/summary.json records the data fingerprint and software versions. Hardware-dependent timing can vary.')
sub('Dataset provenance')
p('Original source: '+S['original_source'],True)
p('Download mirror: '+S['download_url'],True)
p('SHA-256: '+S['sha256'],True)
p('Recorded environment: Python '+S['python']+'; '+', '.join(k+' '+v for k,v in S['versions'].items())+'.',True)
sub('Team contribution record')
p('Team members: '+', '.join(TEAM)+'. Actual individual contributions were not provided. CONTRIBUTIONS.md contains suggested responsibilities and an unfilled confirmation table. It must be updated to describe work each member actually completed. The code, analysis, report, and presentation were prepared with AI assistance; the team must review and understand them before presenting.')
page()
h('References')
refs=[
 ('1','ULB Machine Learning Group. Credit Card Fraud Detection. Kaggle.','https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'),
 ('2','TensorFlow. Classification on imbalanced data. Dataset mirror and imbalance context.','https://www.tensorflow.org/tutorials/structured_data/imbalanced_data'),
 ('3','scikit-learn. RandomForestClassifier. Class weighting and estimator parameters.','https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'),
 ('4','scikit-learn. average_precision_score. AP definition.','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html'),
 ('5','scikit-learn. Common pitfalls and recommended practices. Data leakage and pipelines.','https://scikit-learn.org/stable/common_pitfalls.html'),
 ('6','scikit-learn. Tuning the decision threshold for class prediction.','https://scikit-learn.org/stable/modules/classification_threshold.html')]
for n,title,url in refs:
    p(f'[{n}] {title}')
    story.append(Paragraph(f'<link href="{url}" color="#147d92">{escape(url)}</link>',styles['SmallCustom']))
    md.append(url+'\n');story.append(Spacer(1,8))
p('All web references accessed November 7, 2026. Assignment source: Machine Learning Team Project - Student.pdf, supplied by the user, especially Topic 5 and the general requirements on pages 8-14.',True)
sub('Evidence files')
p('Dataset quality: results/data_dictionary.csv. Validation experiment: results/validation_candidates.csv. Final comparison: results/test_metrics.csv. Default thresholds: results/test_default_threshold.csv. Full scores and predictions: results/test_predictions.csv. Feature analysis: results/random_forest_importance.csv and results/logistic_coefficients.csv.',True)

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#dcebf0'));canvas.line(48,42,564,42)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#526270'))
    canvas.drawString(48,29,'4309-04  /  Fraud Detection');canvas.drawRightString(564,29,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(ROOT/'Fraud_Detection_Report.pdf'),pagesize=(612,792),rightMargin=56,leftMargin=56,topMargin=43,bottomMargin=57,
    title='4309-04 Fraud Detection',author=', '.join(TEAM))
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(ROOT/'Fraud_Detection_Report.md').write_text('\n'.join(md),encoding='utf-8')
print('Report created.')
