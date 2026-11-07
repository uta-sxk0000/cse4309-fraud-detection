# Presentation guide

**Current class presentation:** use `Fraud_Detection_Template_Presentation.pptx` and `DEMO_GUIDE.md`. The sections below are extended notes for the earlier supplementary deck, so their slide numbers do not match the instructor-template version.

For the template deck, use approximately 5-7 minutes for slides and 5-8 minutes for the live notebook, as suggested by the supplied demo guide. Sagar can introduce slides 1-2; Bipin can explain slides 3-5; Bardan can explain slides 6-7 and the model constructors; Samiksha can explain slides 8-10; Samin can lead slides 11-14 and the live demo. These are proposed speaking assignments, not completed contribution claims.

Suggested duration: 10-12 minutes plus questions. Speaking assignments below are suggestions, not records of completed contributions.

## Suggested speaking sections

- Sagar Khadka: slides 1-2, introduction and research question.
- Bipin Adhikari: slides 3-4, dataset and preprocessing.
- Bardan Karki: slides 5-6, algorithms and selection procedure.
- Samiksha Rijal: slides 7-10, results and threshold trade-offs.
- Samin Rijal: slides 11-14, feature importance, limitations, and conclusion.

## Speaking guide

1. **Fraud Detection:** Introduce the team and the three algorithms. The task is binary classification using transaction characteristics.
2. **Research question:** Explain the two errors: missed fraud and false alerts. F1 is a study objective because no actual cost function was supplied.
3. **Dataset:** Begin with 284,807 rows and 492 fraud cases. Remove 1,081 exact duplicates, leaving 283,726 transactions and 473 fraud cases. A classifier predicting all legitimate has 99.833% test accuracy but zero fraud recall.
4. **Partitions:** Seed 42 and stratification create 60/20/20 train/validation/test sets. Fit the scaler only on training data. Retain all 30 features and do not remove unusual transactions simply because they are outliers.
5. **Algorithms:** Logistic Regression provides a linear reference. A tree captures nonlinear rules. A forest averages multiple trees. Each family has ordinary and balanced-weight candidates.
6. **Selection:** Choose weights by validation AP, then choose a threshold by validation F1. Recommend the family with the highest selected validation F1. The test set remains untouched until these choices are frozen.
7. **Metrics chart:** Precision measures how many alerts are fraud, recall measures how much fraud is found, and F1 balances the two. Random Forest leads F1 and recall among the selected test operating points.
8. **Full results:** Random Forest test precision is 0.9103, recall 0.7474, F1 0.8208, AP 0.7991, and ROC-AUC 0.9635. Explain that all metrics describe the same test set.
9. **Errors:** Random Forest finds 71 fraud cases with 7 false alerts; Decision Tree finds 64 with 5 false alerts. The forest gains seven detected cases at the cost of two more false alerts.
10. **Thresholds:** The forest threshold selected on validation is approximately 0.34967. Compared with 0.5, it catches two more test fraud cases but adds four false alerts. Test F1 falls slightly, showing that validation tuning does not guarantee test improvement. Do not silently change the threshold after seeing this.
11. **Features:** V17, V12, and V14 rank highest by forest impurity importance. They are anonymized PCA components, so avoid inventing business meanings or causal claims.
12. **Uncertainty:** There are only 95 test fraud cases. The forest's approximate 95% recall interval is 65.2%-82.4%. Random splitting and historical data limit deployment claims. Chronological evaluation is proposed future work.
13. **Conclusion:** Recommend Random Forest under the fixed study rule. It is a benchmark candidate rather than proof of production readiness. Review the package and complete truthful contributions.
14. **Sources:** Point to the report's links and the scripts. Mention that raw predictions and split identities enable independent verification.

## Likely questions and answers

**Why not use accuracy to choose the model?** The majority class dominates. Always predicting legitimate produces 99.833% accuracy while missing every fraud case.

**Why did you select unweighted models despite severe imbalance?** We tested both treatments on validation data. Ordinary training had higher AP for each family. We still address imbalance through stratified partitions, positive-class metrics, threshold selection, and the explicit weighting comparison.

**Why choose weights by AP and thresholds by F1?** AP compares score rankings across thresholds. F1 selects a particular operating point. This is a declared two-stage design, not necessarily the globally optimal F1 search. For example, balanced Logistic Regression has higher validation F1 at its own threshold, but it loses the AP comparison.

**Why remove duplicates before splitting?** Identical records crossing partitions can inflate evaluation. We treat exact repeated rows as duplicates; they could be distinct real events, so we disclose this assumption.

**Why use 60/20/20 instead of 80/20?** The initial 80% development partition is subdivided to reserve validation data for selection. The final 20% test set stays separate. This follows the assignment's allowance for justified alternative splits.

**Why not train again on training plus validation?** Refitting may change scores, invalidating a threshold learned from the previous fitted model. This study keeps the fitted models fixed. A future nested or out-of-fold procedure could use more data safely.

**Does threshold tuning always improve F1?** No. It maximizes validation F1, which can differ from test F1. Random Forest demonstrates this explicitly.

**Why no SMOTE?** Class weighting is the selected imbalance intervention for this controlled comparison. Synthetic sampling is an optional future experiment and must be restricted to training data or training folds.

**What did scaling improve?** We used scaling for Logistic Regression as part of its pipeline, but did not run a scaling ablation. We cannot claim a measured improvement from scaling alone.

**Does the forest clearly beat every possible alternative?** No. It leads this fixed comparison on this split. Limited fraud examples, fixed settings, and dataset scope prevent a universal claim.

**Can you explain V17 in business terms?** No. Its original feature meaning is unavailable after anonymization. Importance is model-specific and noncausal.

**Was the notebook executed?** The script was executed and produced the supplied results. The notebook is an unexecuted walkthrough for the team to run.

**What did each member do?** Answer with work actually completed and recorded in CONTRIBUTIONS.md. The suggested duties are not verified contribution claims.
