# 4309-04 - Fraud Detection

Team: Sagar Khadka, Bipin Adhikari, Bardan Karki, Samiksha Rijal, Samin Rijal.

## Start here

- [Course-template presentation](Fraud_Detection_Template_Presentation.pptx)
- [Executed live-demo notebook](Fraud_Detection_Live_Demo.ipynb)
- [Live demo instructions](DEMO_GUIDE.md)
- [Final report](Fraud_Detection_Report.pdf)

For the fast demo, install `requirements-demo.txt` and run `python source/demo.py`. All three trained models and three real test cases are included, so inference works offline without downloading the full dataset. The live notebook has been executed successfully with outputs saved.

This package contains a completed, executed benchmark study of Logistic Regression, Decision Tree, and Random Forest. Results are measured, not illustrative. Random Forest was selected using validation data and obtained test precision 0.9103, recall 0.7474, and F1 0.8208. It found 71 of 95 fraud cases and generated 7 false alerts.

## Files

- `Fraud_Detection_Report.pdf`: report with methods, measured results, figures, limitations, references, and contribution status.
- `Fraud_Detection_Report.md`: editable report text.
- `Fraud_Detection_Template_Presentation.pptx`: current 14-slide deck adapted from the instructor's template, with live-demo and code-walkthrough sections.
- `Fraud_Detection_Presentation.pptx`: earlier supplementary deck with additional metric comparisons.
- `Fraud_Detection_Live_Demo.ipynb`: executed notebook for the 2-3-case live demonstration and code explanation.
- `models/`: three saved fitted models with their feature order and validation thresholds.
- `source/prepare_demo.py` and `source/demo.py`: verified model reconstruction and fast offline inference.
- `source/run_analysis.py`: download, quality inspection, partitioning, six candidate fits, final evaluation, and charts.
- `source/build_report.py`: regenerate the report from results.
- `Fraud_Detection_Walkthrough.ipynb`: notebook entry point for running and reviewing the study.
- `requirements.txt`: pinned top-level dependencies used in the run.
- `results/`: metrics, predictions, split identities, feature summaries, curves, and provenance.
- `figures/`: report-quality charts.
- `DATASET.md`: original source, public mirror, acquisition instructions, and fingerprint.
- `CONTRIBUTIONS.md`: suggested responsibilities and an actual-contribution record to complete.
- `Project_Selection.md`: concise project selection statement.
- `Presentation_Notes.md`: speaking guide and likely questions.

## Reproduce the analysis

Use Python 3.12. Commands below assume your terminal is in the extracted `fraud_detection` folder.

```text
python -m venv .venv
```

Activate on Windows PowerShell:

```text
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```text
source .venv/bin/activate
```

Then:

```text
python -m pip install -r requirements.txt
python source/run_analysis.py
python source/build_report.py
```

If activation is unavailable on Windows, call `.venv\Scripts\python.exe` instead of `python` in subsequent commands. The first analysis run downloads approximately 150 MB to `data/creditcard.csv`. Internet is only needed for dependencies and the initial dataset download. Computation can take several minutes depending on hardware. No Kaggle API credentials are needed for the public mirror.

To use a previously downloaded file:

```text
python source/run_analysis.py --data "C:/path/to/creditcard.csv"
```

To keep new results separate:

```text
python source/run_analysis.py --data "C:/path/to/creditcard.csv" --output rerun
```

The report builder reads the package's `results/` folder, not `rerun/`. The notebook requires Jupyter or VS Code notebook support, which is optional and not needed for the scripts. Select the same Python environment in the notebook. If needed, install `jupyterlab ipykernel` separately.

## Method choices

Exact duplicate rows are removed before splitting. The cleaned 283,726 rows contain 473 fraud cases. Seed 42 creates stratified 60/20/20 train/validation/test partitions. Standardization is fitted only to training data and only used for Logistic Regression. No sampling, imputation, encoding, feature selection, or outlier deletion is performed. All 30 input columns are retained.

Each family compares `class_weight=None` with `class_weight='balanced'`. Validation AP chooses its treatment. Validation F1 chooses its threshold, and then the algorithm family. The three selected models are frozen before test evaluation. Models remain fitted on the training partition so validation-selected thresholds remain attached to the fitted score scale. Full constructor parameters are in the analysis script.

The supplied output chose no weighting for all three families. Unweighted training does not mean imbalance was ignored: prevalence-aware metrics, validation threshold selection, stratified splitting, and an empirical weighting comparison address it explicitly.

## Interpreting the result

The test set contains only 95 fraud cases. Random Forest has the highest measured F1, but one split is insufficient to prove a universal winner. Its approximate 95% recall interval is 65.2% to 82.4%. Random splits do not simulate future deployment, and PCA feature names limit explanations. See the report for the full discussion.

Average precision (AP) uses scikit-learn's non-interpolated definition. It is not trapezoidal PR-AUC. Accuracy, precision, recall, F1, ROC-AUC, AP, and confusion counts use the same test data. The positive class is fraud. Threshold equality predicts fraud.

## Before presenting or submitting

Review the code and rerun it if your course expects your own execution. Replace the proposed contribution plan with truthful descriptions of completed work. Confirm the project code 4309-04 with your course, because the supplied PDF names Fraud Detection as Topic 5. If another team uses the same dataset, confirm a meaningfully different question or approach. Follow your instructor's expectations for acknowledging AI assistance. No instructor name, deadline, or completed individual work was invented.

The analysis, code, report, and presentation were prepared with AI assistance. The project is published on GitHub at the team's request. No files were submitted to the instructor or sent to team members.
