# CSE 4309 live demo

The supplied instructor guide permits a notebook or Python implementation. It requests selected code sections, 2-3 test cases or a small evaluation batch, real output, and interpretation. Suggested flow: 5-7 minutes of slides, 5-8 minutes of demonstration, then questions. Follow any separate course timing instructions.

## Fast offline demonstration

From this repository folder, with the Python environment activated:

```text
python -m pip install -r requirements-demo.txt
python source/demo.py
```

Install packages before class. The demo itself needs no internet or full dataset. It uses the included saved models and three small test examples in `results/demo_cases.csv`. The Random Forest training artifact is under 1 MB. Use the pinned environment for these joblib artifacts, and load only model files you trust.

Open **Fraud_Detection_Live_Demo.ipynb** in Jupyter or VS Code, select this environment, and run all cells. It includes saved execution outputs, actual code inspection, real-time model inference, and a recomputation of the full test metrics from recorded predictions. For Jupyter's browser interface, install `jupyterlab` separately and run `jupyter lab`.

## Three cases to explain

These are deliberately selected examples from the held-out test set, not a random sample and not the basis for performance metrics. The case type is defined by Random Forest's result. Selection chooses the earliest source row for each category.

| Source row | Actual class | Forest score | Threshold | Forest prediction | Interpretation |
| --- | --- | --- | --- | --- | --- |
| 541 | Fraud | 0.550616 | 0.349667 | Fraud | True positive: correctly flagged |
| 8625 | Legitimate | 0.378123 | 0.349667 | Fraud | False positive: unnecessary alert |
| 4920 | Fraud | 0.332012 | 0.349667 | Legitimate | False negative: fraud missed |

Logistic Regression and Decision Tree both correctly classify row 8625, but both miss row 4920. This helps explain why a model that leads overall can still make individual mistakes. Scores are model outputs, not calibrated probabilities or guarantees. Displayed values are rounded; inference uses full-precision thresholds.

## Live sequence

1. **Data and preparation:** show the dataset summary, saved split counts, and actual preparation code. Explain duplicate removal and training-only scaling.
2. **Models:** show `make_estimator` and the selected settings. Explain why training was performed before class.
3. **Inference:** execute the three-case prediction cell. Point out the score, threshold, predicted label, actual label, and error type.
4. **Evaluation:** recompute full test metrics from saved predictions. Show why three demonstration cases cannot estimate population performance.
5. **Trade-off:** show the false alert and missed fraud. Explain the fixed threshold and the validation/test separation.

## Rebuild the saved models before class if needed

```text
python source/run_analysis.py
python source/prepare_demo.py --data data/creditcard.csv
python source/demo.py
```

`prepare_demo.py` rebuilds only the three selected models using the recorded training rows and shared constructors. It checks all 56,746 held-out scores against the recorded experiment to absolute tolerance 1e-12, and checks that all binary predictions match exactly. This preparation was executed successfully for the included artifacts. Training never uses the test labels for fitting or selection.

## Readiness

Run the notebook in your own environment before class. Bring the repository, saved models, installed packages, and charger/adapter. All five members should present or demonstrate a meaningful part. Proposed roles in the slides must be replaced with truthful completed contributions when available.

The original supplied presentation template and instructor PDF are not redistributed in this public repository. The adapted presentation retains the template's design and relevant sections.
