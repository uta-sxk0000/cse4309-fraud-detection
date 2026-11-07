# Dataset source and acquisition

Original dataset: ULB Machine Learning Group, **Credit Card Fraud Detection**.

- Original source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Official TensorFlow tutorial documenting the mirror: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
- CSV used in this experiment: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Run `python source/run_analysis.py` from the project folder to download the file automatically. Alternatively, download `creditcard.csv` from the original source, place it under `data/`, and run the same command. If the mirror becomes unavailable, obtain the CSV from Kaggle and pass `--data PATH`. Refer to the original source for its current terms.

The package supplies a source link and reproducible acquisition instructions rather than bundling the full dataset. This satisfies the assignment's dataset-or-source-link deliverable.

Expected source-file SHA-256:

```text
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

The analysis records the actual SHA-256 in `results/summary.json`. A changed hash may mean the file bytes changed; investigate before comparing results. The script checks the schema, binary labels, missing/non-finite values, and nonnegative Time and Amount.

Observed raw shape: 284,807 rows and 31 columns (30 predictors plus Class). Raw fraud cases: 492. After removing 1,081 exact duplicate rows: 283,726 rows and 473 fraud cases. The rows retained use zero-based original CSV row identities, excluding the header.

Feature definitions and observed summary statistics are in the report and `results/data_dictionary.csv`.
