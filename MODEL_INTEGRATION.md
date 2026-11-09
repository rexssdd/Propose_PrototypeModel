# Current Hybrid Model Integration

Place the Kaggle export at:

`storage/app/ml/hybrid_if_svm_inference_bundle.joblib`

The worker now expects the current thesis architecture:

`9 Shukla features -> custom IF score -> [9 features + IF score] -> StandardScaler -> RBF-SVM`.

It also evaluates the standalone Shukla-style IF and standalone RBF-SVM when the uploaded CSV contains a `label` column.

## Python dependencies

Install into the Python environment used by Laravel:

`pip install numpy pandas joblib scikit-learn`

The bundle contains fitted sklearn scalers/SVC objects, so scikit-learn is required. For best compatibility, use the same scikit-learn version reported by the Kaggle export metadata.

## Verify before using the dashboard

Run from the project directory:

`python app/ml/predict.py --bundle storage/app/ml/hybrid_if_svm_inference_bundle.joblib --input path/to/UNSW_NB15_testing-set.csv --name UNSW_NB15_testing-set.csv`

The official test file should contain 82,332 rows. Compare its confusion matrix/metrics with the Kaggle run before treating deployment as verified.
