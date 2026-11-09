# Corrected E1–E5 integration — read first

The included ml/hybrid_improved.ipynb matches the corrected paper. Run it on the official training/test files, then copy lcc_corrected_outputs/E1_E5_models.joblib to storage/app/ml/hybrid_if_svm_inference_bundle.joblib. The existing config path is retained for compatibility, but its contents must now be the NEW E1–E5 bundle.

No trained E3/E5 bundle is supplied: the official dataset was not uploaded and the corrected experiment has not been run here. The original three-model bundle remains in ml/ for reference and is explicitly rejected by the new worker. Do not use it as E5.

Install Python dependencies in the chosen virtual environment: numpy pandas scikit-learn imbalanced-learn scipy matplotlib joblib. Match the versions recorded in the newly trained bundle. Use the Laravel installation steps below. The corrected worker defines the notebook custom classes so joblib can resolve its exported pipelines. Only load trusted bundles.

The dashboard maps baseline=E1, secondary SVM=E2, hybrid=E5. The Ablation Study screen shows all five configurations and paired E2/E4 McNemar. Record detail shows all five predictions plus E5's actual normalized IF score and separate E1 reference score. Notebook and application run SMOTE only during training, never during prediction.

Validation: synthetic E1–E5 model export and CLI integration verified; full research results require rerunning on the official held-out file. PHP/Composer and Chromium were unavailable, so full Laravel/browser checks remain local requirements. The inherited legacy test_prediction_logic.py references missing old training code; test_defense_worker.py describes the earlier worker. Use the corrected E1–E5 CLI integration check when validating this revision.

---

# LCC defense simulator setup

This is a local Laravel web application, not a browser extension. The updated Blade interface has no CDN or Node build dependency.

## Requirements

Use PHP 8.3 or newer compatible with composer.lock, Composer 2, and Python 3.12. Match Python dependency versions to the metadata of the newly trained E1–E5 export. The old bundle version does not establish compatibility with the corrected training run.

From the extracted project directory on Windows PowerShell:

```powershell
composer install
Copy-Item .env.example .env
php artisan key:generate
python -m venv ml-venv
.\ml-venv\Scripts\python.exe -m pip install numpy==2.0.2 pandas==2.3.3 scikit-learn==1.6.1 joblib
```

Set ML_PYTHON in .env to the full path of ml-venv/Scripts/python.exe. Prefer a project directory without spaces because the existing interpreter resolver splits whitespace. SESSION_DRIVER=file and CACHE_STORE=file avoid database setup for this dashboard.

```text
ML_SCRIPT=ml/predict.py
ML_BUNDLE=storage/app/ml/hybrid_if_svm_inference_bundle.joblib
```

For this revision, train the corrected notebook and install the NEW bundle as instructed above. Only load trusted joblib bundles; model deserialization executes Python objects.

```powershell
php artisan config:clear
php artisan ml:doctor
php artisan serve
```

Open the address printed by Artisan. Select Dataset & Validation, upload a CSV, and run inference. The worker evaluates every accepted row (up to 300,000) and returns the first 500 for replay and inspection. Replay is visualization of completed inference, not live capture. Last 100 visible rows are shown to keep the browser responsive. Export JSON includes complete-run metrics and bounded record detail; preview CSV is not a complete prediction export.

## Verification performed for this revision

Five Python behavior tests pass: vectorized forest parity, tied-score metrics, single-class metrics, invalid labels, and malformed table schemas. A two-record synthetic-input CLI smoke test runs the supplied bundle. These synthetic labels are not research validation. UI script syntax checked with Node. Browser interaction checks could not run because the environment has no installed Chromium executable. Browser layout and interaction require local verification.

PHP and Composer are unavailable in the editing environment, so Laravel/PHPUnit and PHP formatting checks were not executable. Run the project feature tests locally. The pre-existing tests/test_prediction_logic.py references a missing training file and the legacy worker architecture; it remains as supplied and is not part of the passing new worker suite. Run the new tests with:

```powershell
python -m unittest discover -s tests -p test_defense_worker.py -v
```

Before defense, run the official held-out CSV and compare confusion counts against the notebook. No official test CSV was supplied here. Hashes identify inputs and models, but cannot by themselves establish held-out provenance.

## Rubric coverage and limitations

Correctness: paper-aligned 9+1 pipeline; saved thresholds; validated binary labels; actual confusion matrices, F0.5, MCC, ROC-AUC, trapezoidal PR-AUC and separate average precision. The notebook calls average precision PR-AUC; compare like with like.

Robustness: duplicate/missing headers, empty input, invalid labels and row limits rejected; invalid feature cells use training medians with explicit counts. Per-file results retain failures.

Efficiency: grouped forest traversal verified against per-record paths; SVM inference batched; no training on uploads. Bundle reloads per file via existing PHP process design. No claim of a persistent model server.

Usability and communication: responsive offline styling, keyboard controls, replay/pause/reset, raw and cleaned feature values, standardized input inspection, scope labels, exports and source provenance.

Ablation evidence: uploaded hybrid.ipynb records only IF, standalone SVM and full hybrid. Two additional configurations are explicitly unverified. The hybrid has higher recall but lower accuracy/F1 than standalone SVM in that saved run.

Feasibility: local Laravel + Python with included model. Training times are not invented; recorded notebook training/inference measurements are separate historical evidence. Additional feature contribution explanations are not implemented.
