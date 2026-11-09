# LCC Recorded Traffic Simulation Dashboard

## Changes
- Replaced `resources/views/analysis/index.blade.php` with the requested navy/teal dashboard.
- Navigation: Overview, Dataset & Validation, Simulation & Features, Model Evaluation, Ablation Study.
- CSV upload invokes the existing POST `/analyse` Laravel controller and the Python inference worker.
- Start/Pause/Reset/Speed replay the *actual returned model predictions* for up to 500 inspectable rows. No invented prediction values.
- Feature table shows observed and imputed model inputs, the continuous E5 IF score, E1–E5 per-row classifications, metrics when labels exist, and JSON export.
- Uploaded rows beyond the 500-row preview are still processed by the Python worker; replay is only the bounded preview.

## Setup
1. Run `composer install` in the Laravel project root, copy `.env.example` to `.env`, and run `php artisan key:generate`.
2. Configure `ML_PYTHON` in `.env` to a Python interpreter with numpy, pandas, scipy, scikit-learn, and joblib installed.
3. **Export the corrected E1–E5 inference bundle from the matching training notebook** and put it at `storage/app/ml/hybrid_if_svm_inference_bundle.joblib` (or configure `ML_BUNDLE` to its absolute path). The older `ml/hybrid_if_svm_inference_bundle.joblib` bundled with the original upload is NOT a verified compatible E1–E5 export. It fails to unpickle in the current test environment because `CustomIsolationForest` is not defined; do not use it as proof of working predictions.
4. Run `php artisan ml:doctor`, then `php artisan serve`.
5. Open the Dataset & Validation tab and upload a CSV containing `sttl,ct_state_ttl,dload,rate,dmean,dttl,ackdat,sload,sbytes` and optional `label` (0 or 1). Click Validate & Run E1–E5. Then use Simulation & Features.

## Testing limitations
- The Blade/JavaScript dashboard was syntax-checked; PHP controller syntax was checked.
- Full Laravel integration and end-to-end model inference could not be run here: the upload contains no Composer `vendor/` installation and does not contain a verified corrected E1–E5 inference bundle.
- This is recorded CSV replay, not live network packet capture. Results from uploaded files are not automatically verified as the official held-out UNSW-NB15 test partition.

## Optional live capture

- `capture_api/capture_service.py` is a separate localhost-only FastAPI service for authorized packet capture.
- It groups packets into short-lived flows, derives the same nine model columns, and runs the same E1–E5 bundle.
- Live flows have no ground-truth labels and use flow-time approximations, so CSV replay remains the baseline/testing path.
- Install Npcap and `capture_api/requirements.txt`, then run `uvicorn capture_api.capture_service:app --host 127.0.0.1 --port 8010`.
