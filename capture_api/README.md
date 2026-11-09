# Local live capture service

This optional service captures packets from an authorized local interface, groups them into short-lived flows, derives the nine model input columns, and sends the flows through the same `app/ml/predict.py` bundle used by CSV analysis.

Live traffic is intentionally separate from the CSV benchmark workflow. Captured flows do not have ground-truth labels, and the derived values are live flow approximations. Use CSV replay for baseline metrics and paper comparisons.

## Windows setup

1. Install Npcap from the official Npcap installer.
2. Create/use the project Python environment and install `capture_api/requirements.txt`.
3. Set `ML_BUNDLE` to the verified corrected E1–E5 bundle.
4. Start locally:

```powershell
uvicorn capture_api.capture_service:app --host 127.0.0.1 --port 8010
```

The dashboard expects `http://127.0.0.1:8010` by default. Override it with `window.LIVE_CAPTURE_API_URL` if needed.

Only capture traffic you are authorized to inspect. The service binds to localhost by default and requires an explicit Start action from the dashboard.
