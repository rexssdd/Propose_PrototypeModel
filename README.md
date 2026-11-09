# LCC E1–E5 Defense Simulator

Local Laravel web prototype paired with ml/hybrid_improved.ipynb from the corrected paper.

Read the first section of SETUP.md. Run the notebook to train the corrected models, then install its E1_E5_models.joblib export at the configured ML_BUNDLE path. The previous three-model export is incompatible and deliberately not installed as the default model.

Features: full-file E1–E5 evaluation, training-only SMOTE pipelines, E1/E5 comparison, E2/E4 McNemar, bounded record replay, nine observed/cleaned features, E5 normalized IF score, separate E1 reference score, standardized hybrid inputs and JSON/preview CSV exports.

Synthetic cross-process bundle inference and UI JavaScript syntax checks passed. Full Laravel and browser checks remain to be run locally. Research results require the full held-out CSV; no fitted corrected E3/E5 model is shipped.
