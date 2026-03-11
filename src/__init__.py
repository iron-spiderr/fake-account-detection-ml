"""
Fake Account Detection System — Core ML Pipeline.

Modules:
    data_engineering    — Data loading, cleaning, imputation, balancing
    feature_extraction  — Metadata features, text embeddings, PCA
    graph_construction  — Social graph centrality features
    base_models         — 10 calibrated classifiers
    soft_voting         — F1-weighted soft voting ensemble
    stacking_shap       — OOF stacking, meta-learner, SHAP explainability
    pipeline            — End-to-end training & prediction orchestration
    output              — Human-readable output formatting
    pca_interpretability — SHAP → original feature name mapping
    instagram_api       — Instagram Graph API client
    scraper             — Instagram web scraper
    realtime_monitor    — Continuous monitoring daemon
"""

import sys as _sys

# ---------------------------------------------------------------------------
# Pickle compatibility shims
# ---------------------------------------------------------------------------
# The saved pipeline.pkl was serialised with the old flat module names.
# Register aliases so joblib.load() can resolve them without retraining.
# Each import is wrapped individually so that one failing submodule
# (e.g. scraper needing instaloader) doesn't prevent the others.
import importlib as _importlib

_ALIASES = {
    "module1_data_engineering": "src.data_engineering",
    "module2_feature_extraction": "src.feature_extraction",
    "module3_graph_construction": "src.graph_construction",
    "module4_base_models": "src.base_models",
    "module5_soft_voting": "src.soft_voting",
    "modules678": "src.stacking_shap",
    "modules910": "src.pipeline",
    "module11_output": "src.output",
    "pca_interpretability": "src.pca_interpretability",
    "scraper": "src.scraper",
}

for _old_name, _real_name in _ALIASES.items():
    if _old_name not in _sys.modules:
        try:
            _sys.modules[_old_name] = _importlib.import_module(_real_name)
        except ImportError:
            pass
