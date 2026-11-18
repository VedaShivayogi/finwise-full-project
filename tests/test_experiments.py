"""Guards the Week-6 evidence: the MLflow runs, the exported table and the written report must agree with the champion model."""
import json
import sqlite3

import pandas as pd
import pytest

from conftest import ROOT

META = json.load(open(ROOT / "artifacts/champion_model_metadata.json"))


@pytest.fixture(scope="module")
def runs():
    return pd.read_csv(ROOT / "reports/mlflow_runs.csv")


def test_mlflow_database_holds_multiple_finished_runs():
    con = sqlite3.connect(ROOT / "mlflow.db")
    n = con.execute("select count(*) from runs where status = 'FINISHED'").fetchone()[0]
    names = {r[0] for r in con.execute("select name from runs where status = 'FINISHED'")}
    con.close()
    assert n >= 2, "need at least 2 hyperparameter runs logged in MLflow"
    assert {"config_A_shallow_fast", "config_B_balanced_default", "config_C_deep_regularized"} <= names


def test_exported_runs_have_different_hyperparameters(runs):
    assert len(runs) >= 2
    param_cols = [c for c in runs.columns if c.startswith("params.") and c not in ("params.n_features",)]
    differing = [c for c in param_cols if runs[c].nunique(dropna=False) > 1]
    assert {"params.max_depth", "params.learning_rate", "params.n_estimators"} <= set(differing)


def test_champion_is_the_best_validation_run_and_matches_metadata(runs):
    best = runs.sort_values("metrics.val_gini", ascending=False).iloc[0]
    assert best["run_id"] == META["mlflow_run_id"]
    assert best["metrics.test_gini"] == pytest.approx(META["metrics"]["test"]["gini"], abs=1e-4)
    assert best["metrics.test_gini"] >= META["target_gini"]


def test_report_is_consistent_with_mlflow(runs):
    text = (ROOT / "experiment_report.md").read_text()
    assert META["mlflow_run_id"] in text
    for name in runs["run_name"]:
        assert name in text
    assert "NOT YET MET" not in text and "3,500" not in text          # the stale pre-fix report said both
    assert f"{META['metrics']['test']['gini']:.4f}" in text


def test_evidence_images_are_present():
    for f in ("docs/mlflow_ui_runs.png", "docs/mlflow_ui_compare.png", "experiment_roc_comparison.png"):
        assert (ROOT / f).stat().st_size > 10_000, f
