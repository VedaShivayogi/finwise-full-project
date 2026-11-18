"""Supplementary evidence for the hyperparameter comparison: 5-fold stratified cross-validation of each MLflow
configuration on TRAIN+VALIDATION only (800 rows; the 200-row test set is never touched).
Each config uses its logged hyperparameters and a fixed number of trees (= its early-stopped best_iteration + 1).

    python scripts/cv_check.py [--db sqlite:///mlflow.db]     ->  reports/cv_check.csv
"""
import argparse, sys, tempfile
from pathlib import Path
import mlflow, numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data_pipeline as dp

KEYS = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
        "reg_lambda", "scale_pos_weight", "random_state"]
ap = argparse.ArgumentParser(); ap.add_argument("--db", default="sqlite:///mlflow.db"); a = ap.parse_args()
mlflow.set_tracking_uri(a.db)
runs = mlflow.search_runs(experiment_names=["finwise_champion_model"], filter_string="status = 'FINISHED'")
res = dp.run_pipeline(str(ROOT / "finwise_loan_applications.csv"), artifacts_dir=tempfile.mkdtemp())
X = pd.concat([res["X_train"], res["X_val"]]).reset_index(drop=True)
y = pd.concat([res["y_train"], res["y_val"]]).reset_index(drop=True).to_numpy()
rows = []
for _, r in runs.iterrows():
    p = {k: float(r[f"params.{k}"]) for k in KEYS if isinstance(r.get(f"params.{k}"), str)}
    for k in ("max_depth", "random_state"):
        if k in p: p[k] = int(p[k])
    n_trees = int(r["metrics.best_iteration"]) + 1
    g = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = xgb.XGBClassifier(n_estimators=n_trees, n_jobs=1, **p).fit(X.iloc[tr], y[tr])
        g.append(2 * roc_auc_score(y[te], m.predict_proba(X.iloc[te])[:, 1]) - 1)
    rows.append({"run_name": r["tags.mlflow.runName"], "run_id": r["run_id"], "trees": n_trees,
                 "cv_gini_mean": float(np.mean(g)), "cv_gini_std": float(np.std(g)), "folds": ";".join(f"{x:.3f}" for x in g)})
out = pd.DataFrame(rows).sort_values("cv_gini_mean", ascending=False)
(ROOT / "reports").mkdir(exist_ok=True)
out.to_csv(ROOT / "reports/cv_check.csv", index=False)
print(out.round(4).to_string(index=False))
