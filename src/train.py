import joblib, sklearn, datetime as dt
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from xg_features import build_features, FEATURES

df = pd.read_parquet("../data/raw/all_shots.parquet")
shots = df[(df["type"] == "Shot") & (df["period"] < 5) & (df["shot_type"] != "Penalty")].copy()
y = (shots["shot_outcome"] == "Goal").astype(int)
X = build_features(shots)

model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(max_iter=1000))

oof = cross_val_predict(model, X, y, groups=shots["match_id"],
                        cv=GroupKFold(4), method="predict_proba")[:, 1]
metrics = {"auc": roc_auc_score(y, oof), "brier": brier_score_loss(y, oof),
           "logloss": log_loss(y, oof), "n_shots": len(y)}
print(metrics)

model.fit(X, y)                       # final model on all data
joblib.dump({"model": model, "features": FEATURES, "metrics": metrics,
             "sklearn": sklearn.__version__, "trained": dt.date.today().isoformat()},
            "../models/xg_v1.joblib", compress=3)