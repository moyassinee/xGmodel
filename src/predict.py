import joblib, numpy as np, pandas as pd
from xg_features import build_features

PENALTY_XG = 0.76
bundle = joblib.load("models/xg_v1.joblib")
model = bundle["model"]

def predict_xg(events: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=events.index, name="xg")
    shots = events[events["type"] == "Shot"]
    shots = shots[shots["period"] < 5]                       # shootouts -> NaN
    is_pen = shots["shot_type"] == "Penalty"
    out.loc[shots[is_pen].index] = PENALTY_XG
    open_ = shots[~is_pen]
    if len(open_):
        out.loc[open_.index] = model.predict_proba(build_features(open_))[:, 1]
    return out