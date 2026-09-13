import pandas as pd


def build_model_frame(shots: pd.DataFrame):
    numeric_features = ["distance", "angle", "under_pressure", "one_on_one", "first_time"]
 
    # only use freeze-frame derived features if we actually have coverage
    has_ff_coverage = shots["defenders_in_cone"].notna().mean() > 0.5
    if has_ff_coverage:
        shots["defenders_in_cone"] = shots["defenders_in_cone"].fillna(0)
        shots["keeper_is_set"] = shots["keeper_is_set"].fillna(0)
        numeric_features += ["defenders_in_cone", "keeper_is_set"]
        print("360 freeze-frame coverage detected — including defender features.")
    else:
        print("Low/no freeze-frame coverage — using location-based features only.")
 
    categorical_features = ["body_part", "technique", "shot_type", "play_pattern"]
 
    X = pd.get_dummies(shots[numeric_features + categorical_features], columns=categorical_features)
    y = shots["is_goal"]
    return X, y, list(X.columns)