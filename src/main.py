from pipeline.features import engineer_features
from pipeline.model import build_model_frame
from pipeline.data import load_shots
from pipeline.train import train_and_evaluate
from functions import load_config

CONFIG = load_config()  # load config.yaml

# then pull values out like:
COMPETITION_ID = CONFIG["data"]["competition_id"]
SEASON_ID = CONFIG["data"]["season_id"]

if __name__ == "__main__":
    raw_shots = load_shots(COMPETITION_ID, SEASON_ID)
    shots = engineer_features(raw_shots)
    print(f"\nShots after removing penalties: {len(shots)}  |  goal rate: {shots['is_goal'].mean():.3f}")
 
    X, y, feature_names = build_model_frame(shots)
    results, splits = train_and_evaluate(X, y)
 
    # save best model + feature schema for reuse
    import joblib
    best_name = max(results, key=lambda k: results[k]["auc"])
    joblib.dump(
        {"model": results[best_name]["model"], "features": feature_names},
        "xg_model.joblib",
    )
    print(f"\nSaved best model ('{best_name}') to xg_model.joblib")