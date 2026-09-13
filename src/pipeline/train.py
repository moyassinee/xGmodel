def train_and_evaluate(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    from xgboost import XGBClassifier
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
 
    results = {}
 
    # --- Baseline: logistic regression ---
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    p_logreg = logreg.predict_proba(X_test)[:, 1]
    results["logreg"] = {
        "model": logreg,
        "auc": roc_auc_score(y_test, p_logreg),
        "log_loss": log_loss(y_test, p_logreg),
        "brier": brier_score_loss(y_test, p_logreg),
    }
 
    # --- Stronger model: gradient boosted trees ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    p_xgb = xgb.predict_proba(X_test)[:, 1]
    results["xgb"] = {
        "model": xgb,
        "auc": roc_auc_score(y_test, p_xgb),
        "log_loss": log_loss(y_test, p_xgb),
        "brier": brier_score_loss(y_test, p_xgb),
    }
 
    print("\nModel performance (held-out test set):")
    for name, r in results.items():
        print(f"  {name:8s}  AUC={r['auc']:.3f}  LogLoss={r['log_loss']:.3f}  Brier={r['brier']:.3f}")
 
    return results, (X_train, X_test, y_train, y_test)