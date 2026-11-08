import numpy as np
import pandas as pd

FEATURES = ["distance_from_goal", "shot_angle", "half_volley", "header", "first_time",
            "under_pressure", "open_play", "key_pass", "n_def_in_cone", "nearest_def",
            "gk_dist_goal", "gk_offset", "gk_in_cone", "has_freeze_frame"]

REQUIRED_COLS = ["location", "shot_body_part", "shot_technique", "shot_type",
                 "shot_first_time", "under_pressure", "shot_key_pass_id",
                 "shot_freeze_frame"]

def in_triangle(p, a, b, c):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)

def freeze_features(row):
    ff = row["shot_freeze_frame"]
    shooter = row["location"]
    post1, post2 = (120, 36), (120, 44)
    out = dict(n_def_in_cone=np.nan, gk_dist_goal=np.nan,
               gk_offset=np.nan, gk_in_cone=np.nan, nearest_def=np.nan)
    if not isinstance(ff, (list, np.ndarray)) or len(ff) == 0:
        return pd.Series(out)

    defenders = [p["location"] for p in ff if not p["teammate"] and not p["position"]["name"] == "Goalkeeper"]
    gk = [p["location"] for p in ff if not p["teammate"] and p["position"]["name"] == "Goalkeeper"]

    out["n_def_in_cone"] = sum(in_triangle(d, shooter, post1, post2) for d in defenders)
    if defenders:
        out["nearest_def"] = min(np.hypot(d[0]-shooter[0], d[1]-shooter[1]) for d in defenders)
    if gk:
        g = gk[0]
        out["gk_dist_goal"] = np.hypot(120 - g[0], g[1] - 40)
        out["gk_offset"] = abs(g[1] - 40)
        out["gk_in_cone"] = int(in_triangle(g, shooter, post1, post2))
    return pd.Series(out)

def build_features(shots: pd.DataFrame) -> pd.DataFrame:
    df = shots.reindex(columns=REQUIRED_COLS).copy()   # missing columns become NaN, not KeyError
    x = df["location"].apply(lambda l: l[0])
    y = df["location"].apply(lambda l: l[1])

    dx, dy = 120 - x, (y - 40).abs()
    df["distance_from_goal"] = np.hypot(dx, dy)
    a = np.arctan2(8 * dx, dx**2 + dy**2 - 16)
    df["shot_angle"] = np.where(a < 0, a + np.pi, a)

    df["header"] = (df["shot_body_part"] == "Head").astype(int)
    df["half_volley"] = (df["shot_technique"] == "Half Volley").astype(int)
    df["open_play"] = (df["shot_type"] == "Open Play").astype(int)
    df["first_time"] = df["shot_first_time"].fillna(False).astype(int)
    df["under_pressure"] = df["under_pressure"].fillna(False).astype(int)
    df["key_pass"] = df["shot_key_pass_id"].notna().astype(int)

    ff = pd.DataFrame([freeze_features(f, s) for f, s in zip(df["shot_freeze_frame"], df["location"])],
                      index=df.index)
    df = df.join(ff)
    df["has_freeze_frame"] = df["n_def_in_cone"].notna().astype(int)
    return df[FEATURES]