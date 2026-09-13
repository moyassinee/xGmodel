
import math
import warnings
import numpy as np
import pandas as pd
from statsbombpy import sb
from functions import load_config

CONFIG = load_config()  # load config.yaml

GOAL_X = CONFIG["pitch"]["goal_x"]  # x-coordinate of goal center
GOAL_Y = CONFIG["pitch"]["goal_y"]  # y-coordinate of goal center
GOAL_POST_Y1 = CONFIG["pitch"]["goal_post_y1"]  # y-coordinate of left goal post
GOAL_POST_Y2 = CONFIG["pitch"]["goal_post_y2"]  # y-coordinate of


def shot_distance_angle(x: float, y: float) -> tuple[float, float]:
    """Distance to goal center and the shooting angle (radians) subtended by the goal mouth."""
    distance = math.hypot(GOAL_X - x, GOAL_Y - y)
 
    # angle subtended by the goal posts from the shot location
    angle_top = math.atan2(GOAL_POST_Y2 - y, GOAL_X - x)
    angle_bottom = math.atan2(GOAL_POST_Y1 - y, GOAL_X - x)
    angle = abs(angle_top - angle_bottom)
    return distance, angle
 
 
def count_defenders_in_cone(freeze_frame, x, y):
    """
    Count opposition players (excluding GK) positioned between the shooter and
    the goal, inside the shot's angular cone. Also flags whether the GK is
    'set' (i.e. roughly between the shooter and goal center).
    Returns (defenders_in_cone, keeper_is_set) — both 0/NaN if no freeze frame.
    """
    if not isinstance(freeze_frame, list) or len(freeze_frame) == 0:
        return np.nan, np.nan
 
    angle_to_goal = math.atan2(GOAL_Y - y, GOAL_X - x)
    defenders_in_cone = 0
    keeper_is_set = 0
 
    for player in freeze_frame:
        loc = player.get("location")
        if not loc:
            continue
        px, py = loc
        # only defenders (teammate == False means opposition)
        is_teammate = player.get("teammate", False)
        is_keeper = player.get("position", {}).get("name") == "Goalkeeper"
 
        if is_teammate:
            continue
 
        # must be between shooter and the goal line, roughly in the shot direction
        if px <= x:
            continue
 
        angle_to_player = math.atan2(py - y, px - x)
        angle_diff = abs(angle_to_player - angle_to_goal)
 
        if is_keeper:
            keeper_is_set = 1 if angle_diff < 0.3 else 0
        elif angle_diff < 0.3:  # ~17 degrees either side of the shot line
            defenders_in_cone += 1
 
    return defenders_in_cone, keeper_is_set
 
 
def engineer_features(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
 
    # location comes back as [x, y]
    shots["x"] = shots["location"].apply(lambda loc: loc[0])
    shots["y"] = shots["location"].apply(lambda loc: loc[1])
 
    dist_angle = shots.apply(lambda r: shot_distance_angle(r["x"], r["y"]), axis=1)
    shots["distance"] = dist_angle.apply(lambda t: t[0])
    shots["angle"] = dist_angle.apply(lambda t: t[1])
 
    # freeze frame based features (present for 360 data; NaN otherwise, handled below)
    ff_col = "shot_freeze_frame" if "shot_freeze_frame" in shots.columns else None
    if ff_col:
        cone = shots.apply(
            lambda r: count_defenders_in_cone(r[ff_col], r["x"], r["y"]), axis=1
        )
        shots["defenders_in_cone"] = cone.apply(lambda t: t[0])
        shots["keeper_is_set"] = cone.apply(lambda t: t[1])
    else:
        shots["defenders_in_cone"] = np.nan
        shots["keeper_is_set"] = np.nan
 
    # categorical shot context
    shots["body_part"] = shots["shot_body_part"].fillna("Unknown")
    shots["technique"] = shots["shot_technique"].fillna("Unknown")
    shots["shot_type"] = shots["shot_type"].fillna("Unknown")  # open play / free kick / penalty
    shots["play_pattern"] = shots["play_pattern"].fillna("Unknown")
    shots["under_pressure"] = shots["under_pressure"].fillna(False).astype(int)
    shots["one_on_one"] = shots.get("shot_one_on_one", pd.Series(False, index=shots.index)).fillna(False).astype(int)
    shots["first_time"] = shots.get("shot_first_time", pd.Series(False, index=shots.index)).fillna(False).astype(int)
 
    # target: did the shot result in a goal?
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
 
    # penalties are near-deterministic (~76% conversion, no defensive structure) —
    # standard practice is to model them separately or exclude them
    shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)
 
    return shots