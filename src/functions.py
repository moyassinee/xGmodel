import yaml
from pathlib import Path
import pandas as pd
from statsbombpy import sb

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

"""Generic helpers to load StatsBomb open data."""

def get_competitions() -> pd.DataFrame:
    """All free competitions/seasons with their ids."""
    return sb.competitions()

def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """All matches of a competition-season."""
    cols = ["match_id", "match_date", "home_team", "away_team", "home_score", "away_score"]
    return sb.matches(competition_id=competition_id, season_id=season_id)[cols]


def find_matches(matches: pd.DataFrame, team: str | None = None, date: str | None = None) -> pd.DataFrame:
    """Filter matches by team name (partial, case-insensitive) and/or exact date 'YYYY-MM-DD'."""
    out = matches
    if team:
        mask = (
            out["home_team"].str.contains(team, case=False)
            | out["away_team"].str.contains(team, case=False)
        )
        out = out[mask]
    if date:
        out = out[out["match_date"] == date]
    return out


def load_events(match_id: int, use_cache: bool = True) -> pd.DataFrame:
    """All events of one match. Cached locally in data/raw so you only download once."""
    events = sb.events(match_id=match_id)
    return events

#statsBomb stores coordinates as a Python list in one cell, like [108.3, 41.2], 
# and we can’t do math or plotting directly on that
def add_xy(events: pd.DataFrame, col: str = "location", prefix: str = "") -> pd.DataFrame:
    """Split a [x, y] list column into two numeric columns.

    add_xy(events)                                  -> x, y
    add_xy(events, "pass_end_location", "end_")     -> end_x, end_y
    """
    out = events.copy()
    coords = out[col].apply(lambda v: v if isinstance(v, list) else [None, None])
    out[f"{prefix}x"] = coords.str[0]
    out[f"{prefix}y"] = coords.str[1]
    return out


def filter_events(events: pd.DataFrame, event_type: str, team: str | None = None) -> pd.DataFrame:
    """Events of one type (e.g. 'Shot', 'Pass', 'Pressure'), optionally for one team."""
    out = events[events["type"] == event_type]
    if team:
        out = out[out["team"] == team]
    return out.reset_index(drop=True)