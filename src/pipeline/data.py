import math
import warnings
 
import numpy as np
import pandas as pd
from statsbombpy import sb
import requests
 
warnings.filterwarnings("ignore")  # silences statsbombpy's "no auth, open data only" warning

def load_shots(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch all shot events across every match in a competition/season."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"Found {len(matches)} matches")
 
    all_shots = []
    for i, match_id in enumerate(matches["match_id"], 1):
        try:
            events = sb.events(match_id=match_id)
        except Exception as e:
            print(f"  skipping match {match_id}: {e}")
            continue
        shots = events[events["type"] == "Shot"].copy()
        shots["match_id"] = match_id
        all_shots.append(shots)
        if i % 10 == 0:
            print(f"  processed {i}/{len(matches)} matches")
 
    df = pd.concat(all_shots, ignore_index=True)
    print(f"Total shots pulled: {len(df)}")
    return df