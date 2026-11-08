# app.py  (run: uvicorn app:app)
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from predict import predict_xg

app = FastAPI()

class Shot(BaseModel):
    type: str = "Shot"
    period: int
    location: list[float]
    shot_type: str | None = None
    shot_body_part: str | None = None
    shot_technique: str | None = None
    shot_first_time: bool | None = None
    under_pressure: bool | None = None
    shot_key_pass_id: str | None = None
    shot_freeze_frame: list[dict] | None = None

@app.post("/xg")
def xg(shot: Shot):
    return {"xg": float(predict_xg(pd.DataFrame([shot.model_dump()])).iloc[0])}