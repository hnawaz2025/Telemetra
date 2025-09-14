# src/backend/api_server.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import json

app = Flask(__name__)
CORS(app)

ROOT = Path(__file__).resolve().parent.parent  # project root
OUT = ROOT / "outputs"
DATA = ROOT / "data"

def safe_read_csv(p: Path, **kw):
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, **kw)

@app.get("/api/meta")
def meta():
    best_info = OUT / "best_model_info.json"
    model_scores = OUT / "model_scores.json"
    meta = {}
    if best_info.exists():
        with open(best_info) as f:
            m = json.load(f)
        meta["model_name"] = m.get("name")
        meta["auc"] = (m.get("metrics") or {}).get("roc_auc")
    # add timestamps (best effort)
    meta["trained_at"] = (best_info.stat().st_mtime if best_info.exists() else None)
    meta["data_as_of"] = (OUT.stat().st_mtime if OUT.exists() else None)
    # optional recommended threshold
    meta["threshold"] = 0.5
    return jsonify(meta)

@app.get("/api/metrics")
def metrics():
    comp = safe_read_csv(OUT / "model_comparison.csv")
    out = {}
    if not comp.empty:
        # pick best by roc_auc if present
        if "roc_auc" in comp.columns:
            best = comp.sort_values("roc_auc", ascending=False).head(1).to_dict(orient="records")[0]
            out["auc"] = best.get("roc_auc")
            out["precision"] = best.get("precision")
            out["recall"] = best.get("recall")
            out["f1"] = best.get("f1_score")
    return jsonify(out)

@app.get("/api/driver_risk")
def driver_risk():
    # produced by your run script
    df = safe_read_csv(OUT / "driver_risk_scores_test.csv")
    # Harmonize expected columns
    # driver_index,risk_prediction,risk_probability,risk_score,risk_category
    if not df.empty:
        # rename and enrich
        cols = {c.lower(): c for c in df.columns}
        did_col = cols.get("driver_index") or cols.get("driver_id") or list(df.columns)[0]
        df = df.rename(columns={did_col: "driver_id"})
        # attach names/gender if available
        prof = safe_read_csv(DATA / "driver_profile.csv")
        if not prof.empty and "user_id" in prof.columns:
            df = df.merge(prof[["user_id","first_name","last_name","gender"]], left_on="driver_id", right_on="user_id", how="left")
            df["name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna("")).str.strip()
            df.drop(columns=[c for c in ["user_id","first_name","last_name"] if c in df.columns], inplace=True)
        # trips count per driver (optional)
        trips = safe_read_csv(DATA / "trips.csv")
        if not trips.empty and "user_id" in trips.columns:
            tc = trips.groupby("user_id")["trip_id"].nunique().reset_index().rename(columns={"user_id":"driver_id","trip_id":"trips"})
            df = df.merge(tc, on="driver_id", how="left")
        # final select/order
        keep = ["driver_id","name","gender","risk_probability","risk_score","risk_category","risk_prediction","trips"]
        for k in keep:
            if k not in df.columns:
                df[k] = None
        df = df[keep]
    return jsonify(df.to_dict(orient="records"))

@app.get("/api/driver/<driver_id>/trips")
def driver_trips(driver_id):
    limit = int(request.args.get("limit", 50))
    trips = safe_read_csv(DATA / "trips.csv")
    if trips.empty:
        return jsonify([])
    # normalize IDs as string compare
    trips["user_id_str"] = trips["user_id"].astype(str)
    sub = trips.loc[trips["user_id_str"] == str(driver_id)].copy()
    # pick basic fields if present
    want = [c for c in ["trip_id","ts","speed","accel","is_night","is_rush_hour"] if c in sub.columns]
    if not want:
        # fallback
        want = ["trip_id","ts"]
    sub = sub[want].tail(limit)
    # rename flags to match component
    if "is_night" in sub.columns: sub = sub.rename(columns={"is_night":"night"})
    if "is_rush_hour" in sub.columns: sub = sub.rename(columns={"is_rush_hour":"rush"})
    return jsonify(sub.to_dict(orient="records"))

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    app.run(host="0.0.0.0", port=8000, debug=True)
