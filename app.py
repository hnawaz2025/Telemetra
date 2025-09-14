# app.py

from pathlib import Path
from typing import Dict, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- App config ----------
st.set_page_config(
    page_title="Telematics Driver Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = ROOT / "data"
OUT_DIR_DEFAULT = ROOT / "outputs"

# ---------- Utilities ----------
def safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return pd.DataFrame()

def lower_map(df: pd.DataFrame) -> Dict[str, str]:
    return {c.lower(): c for c in df.columns}

def col(df: pd.DataFrame, *names: str) -> Optional[str]:
    if df.empty:
        return None
    low = lower_map(df)
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None

def find_scores_file(root: Path) -> Optional[Path]:
    # Prefer outputs first, then data; support both filenames
    candidates = [
        root / "outputs/behavioral_risk_scores.csv",
        root / "data/behavioral_risk_scores.csv",
        root / "data/driver_risk_scores_test.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def ensure_risk_columns(rs: pd.DataFrame) -> pd.DataFrame:
    """Normalize risk file column names and derive missing ones."""
    if rs.empty:
        return rs
    rs = rs.copy()
    # id
    idc = col(rs, "driver_id", "driver_index", "user_id") or rs.columns[0]
    rs = rs.rename(columns={idc: "driver_id"})
    rs["driver_id"] = rs["driver_id"].astype(str).str.strip()
    # standard names
    ren = {}
    for c in rs.columns:
        lc = c.lower()
        if lc == "risk_probability": ren[c] = "risk_probability"
        if lc == "risk_score":       ren[c] = "risk_score"
        if lc == "risk_category":    ren[c] = "risk_category"
        if lc == "risk_prediction":  ren[c] = "risk_prediction"
    if ren:
        rs = rs.rename(columns=ren)
    # derive prob/score/category
    if "risk_probability" not in rs.columns and "risk_score" in rs.columns:
        rs["risk_probability"] = pd.to_numeric(rs["risk_score"], errors="coerce").fillna(0)/100.0
    if "risk_score" not in rs.columns and "risk_probability" in rs.columns:
        rs["risk_score"] = (pd.to_numeric(rs["risk_probability"], errors="coerce").fillna(0).clip(0,1)*100).round(2)
    if "risk_category" not in rs.columns and "risk_probability" in rs.columns:
        p = pd.to_numeric(rs["risk_probability"], errors="coerce").fillna(0.0)
        rs["risk_category"] = pd.cut(p, [-1,0.2,0.5,0.8,1.1], labels=["Low","Medium","High","Very High"]).astype(str)
    return rs

def derive_event_flags(trips: pd.DataFrame) -> pd.DataFrame:
    """Compute flags from trip rows only (no history)."""
    if trips.empty: return trips
    trips = trips.copy()

    spd = col(trips, "speed")
    lim = col(trips, "limit")
    acc = col(trips, "accel")
    brk = col(trips, "is_brake")
    hr  = col(trips, "hour")
    rain= col(trips, "rain")

    # speed flags
    trips["_over"] = 0
    trips["_excess"] = 0
    if spd and lim:
        trips["_over"]   = (pd.to_numeric(trips[spd], errors="coerce") > pd.to_numeric(trips[lim], errors="coerce")).astype(int)
        trips["_excess"] = (pd.to_numeric(trips[spd], errors="coerce") > pd.to_numeric(trips[lim], errors="coerce")*1.2).astype(int)

    # accel/brake flags
    trips["_hard_accel"] = 0
    trips["_hard_brake"] = 0
    if acc:
        a = pd.to_numeric(trips[acc], errors="coerce")
        trips["_hard_accel"] = (a > 2.5).astype(int)
        trips["_hard_brake"] = (a < -3.0).astype(int)

    trips["_brake_flag"] = 0
    if brk:
        trips["_brake_flag"] = pd.to_numeric(trips[brk], errors="coerce").fillna(0).astype(int)

    # periods
    trips["_is_night"] = 0
    trips["_is_rush"]  = 0
    if hr:
        h = pd.to_numeric(trips[hr], errors="coerce").fillna(-1)
        trips["_is_night"] = (((h >= 22) | (h <= 5)) & (h >= 0)).astype(int)
        trips["_is_rush"] = (((h >= 7) & (h <= 9)) | ((h >= 17) & (h <= 19))).astype(int)

    # weather
    trips["_adverse"] = 0
    if rain:
        r = pd.to_numeric(trips[rain], errors="coerce").fillna(0)
        trips["_adverse"] = (r > 0).astype(int)

    # distance fallback (if no distance column)
    dcol = col(trips, "distance_km")
    if dcol is None:
        trips["_distance_km"] = 0.3  # crude fallback
    else:
        trips["_distance_km"] = pd.to_numeric(trips[dcol], errors="coerce").fillna(0.0)

    return trips

def aggregate_driver(trips: pd.DataFrame, trip_labels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one-row-per-driver metrics from trips only."""
    if trips.empty: return pd.DataFrame()
    trips = trips.copy()
    uid = col(trips, "user_id")
    tid = col(trips, "trip_id")
    spd = col(trips, "speed")

    # join claims if present
    if not trip_labels.empty and tid and col(trip_labels, "trip_id") and col(trip_labels, "claim"):
        t_tid = col(trip_labels, "trip_id")
        t_clm = col(trip_labels, "claim")
        ttmp = trip_labels[[t_tid, t_clm]].copy()
        ttmp[t_clm] = pd.to_numeric(ttmp[t_clm], errors="coerce").fillna(0).astype(int)
        trips = trips.merge(ttmp, left_on=tid, right_on=t_tid, how="left").rename(columns={t_clm: "_claim"})
    else:
        trips["_claim"] = 0

    # aggregate
    agg = trips.groupby(uid).agg(
        rows=("user_id","size"),
        n_trips=((tid if tid else uid), "nunique"),
        mean_speed=(spd, "mean") if spd else ("_over", "mean"),
        max_speed=(spd, "max") if spd else ("_over", "max"),
        speed_var=(spd, "var") if spd else ("_over", "var"),
        pct_over_limit=("_over","mean"),
        pct_excess_speed=("_excess","mean"),
        hard_accel_rate=("_hard_accel","mean"),
        hard_brake_rate=("_hard_brake","mean"),
        brake_frequency=("_brake_flag","mean"),
        night_frac=("_is_night","mean"),
        rush_frac=("_is_rush","mean"),
        adverse_frac=("_adverse","mean"),
        km_total=("_distance_km","sum"),
    ).reset_index().rename(columns={uid:"driver_id"})

    # events per 100km
    ev_num = (
        agg["pct_excess_speed"].fillna(0) +
        agg["hard_accel_rate"].fillna(0) +
        agg["hard_brake_rate"].fillna(0)
    ) * agg["rows"].fillna(0)
    km = agg["km_total"].replace(0, np.nan)
    agg["events_per_100km"] = 0.0
    mask = km > 0
    agg.loc[mask, "events_per_100km"] = 100.0 * ev_num.loc[mask] / km.loc[mask]
    agg["driver_claim_rate"] = 0.0

    # driver-level claims if any
    if "_claim" in trips.columns and tid:
        dcl = trips.groupby(uid)["_claim"].sum().reset_index()
        dcl = dcl.rename(columns={uid:"driver_id", "_claim":"claims_sum"})
        agg = agg.merge(dcl, on="driver_id", how="left")
        agg["claims_sum"] = agg["claims_sum"].fillna(0).astype(int)
        agg["driver_claim_rate"] = np.where(agg["n_trips"]>0, agg["claims_sum"]/agg["n_trips"], 0.0)

    # normalize id dtype
    agg["driver_id"] = agg["driver_id"].astype(str).str.strip()
    return agg

def synthetic_risk(agg: pd.DataFrame) -> pd.Series:
    """Fallback ‘probability’ using only behavior (no model file)."""
    if agg.empty: return pd.Series(dtype=float)
    feats = ["pct_excess_speed","pct_over_limit","hard_accel_rate","hard_brake_rate",
             "brake_frequency","night_frac","adverse_frac","events_per_100km"]
    X = agg[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    score = (
        X["pct_excess_speed"]*1.5 +
        X["pct_over_limit"]*0.5 +
        X["hard_accel_rate"]*1.0 +
        X["hard_brake_rate"]*1.0 +
        X["brake_frequency"]*0.3 +
        X["night_frac"]*0.2 +
        X["adverse_frac"]*0.2 +
        (X["events_per_100km"]/50.0)
    )
    rank = (score.rank(method="average") - 1) / max(1, len(score)-1)
    return rank.clip(0,1)

def compute_premium_row(
    row: pd.Series,
    base_premium: float,
    risk_weight: float,
    night_weight: float,
    adverse_weight: float,
    excess_weight: float,
    cap_pct: float,
) -> float:
    """Premium = base * (1 + weighted surcharges), capped."""
    rp = float(row.get("risk_probability", np.nan))
    if np.isnan(rp):
        rp = float(row.get("synthetic_probability", 0.0))
    night = float(row.get("night_frac", 0.0))
    adv   = float(row.get("adverse_frac", 0.0))
    exc   = float(row.get("pct_excess_speed", 0.0))
    uplift = (
        risk_weight   * rp +
        night_weight  * night +
        adverse_weight* adv +
        excess_weight * exc
    )
    uplift = max(0.0, uplift)
    uplift = min(uplift, cap_pct/100.0)
    return round(base_premium * (1.0 + uplift), 2)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
data_dir_input = st.sidebar.text_input("Data directory", str(DATA_DIR_DEFAULT))
outputs_dir_input = st.sidebar.text_input("Outputs directory (optional)", str(OUT_DIR_DEFAULT))

DATA_DIR = Path(data_dir_input).expanduser().resolve()
OUT_DIR = Path(outputs_dir_input).expanduser().resolve()
OUT_DIR.mkdir(exist_ok=True, parents=True)

st.sidebar.markdown("#### Pricing weights")
base_premium = st.sidebar.number_input("Base premium ($/period)", 10.0, 5000.0, 600.0, 10.0)
risk_weight  = st.sidebar.slider("Weight: model risk",   0.0, 2.0, 0.8, 0.05)
excess_w     = st.sidebar.slider("Weight: excess speed", 0.0, 2.0, 0.4, 0.05)
night_w      = st.sidebar.slider("Weight: night driving",0.0, 2.0, 0.2, 0.05)
adv_w        = st.sidebar.slider("Weight: adverse wx",   0.0, 2.0, 0.2, 0.05)
cap_pct      = st.sidebar.slider("Cap total surcharge (%)", 0, 200, 50, 5)

st.sidebar.markdown("---")
st.sidebar.caption("If no model scores file is provided, the app uses a synthetic risk built only from your trips.")

# ---------- Load data ----------
trips = safe_read_csv(DATA_DIR / "trips.csv", dtype={"user_id": str, "trip_id": str})
if trips.empty:
    st.error("`data/trips.csv` not found or empty.")
    st.stop()

# Optional tables
driver_profile      = safe_read_csv(DATA_DIR / "driver_profile.csv", dtype={"user_id": str})
vehicle_assignments = safe_read_csv(DATA_DIR / "vehicle_assignments.csv", dtype={"user_id": str, "vehicle_id": str})
vehicle_info        = safe_read_csv(DATA_DIR / "vehicle_info.csv", dtype={"vehicle_id": str})
trip_labels         = safe_read_csv(DATA_DIR / "trip_labels.csv")  # flexible schema; we’ll coerce if present

# Derive flags + aggregate
trips_flagged = derive_event_flags(trips)
agg = aggregate_driver(trips_flagged, trip_labels)

# Attach profile name/gender if present
if not driver_profile.empty and col(driver_profile, "user_id"):
    first = col(driver_profile, "first_name")
    last  = col(driver_profile, "last_name")
    prof = driver_profile.copy()
    prof["user_id"] = prof[col(driver_profile, "user_id")].astype(str).str.strip()
    if first or last:
        prof["name"] = (
            prof.get(first,"").fillna("").astype(str) + " " +
            prof.get(last,"").fillna("").astype(str)
        ).str.strip()
    else:
        prof["name"] = ""
    keep_cols = ["user_id","name"] + (["gender"] if "gender" in prof.columns else [])
    agg = agg.merge(prof[keep_cols].rename(columns={"user_id":"driver_id"}), on="driver_id", how="left")

# Attach vehicle info via assignments if available
if (not vehicle_assignments.empty and
    col(vehicle_assignments,"user_id") and col(vehicle_assignments,"vehicle_id")):
    vasn = vehicle_assignments.copy()
    vasn["user_id"] = vasn[col(vehicle_assignments,"user_id")].astype(str).str.strip()
    vasn["vehicle_id"] = vasn[col(vehicle_assignments,"vehicle_id")].astype(str).str.strip()
    agg = agg.merge(vasn.rename(columns={"user_id":"driver_id"}), on="driver_id", how="left")
    if not vehicle_info.empty and col(vehicle_info,"vehicle_id"):
        vinfo = vehicle_info.copy()
        vinfo["vehicle_id"] = vinfo[col(vehicle_info,"vehicle_id")].astype(str).str.strip()
        agg = agg.merge(vinfo, on="vehicle_id", how="left")
        # Optional safety score
        sc = [c for c in ["airbags","sensors","safety_features_count"] if c in agg.columns]
        if sc:
            agg["safety_score"] = agg[sc].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

# Attach model risk if exists; else synthesize
rs_file = find_scores_file(ROOT)
risk_df = ensure_risk_columns(safe_read_csv(rs_file)) if rs_file else pd.DataFrame()
if not risk_df.empty:
    # Enforce string ids on both sides to avoid dtype merge errors
    agg["driver_id"] = agg["driver_id"].astype(str).str.strip()
    risk_df["driver_id"] = risk_df["driver_id"].astype(str).str.strip()
    risk_pick = risk_df[["driver_id","risk_probability","risk_score","risk_category"]].drop_duplicates("driver_id")
    agg = agg.merge(risk_pick, on="driver_id", how="left")
else:
    agg["synthetic_probability"] = synthetic_risk(agg)
    agg["risk_probability"] = np.nan
    agg["risk_score"] = (agg["synthetic_probability"]*100).round(2)
    agg["risk_category"] = pd.cut(
        agg["synthetic_probability"].fillna(0.0),
        [-1, 0.2, 0.5, 0.8, 1.1],
        labels=["Low","Medium","High","Very High"]
    ).astype(str)

# Premium
agg["predicted_premium"] = agg.apply(
    lambda r: compute_premium_row(
        r, base_premium, risk_weight, night_w, adv_w, excess_w, cap_pct
    ), axis=1
)

# ---------- Header ----------
st.title("Telematics Driver Dashboard")
st.caption("Driver-specific risk, behavior insights, and premium estimates — using only your local CSVs.")

# ---------- Driver picker ----------
pick_left, pick_right = st.columns([0.65, 0.35])
with pick_left:
    options = agg[["driver_id"]].copy()
    if "name" in agg.columns:
        options["name"] = agg["name"].fillna("").astype(str)
    else:
        options["name"] = ""
    options["label"] = options.apply(
        lambda r: f"{r['driver_id']} — {r['name']}" if r["name"].strip() else str(r["driver_id"]),
        axis=1
    )
    default_idx = 0
    driver_label = st.selectbox("Select a driver", options["label"].tolist(), index=default_idx)
    pick_id = options.loc[options["label"] == driver_label, "driver_id"].iloc[0]
with pick_right:
    st.metric("Drivers loaded", f"{len(agg):,}")
    if "risk_probability" in agg.columns and not agg["risk_probability"].isna().all():
        st.metric("Model risk available", "Yes")
    else:
        st.metric("Model risk available", "No (using synthetic)")

# Current driver row
drow = agg[agg["driver_id"] == pick_id].head(1).copy()
if drow.empty:
    st.warning("No aggregate row for this driver.")
    st.stop()

# ---------- KPIs for selected driver ----------
k1,k2,k3,k4,k5 = st.columns(5)
prob = float(drow["risk_probability"].fillna(drow.get("synthetic_probability",0.0)).values[0])
score= float(drow["risk_score"].values[0]) if "risk_score" in drow.columns else round(prob*100,2)
cat  = str(drow["risk_category"].values[0])
k1.metric("Risk probability", f"{prob*100:.1f}%")
k2.metric("Risk category", cat)
k3.metric("Excess speed %", f"{float(drow['pct_excess_speed']*100):.1f}%")
k4.metric("Hard accel/brake", f"{float(drow['hard_accel_rate']+drow['hard_brake_rate']):.2f}")
k5.metric("Predicted premium", f"${float(drow['predicted_premium']):,.2f}")

st.markdown("---")

# ---------- Visualizations (driver-specific from trips only) ----------
uid_col = col(trips_flagged, "user_id")
t_driver = trips_flagged[trips_flagged[uid_col].astype(str).str.strip() == pick_id].copy()

v1, v2 = st.columns([0.55, 0.45])

with v1:
    st.subheader("Speed distribution")
    spd = col(t_driver, "speed")
    if spd:
        chart = alt.Chart(t_driver).mark_bar().encode(
            x=alt.X(f"{spd}:Q", bin=alt.Bin(maxbins=35), title="Speed"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[f"mean({spd}):Q"]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No speed column in trips.")

    st.subheader("Speed vs. Limit (recent)")
    lim = col(t_driver, "limit")
    if spd and lim:
        recent = t_driver.tail(1000).reset_index(drop=True).reset_index().rename(columns={"index":"seq"})
        base = alt.Chart(recent).encode(x=alt.X("seq:Q", title="Sequence"))
        layer = alt.layer(
            base.mark_line().encode(y=alt.Y(f"{spd}:Q", title="Speed / Limit")),
            base.mark_line().encode(y=alt.Y(f"{lim}:Q"))
        ).resolve_scale(y='shared').properties(height=240)
        st.altair_chart(layer, use_container_width=True)
    else:
        st.info("Need both speed and limit to draw this.")

with v2:
    st.subheader("Time of day")
    hr = col(t_driver, "hour")
    if hr:
        td = t_driver.copy()
        td["hour_i"] = pd.to_numeric(td[hr], errors="coerce").fillna(-1).astype(int)
        td = td[(td["hour_i"]>=0) & (td["hour_i"]<=23)]
        chart = alt.Chart(td).mark_bar().encode(
            x=alt.X("hour_i:O", title="Hour"),
            y=alt.Y("count():Q", title="Events"),
            tooltip=["hour_i","count()"]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No hour column in trips.")

    st.subheader("Event rates")
    ev_tbl = pd.DataFrame({
        "Metric": ["Over limit", "Excess speed", "Hard accel", "Hard brake", "Night", "Adverse"],
        "Rate": [
            float(drow["pct_over_limit"])*100,
            float(drow["pct_excess_speed"])*100,
            float(drow["hard_accel_rate"])*100,
            float(drow["hard_brake_rate"])*100,
            float(drow["night_frac"])*100,
            float(drow["adverse_frac"])*100,
        ]
    })
    bar = alt.Chart(ev_tbl).mark_bar().encode(
        x=alt.X("Rate:Q", title="%"),
        y=alt.Y("Metric:N", sort="-x")
    ).properties(height=220)
    st.altair_chart(bar, use_container_width=True)

st.markdown("---")

# ---------- Map (optional if lat/lon exist) ----------
latc = col(t_driver, "lat")
lonc = col(t_driver, "lon")
if latc and lonc:
    st.subheader("Trip points (sample)")
    m = t_driver[[latc, lonc]].dropna().copy()
    m[latc] = pd.to_numeric(m[latc], errors="coerce")
    m[lonc] = pd.to_numeric(m[lonc], errors="coerce")
    m = m.dropna()
    if len(m) > 0:
        m = m.sample(min(2000, len(m)), random_state=42)
        m = m.rename(columns={latc:"lat", lonc:"lon"})
        st.map(m, zoom=8)
    else:
        st.caption("Lat/Lon present but not numeric; skipping map.")
else:
    st.caption("No lat/lon columns found; skipping map.")

# ---------- Premium sandbox for selected driver ----------
st.markdown("### Premium sandbox")
c1,c2,c3 = st.columns(3)
delta_excess = c1.slider("Assume excess speed reduction (%)", 0, 50, 10, 5)
delta_night  = c2.slider("Assume night driving reduction (%)", 0, 50, 0, 5)
delta_adv    = c3.slider("Assume adverse weather exposure reduction (%)", 0, 50, 0, 5)

sim = drow.copy()
sim["pct_excess_speed"] = max(0.0, float(drow["pct_excess_speed"]) * (1 - delta_excess/100.0))
sim["night_frac"]       = max(0.0, float(drow["night_frac"]) * (1 - delta_night/100.0))
sim["adverse_frac"]     = max(0.0, float(drow["adverse_frac"]) * (1 - delta_adv/100.0))
sim_premium = compute_premium_row(sim.iloc[0], base_premium, risk_weight, night_w, adv_w, excess_w, cap_pct)

pc1, pc2 = st.columns(2)
pc1.metric("Current predicted premium", f"${float(drow['predicted_premium']):,.2f}")
pc2.metric("Simulated premium", f"${sim_premium:,.2f}")

st.markdown("---")

# ---------- Leaderboards ----------
st.subheader("Leaderboards")
lb1, lb2 = st.columns(2)

risk_col = "risk_probability" if ("risk_probability" in agg.columns and not agg["risk_probability"].isna().all()) else "synthetic_probability"

with lb1:
    st.caption("Best by (lowest) risk")
    lb = agg[["driver_id","name" if "name" in agg.columns else "driver_id", risk_col, "predicted_premium","n_trips","km_total"]].copy()
    if "name" not in lb.columns:
        lb["name"] = ""
    lb = lb.sort_values(risk_col, ascending=True).head(10)
    lb[risk_col] = (lb[risk_col]*100).round(1)
    lb = lb.rename(columns={risk_col:"risk_%"}).reset_index(drop=True)
    st.dataframe(lb, use_container_width=True, height=300)

with lb2:
    st.caption("Best by (lowest) behavior risk")
    beh = agg[["driver_id","name" if "name" in agg.columns else "driver_id",
               "pct_excess_speed","hard_accel_rate","hard_brake_rate","events_per_100km","predicted_premium"]].copy()
    if "name" not in beh.columns:
        beh["name"] = ""
    beh["behavior_score"] = (
        beh["pct_excess_speed"].fillna(0)*1.5 +
        beh["hard_accel_rate"].fillna(0)*1.0 +
        beh["hard_brake_rate"].fillna(0)*1.0 +
        (beh["events_per_100km"].fillna(0)/50.0)
    )
    beh = beh.sort_values("behavior_score", ascending=True).head(10)
    beh["excess_%"] = (beh["pct_excess_speed"]*100).round(1)
    beh = beh.drop(columns=["pct_excess_speed"]).reset_index(drop=True)
    st.dataframe(beh, use_container_width=True, height=300)

st.markdown("---")

# ---------- Raw trip table for selected driver (recent rows) ----------
st.subheader("Recent trip rows (driver)")
show_cols = []
for want in ["trip_id","ts","timestamp","speed","limit","accel","hour","rain","is_brake",
             "lat","lon","_over","_excess","_hard_accel","_hard_brake","_is_night","_adverse"]:
    c = col(t_driver, want) if not want.startswith("_") else (want if want in t_driver.columns else None)
    if c and c not in show_cols:
        show_cols.append(c)
if not show_cols:
    st.caption("No standard trip columns found to display.")
else:
    st.dataframe(t_driver[show_cols].tail(300).reset_index(drop=True), use_container_width=True, height=320)

# ---------- Footer ----------
st.caption(
    "Notes: If no model scores file is provided, risk uses a synthetic rank-based proxy derived from behavior. "
    "Premium is an illustrative formula using only your metrics and the weights you set above."
)
