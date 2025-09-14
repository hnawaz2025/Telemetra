# src/simulations.py
"""
Telemetra — Telematics Data Simulation & Export
Generates realistic telematics points, driver profiles (with age/gender/height),
vehicles, assignments, and synthetic trip labels for end-to-end POC.

Outputs (default ./data):
  - data/trips.csv                 # point-level telematics (for pipeline.py)
  - data/driver_profile.csv        # driver metadata (incl. age, gender, height_cm)
  - data/vehicle_info.csv
  - data/vehicle_assignments.csv
  - data/trip_labels.csv

Optional:
  - SQLite snapshot telematics_data.db with trips + driver_profiles
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import sqlite3
import random


# -----------------------------
# Config / Constants
# -----------------------------
MPH_PER_SEC_FROM_30S = 1.0 / 0.5  # since accel = d(mph) / sec with 30s step => divide diff by 0.5
BRAKE_THRESHOLD_MPH_PER_S = -8.0   # "hard brake" threshold (mph per second)
RAPID_ACCEL_THRESHOLD_MPH_PER_S = 6.0
COMMUTE_HOURS = {7, 8, 17, 18}
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class DriverProfile:
    """Driver profile for generating realistic telematics data"""
    driver_id: str
    age: int
    experience_years: int
    risk_level: str              # 'low', 'medium', 'high'
    driving_style: str           # 'conservative', 'normal', 'aggressive'
    typical_trips_per_week: int
    preferred_times: List[int]   # hours (0..23)
    urban_ratio: float           # [0,1] proportion of urban
    gender: str                  # 'female' | 'male' | 'nonbinary' | 'unspecified'
    height_cm: int               # integer centimeters

# -----------------------------
# Simulator
# -----------------------------
class TelematicsDataSimulator:
    """Generates realistic telematics data + exports POC CSVs"""

    def __init__(self, seed: int = 5):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self.driver_profiles: List[DriverProfile] = []

        # Behavior priors by risk bucket
        self.risk_parameters = {
            'low': {
                'speeding_probability': 0.05,
                'speed_variance_factor': 0.8,
                'night_driving_preference': 0.10
            },
            'medium': {
                'speeding_probability': 0.15,
                'speed_variance_factor': 1.0,
                'night_driving_preference': 0.20
            },
            'high': {
                'speeding_probability': 0.35,
                'speed_variance_factor': 1.5,
                'night_driving_preference': 0.40
            }
        }

    # -------- Driver Profiles --------
    def generate_driver_profiles(self, num_drivers: int) -> List[DriverProfile]:
        profiles: List[DriverProfile] = []

        for i in range(num_drivers):
            # Age weighted towards working age
            age = int(self.rng.choice(
                [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                p=[0.03, 0.06, 0.10, 0.12, 0.12, 0.12, 0.11, 0.10, 0.08, 0.07, 0.05, 0.03, 0.01]))

            # Driving experience
            experience = int(max(0, min(age - 16, self.rng.normal(10, 5))))

            # Risk level prior from age/experience
            if age < 25 or experience < 2:
                risk_weights = [0.2, 0.4, 0.4]  # more high risk when young/inexperienced
            elif age > 65:
                risk_weights = [0.3, 0.5, 0.2]  # mixed risk when older
            else:
                risk_weights = [0.45, 0.4, 0.15] # more low/medium in mid-age

            risk_level = self.rng.choice(['low', 'medium', 'high'], p=risk_weights)
            style_map = {'low': 'conservative', 'medium': 'normal', 'high': 'aggressive'}
            driving_style = style_map[risk_level]

            # Trip frequency
            trips_per_week = int(max(1, self.rng.normal(10, 3)))

            # Preferred hours
            if age < 30:
                preferred_times = [7, 8, 17, 18, 19, 20, 21, 22]
            elif age > 65:
                preferred_times = [9, 10, 11, 14, 15, 16]
            else:
                preferred_times = [7, 8, 12, 17, 18, 19]

            # Urban ratio
            urban_ratio = float(self.rng.beta(3, 2))

            # Gender + height
            gender = self.rng.choice(
                ["female", "male", "nonbinary", "unspecified"],
                p=[0.49, 0.49, 0.01, 0.01])
            if gender == "male":
                h_mean, h_sd = 175, 7
            elif gender == "female":
                h_mean, h_sd = 162, 6.5
            else:
                h_mean, h_sd = 168, 7
            height_cm = int(np.clip(self.rng.normal(h_mean, h_sd), 145, 200))

            profiles.append(DriverProfile(
                driver_id=f"driver_{i+1:04d}",
                age=age,
                experience_years=experience,
                risk_level=risk_level,
                driving_style=driving_style,
                typical_trips_per_week=trips_per_week,
                preferred_times=preferred_times,
                urban_ratio=urban_ratio,
                gender=gender,
                height_cm=height_cm))

        self.driver_profiles = profiles
        return profiles

    # -------- Trip Simulation --------
    def simulate_trip(self, profile: DriverProfile, trip_start: datetime, sample_period_sec: int = 30) -> pd.DataFrame:
        """Simulate a single trip at 30s cadence; returns point-level rows."""

        # Duration (minutes): longer at commute hours
        if trip_start.hour in COMMUTE_HOURS:
            duration_min = int(max(10, self.rng.normal(35, 12)))
        else:
            duration_min = int(max(5, self.rng.normal(20, 10)))

        n_points = int(duration_min * 60 // sample_period_sec)
        if n_points < 2:
            n_points = 2

        # Urban vs highway
        is_urban = self.rng.random() < profile.urban_ratio
        risk = self.risk_parameters[profile.risk_level]

        # Base speeds and limits
        if is_urban:
            base_speed = float(self.rng.normal(30, 6))
            speed_limit = 35
            traffic_factor = float(self.rng.uniform(0.7, 1.0))
        else:
            base_speed = float(self.rng.normal(65, 8))
            speed_limit = 70
            traffic_factor = float(self.rng.uniform(0.8, 1.0))

        # Speed time series (mph)
        speeds = []
        curr_speed = max(0.0, base_speed * traffic_factor + self.rng.normal(0, 2))
        for i in range(n_points):
            # occasional slowdowns/stops
            if self.rng.random() < 0.05:
                target = float(self.rng.uniform(0, 15))
            else:
                variance = risk['speed_variance_factor']
                target = base_speed * traffic_factor + self.rng.normal(0, 4 * variance)
                # speeding behavior
                if self.rng.random() < risk['speeding_probability']:
                    target = speed_limit + float(self.rng.uniform(5, 20))
            # smooth transitions
            max_change = float(self.rng.uniform(3, 20))  # mph per 30s
            diff = target - curr_speed
            diff = np.clip(diff, -max_change, max_change)
            curr_speed = max(0.0, curr_speed + diff)
            speeds.append(curr_speed)

        # Lat/Lon simple path
        start_lat = float(self.rng.uniform(39.0, 39.5))   # Baltimore-ish band
        start_lon = float(self.rng.uniform(-76.8, -76.3))
        lats, lons = self._generate_route(start_lat, start_lon, speeds)

        # Time indices
        timestamps = [trip_start + timedelta(seconds=i * sample_period_sec) for i in range(n_points)]
        ts_idx = list(range(n_points))
        hours = [t.hour for t in timestamps]

        # Weather (per point)
        weather = self.rng.choice(['clear', 'rain', 'snow', 'fog'], size=n_points, p=[0.7, 0.2, 0.05, 0.05])
        rain_flag = (weather == 'rain').astype(int)

        # Acceleration (mph/s) from speed diffs over 30s
        speed_arr = np.array(speeds, dtype=float)
        accel = np.zeros_like(speed_arr)
        accel[1:] = (speed_arr[1:] - speed_arr[:-1]) / (sample_period_sec / 1.0)  # mph per sec
        # events
        is_brake = accel < -0.5

        df = pd.DataFrame({
            "user_id": profile.driver_id,
            "trip_id": f"{profile.driver_id}_{int(trip_start.timestamp())}",
            "ts": ts_idx,
            "speed": speed_arr,
            "accel": accel,
            "limit": speed_limit,
            "is_brake": is_brake.astype(int),      # pipeline casts to bool anyway
            "lat": lats,
            "lon": lons,
            "hour": hours,
            "rain": rain_flag})
        return df

    def simulate_driver_period(self, profile: DriverProfile, start_date: datetime, days: int = 30, sample_period_sec: int = 30) -> pd.DataFrame:
        all_trips: List[pd.DataFrame] = []
        d = start_date
        for _ in range(days):
            # expected trips this day
            if d.weekday() < 5:
                lam = profile.typical_trips_per_week / 5 * 1.2
            else:
                lam = profile.typical_trips_per_week / 5 * 0.4
            n_trips = int(max(0, self.rng.poisson(lam)))

            for _ in range(n_trips):
                hour = int(self.rng.choice(profile.preferred_times))
                minute = int(self.rng.integers(0, 60))
                trip_time = d.replace(hour=hour, minute=minute, second=0, microsecond=0)
                trip_df = self.simulate_trip(profile, trip_time, sample_period_sec=sample_period_sec)
                all_trips.append(trip_df)

            d += timedelta(days=1)

        return pd.concat(all_trips, ignore_index=True) if all_trips else pd.DataFrame()

    def _generate_route(self, lat0: float, lon0: float, speeds: List[float]) -> Tuple[List[float], List[float]]:
        lats = [lat0]; lons = [lon0]
        bearing = float(self.rng.uniform(0, 2*np.pi))
        for v in speeds[1:]:
            if v <= 0:
                lats.append(lats[-1])
                lons.append(lons[-1])
                continue
            # distance in miles during 30s
            distance_miles = v * (30.0 / 3600.0)
            distance_deg = distance_miles / 69.0
            bearing += float(self.rng.normal(0, 0.08))
            new_lat = lats[-1] + distance_deg * np.cos(bearing)
            new_lon = lons[-1] + distance_deg * np.sin(bearing)
            lats.append(new_lat); lons.append(new_lon)
        return lats, lons

    # -------------------------
    # Export helpers
    # -------------------------
    def export_to_poc_csvs(
        self,
        points: pd.DataFrame,
        out_dir: str = "data",
        use_sqlite: bool = False,
        sqlite_path: str = "telematics_data.db"
    ) -> None:
        """
        Writes the POC CSVs the pipeline expects + labels.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1) trips.csv  — exactly what pipeline.py expects
        trips_cols = ["user_id", "trip_id", "ts", "speed", "accel", "limit", "is_brake", "lat", "lon", "hour", "rain"]
        trips = points[trips_cols].copy()
        trips.to_csv(out / "trips.csv", index=False)

        # 2) driver_profile.csv — include demographics
        drows = []
        for p in self.driver_profiles:
            drows.append({
                "user_id": p.driver_id,
                "policy_id": f"POL{self.rng.integers(10000, 99999)}",
                "num_drivers": int(self.rng.integers(1, 4)),
                "primary_use": self.rng.choice(["commute","pleasure","business"], p=[0.6,0.3,0.1]),
                "annual_mileage_declared": int(np.clip(self.rng.normal(12000, 4000), 3000, 35000)),
                "prior_claims_count": int(max(0, self.rng.poisson(0.3))),
                "tickets_count": int(max(0, self.rng.poisson(0.8))),
                "dui_flag": int(self.rng.binomial(1, 0.03)),
                "age": p.age,
                "gender": p.gender,
                "height_cm": p.height_cm,})
        dprof = pd.DataFrame(drows)
        dprof.to_csv(out / "driver_profile.csv", index=False)

        # 3) vehicle_info.csv — simple catalog
        makes_models = [
            ("Toyota","Camry"), ("Honda","Civic"), ("Honda","CR-V"), ("Ford","F-150"),
            ("Tesla","Model 3"), ("Subaru","Outback"), ("Hyundai","Elantra"), ("Nissan","Rogue")]
        vins, vin_rows = [], []
        for i in range(len(self.driver_profiles)):
            make, model = random.choice(makes_models)
            year = int(self.rng.integers(2006, 2025))
            variant = random.choice(["Base","Sport","Limited","Touring","XLT","SE","LE","EX"])
            airbags = int(self.rng.integers(2, 10))
            sensors = int(self.rng.integers(0, 8))
            safety_features_count = airbags + (sensors // 2)
            vid = f"veh_{i+1:04d}"
            vins.append(vid)
            vin_rows.append({
                "vehicle_id": vid,
                "make": make,
                "model": model,
                "variant": variant,
                "year": year,
                "airbags": airbags,
                "sensors": sensors,
                "safety_features_count": safety_features_count})
        vinfo = pd.DataFrame(vin_rows)
        vinfo.to_csv(out / "vehicle_info.csv", index=False)

        # 4) vehicle_assignments.csv — 1 vehicle per user (simple)
        assign_rows = []
        for i, p in enumerate(self.driver_profiles):
            assign_rows.append({"user_id": p.driver_id, "vehicle_id": vins[i % len(vins)]})
        vassign = pd.DataFrame(assign_rows)
        vassign.to_csv(out / "vehicle_assignments.csv", index=False)

        # 5) trip_labels.csv — synthetic claims + severities (trip-level)
        labels = self._generate_trip_labels(points, dprof)
        labels.to_csv(out / "trip_labels.csv", index=False)

        print(f"[sim] Wrote CSVs in {out.resolve()}")

        if use_sqlite:
            self.save_to_database(points, dprof, sqlite_path)
            print(f"[sim] SQLite snapshot -> {sqlite_path}")

    # -------------------------
    # Synthetic label generator
    # -------------------------
    def _generate_trip_labels(self, points: pd.DataFrame, dprof: pd.DataFrame) -> pd.DataFrame:
        """
        Create trip-level binary claim + severity based on per-trip aggregates.
        No demographic bias: gender not included in label logit; small age U-shape.
        """
        # Aggregate per trip
        g = points.groupby("trip_id", dropna=False)

        def frac_true(s):
            s = pd.to_numeric(s, errors="coerce")
            if s.isna().all(): return 0.0
            return float(np.nanmean(s > 0.5)) if s.dtype != bool else float(np.nanmean(s))

        agg = pd.DataFrame({
            "user_id": g["user_id"].first(),
            "miles": (g["speed"].mean() * (g["ts"].max() + 1) * (30/3600.0)),  # approx mph * hours
            "over_limit": (g.apply(lambda gg: float(np.mean(gg["speed"].to_numpy() > gg["limit"].to_numpy())))),
            "harsh_brakes": g["is_brake"].apply(lambda s: float(np.mean(pd.to_numeric(s, errors="coerce") < 0))),
            "mean_speed": g["speed"].mean(),
            "std_speed": g["speed"].std().fillna(0.0),
            "night": g["hour"].apply(lambda s: float(np.mean(s.isin(list(NIGHT_HOURS))))),
            "rain": g["rain"].mean(),}).reset_index()

        # Merge age for optional effect
        agg = agg.merge(dprof[["user_id","age"]], on="user_id", how="left")

        # Logistic model for claim probability (small, plausible effects)
        z = (
            2.1 * agg["over_limit"].fillna(0) +
            2.8 * agg["harsh_brakes"].fillna(0) +
            1.4 * agg["night"].fillna(0) +
            1.0 * agg["rain"].fillna(0) +
            0.05 * (agg["std_speed"].fillna(0) / 5.0) -
            0.00003 * agg["miles"].fillna(0)  # more miles, more exposure but also more stable
            )

        # Small age U-shape (no gender): higher claim tendency <25 and >70
        age = pd.to_numeric(agg["age"], errors="coerce").fillna(40)
        z += np.where(age < 25, 0.35, 0.0) + np.where(age > 70, 0.20, 0.0)

        p = 1.0 / (1.0 + np.exp(-z))
        # Calibrate to a reasonable base rate (clip & shrink)
        p = np.clip(p, 0.01, 0.35)

        rnd = self.rng.random(len(agg))
        claim = (rnd < p).astype(int)

        # Severity only when claim==1; gamma-like
        sev = np.zeros(len(agg), dtype=float)
        # severity scale up with harshness & night & rain
        sev_scale = 1.0 + 1.5*agg["harsh_brakes"].fillna(0) + 0.8*agg["night"].fillna(0) + 0.6*agg["rain"].fillna(0)
        sev[claim == 1] = np.clip(self.rng.gamma(shape=1.5, scale=sev_scale[claim==1].to_numpy()), 0.1, 10.0)

        labels = pd.DataFrame({
            "trip_id": agg["trip_id"],
            "user_id": agg["user_id"],
            "claim": claim,
            "severity": sev
        })
        return labels

    # -------------------------
    # Optional: SQLite sink
    # -------------------------
    def save_to_database(self, points: pd.DataFrame, dprof: pd.DataFrame, db_path: str = "telematics_data.db"):
        conn = sqlite3.connect(db_path)
        # Trips table (point-level)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                trip_id TEXT, user_id TEXT, ts INTEGER,
                speed REAL, accel REAL, limit REAL,
                is_brake INTEGER,
                lat REAL, lon REAL,
                hour INTEGER, rain INTEGER
            )
        """)
        points.to_sql("trips", conn, if_exists="append", index=False)

        # Driver profiles snapshot (include demographics)
        conn.execute("DROP TABLE IF EXISTS driver_profiles")
        dprof.to_sql("driver_profiles", conn, if_exists="replace", index=False)
        conn.close()


# -----------------------------
# CLI runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Telemetra telematics simulator (with demographics).")
    ap.add_argument("--drivers", type=int, default=100, help="Number of drivers to simulate.")
    ap.add_argument("--days", type=int, default=30, help="Days of data per driver.")
    ap.add_argument("--start_days_ago", type=int, default=60, help="Start offset in days from now.")
    ap.add_argument("--sample_period_sec", type=int, default=30, help="Sampling granularity.")
    ap.add_argument("--out_dir", type=str, default="data", help="Directory to write CSVs.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    ap.add_argument("--sqlite", action="store_true", help="Also write a SQLite snapshot.")
    ap.add_argument("--sqlite_path", type=str, default="telematics_data.db")
    args = ap.parse_args()

    sim = TelematicsDataSimulator(seed=args.seed)
    print(f"[sim] Generating {args.drivers} driver profiles …")
    profiles = sim.generate_driver_profiles(args.drivers)

    print(f"[sim] Simulating trips for {args.days} days per driver …")
    start_date = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) - timedelta(days=args.start_days_ago)

    all_points: List[pd.DataFrame] = []
    for p in profiles:
        df = sim.simulate_driver_period(p, start_date=start_date, days=args.days, sample_period_sec=args.sample_period_sec)
        if not df.empty:
            all_points.append(df)

    if not all_points:
        raise SystemExit("[sim] No trips were generated; try reducing drivers or increasing days.")

    points = pd.concat(all_points, ignore_index=True)

    print(f"[sim] Exporting CSVs to {args.out_dir} …")
    sim.export_to_poc_csvs(points, out_dir=args.out_dir, use_sqlite=args.sqlite, sqlite_path=args.sqlite_path)
    print("[sim] Done.")

if __name__ == "__main__":
    main()
