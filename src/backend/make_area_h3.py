# src/backend/make_area_h3.py
import argparse, numpy as np, pandas as pd
import h3

def to_h3(lat, lon, res: int):
    # Works with h3 v3 and v4
    try:
        return h3.geo_to_h3(float(lat), float(lon), res)           # v3
    except AttributeError:
        return h3.latlng_to_cell(float(lat), float(lon), res)      # v4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="data/trips.csv")
    ap.add_argument("--out", default="data/area_context.csv")
    ap.add_argument("--h3_res", type=int, default=8)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    tr = pd.read_csv(args.trips, usecols=["lat","lon"])
    keys = tr.apply(lambda r: to_h3(r["lat"], r["lon"], args.h3_res), axis=1).dropna().unique()

    rng = np.random.default_rng(args.seed)
    area = pd.DataFrame({"segment_key": keys})
    area["crime_index"]               = rng.uniform(0.7, 1.5, len(area))
    area["accident_index"]            = rng.uniform(0.7, 1.8, len(area))
    area["repair_cost_index"]         = rng.uniform(0.9, 1.4, len(area))
    area["parts_availability_index"]  = rng.uniform(0.7, 1.3, len(area))
    area.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(area)} H3 cells (res={args.h3_res}).")

if __name__ == "__main__":
    main()
