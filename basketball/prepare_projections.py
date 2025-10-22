import math
import pandas as pd

IN_PATH = r".\NBA.csv"
OUT_PATH = r".\projections_for_app.csv"

df = pd.read_csv(IN_PATH)

# --- Helper functions ---

def num(series_name: str):
    """Convert a column to numeric, or return NaN series if missing."""
    if series_name not in df.columns:
        return pd.Series([math.nan] * len(df))
    return pd.to_numeric(df[series_name], errors="coerce")

def with_cap(val: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clamp values between lo and hi."""
    return val.clip(lower=lo, upper=hi)

def ratio(series: pd.Series, expo: float, lo: float, hi: float) -> pd.Series:
    """Normalize a series to its median, apply exponent and cap."""
    med = series.median(skipna=True)
    if pd.notna(med) and med != 0:
        r = (series / med) ** expo
        return with_cap(r.fillna(1.0), lo, hi)
    return pd.Series(1.0, index=series.index)

# --- Core projections ---

# Use Proj from CSV (since there's no Proj_adj or FP)
proj_adj = num("Proj")
fp_raw = proj_adj  # fallback placeholder, since FP doesn't exist
base_mean = proj_adj.fillna(fp_raw)

# Minutes and rate-based alternative projection
fppm = num("FPPM")
min_now = num("Min")
min_l3 = num("MinL3")
minutes_avail = min_now.fillna(min_l3)
alt_mean = fppm * minutes_avail

# Blend mean: prioritize base, otherwise alt
combined_mean = base_mean.copy()
mask_missing = combined_mean.isna()
combined_mean[mask_missing] = alt_mean[mask_missing]
# If both present, blend 60/40
both_mask = (~base_mean.isna()) & (~alt_mean.isna())
combined_mean[both_mask] = 0.6 * base_mean[both_mask] + 0.4 * alt_mean[both_mask]

# --- Scaling factors for ceiling ---
dvp = num("DvP")
xpts = num("xPts")
pace = num("Game Pace")
usg = num("USG%")

pace_factor = ratio(pace, 0.5, 0.90, 1.10)
teamx_factor = ratio(xpts, 0.5, 0.90, 1.10)
dvp_factor = with_cap(dvp.fillna(1.0), 0.90, 1.10) if dvp.notna().any() else pd.Series(1.0, index=df.index)
usg_factor = ratio(usg, 0.3, 0.95, 1.08)

ceil_mult_base = 1.15
ceiling = combined_mean * ceil_mult_base * pace_factor * teamx_factor * dvp_factor * usg_factor

# --- Build output DataFrame ---
out = pd.DataFrame({
    "id": df.get("Ids", pd.Series(dtype=str)),
    "name": df.get("Name", pd.Series(dtype=str)).astype(str).str.strip(),
    "team": df.get("Team", pd.Series(dtype=str)).astype(str).str.strip(),
    "position": df.get("Pos", pd.Series(dtype=str)).astype(str).str.strip(),
    "salary": pd.to_numeric(df.get("Salary", pd.Series(dtype=float)), errors="coerce"),
    "fpts": combined_mean,
    "ceiling": ceiling,
    "ownership": pd.NA,
}).dropna(subset=["position", "salary"]).reset_index(drop=True)

# Keep only valid DFS positions
valid_pos = {"PG", "SG", "SF", "PF", "C"}
out = out[out["position"].isin(valid_pos)].reset_index(drop=True)

# --- Write output ---
out.to_csv(OUT_PATH, index=False)
print(
    f"✅ Wrote {OUT_PATH} | rows: {len(out)} | fpts notna: {out['fpts'].notna().sum()} | ceiling notna: {out['ceiling'].notna().sum()}"
)
