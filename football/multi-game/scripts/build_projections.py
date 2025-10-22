# scripts/build_projections.py
import pandas as pd

RAW_STATS_CSV = "../data/champions_league_raw_stats.csv"
PROJECTIONS_OUT = "../data/projections.csv"

df = pd.read_csv(RAW_STATS_CSV)

# Ensure numeric
num_cols = ['Min','Sh','SoT','xG','xA','Gls','Ast']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Compute fantasy points based on your scoring system
def compute_fpts(row):
    fpts = 0
    pos = row['Pos'].lower()
    # Appearance
    fpts += 1 if row['Min']>0 else 0
    fpts += 3 if row['Min']>=60 else 0
    # Assists
    fpts += row['xA']*3 + row.get('Ast',0)*3
    # Goals, position-based
    if pos == 'gk':
        fpts += row.get('Gls',0)*8
        fpts += 0.7*4  # Clean sheet probability estimate
        fpts += row.get('SoT',0)*1
    elif pos == 'defender':
        fpts += row.get('Gls',0)*6
        fpts += 0.6*4  # Clean sheet probability estimate
        fpts += row.get('SoT',0)*0.6
    elif pos == 'midfielder':
        fpts += 1 if row['Min']>=90 else 0
        fpts += row.get('Gls',0)*5
        fpts += 0.3*1  # Clean sheet probability estimate
        fpts += row.get('SoT',0)*0.4
    elif pos == 'forward' or pos == 'fw':
        fpts += 1 if row['Min']>=90 else 0
        fpts += row.get('Gls',0)*4
        fpts += row.get('SoT',0)*0.4
    return fpts

df['fpts'] = df.apply(compute_fpts, axis=1)
df['ceiling'] = df['fpts']*1.15

# Add required columns for optimizer
df['Ids'] = range(1, len(df) + 1)  # Generate IDs
df['Name'] = df['Player']  # Use Player as Name
df['Team'] = df['Squad']   # Use Squad as Team
df['Salary'] = 10.0  # Default salary

# Save projections
proj_out = df[['Ids','Name','Team','Pos','Salary','fpts','ceiling']].copy()
proj_out.to_csv(PROJECTIONS_OUT, index=False)
print(f"Saved {PROJECTIONS_OUT} | rows: {len(proj_out)}")
