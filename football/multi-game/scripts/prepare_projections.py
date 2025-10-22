# scripts/prepare_projections.py
import pandas as pd

PROJECTIONS_CSV = "../data/projections.csv"
PREPARED_OUT = "../data/projections_for_app.csv"

df = pd.read_csv(PROJECTIONS_CSV)

# Ensure numeric
for col in ['Salary','fpts','ceiling']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Clip ceiling to at least fpts
df['ceiling'] = df['ceiling'].clip(lower=df['fpts'], upper=df['ceiling'].max())
df['Salary'] = df['Salary'].clip(lower=1)

# Final output
out = df.rename(columns={
    'Ids':'id',
    'Name':'name',
    'Team':'team',
    'Pos':'position',
    'Salary':'salary',
    'fpts':'fpts',
    'ceiling':'ceiling'
}).copy()
out['ownership'] = pd.NA

# Map positions to standard names
pos_mapping = {
    'GK': 'goalkeeper',
    'DEF': 'defender', 
    'MID': 'midfielder',
    'FW': 'forward'
}
out['position'] = out['position'].map(pos_mapping).fillna(out['position'])

# Keep only valid positions
valid_pos = {'goalkeeper','defender','midfielder','forward'}
out = out[out['position'].str.lower().isin(valid_pos)].reset_index(drop=True)

out.to_csv(PREPARED_OUT, index=False)
print(f"Saved {PREPARED_OUT} | rows: {len(out)}")
