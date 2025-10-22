# scripts/prepare_tournament_data.py
import pandas as pd
import numpy as np

TOURNAMENT_CSV = "../data/tournament.csv"
OUTPUT_CSV = "../data/tournament_for_app.csv"

print("Loading tournament data...")
df = pd.read_csv(TOURNAMENT_CSV)

print(f"Loaded {len(df)} players from tournament.csv")

# Generate realistic stats based on position and price
np.random.seed(42)  # For reproducible results

def generate_stats(row):
    pos = row['Position'].lower()
    price = float(row['Price'])
    
    # Base stats on position and price (higher price = better stats)
    price_factor = price / 15.0  # Normalize price factor
    
    if pos == 'goalkeeper':
        return {
            'goals': 0,
            'assists': np.random.randint(0, 3),
            'shots_on_target': np.random.randint(0, 2),
            'clean_sheet_prob': 0.3 + (price_factor * 0.4),  # 0.3 to 0.7
            'yellow_cards': np.random.randint(0, 3),
            'red_cards': np.random.randint(0, 1),
            'saves': np.random.randint(50, 150),
            'penalty_saves': np.random.randint(0, 3),
            'minutes_played': np.random.randint(2000, 3000)
        }
    elif pos == 'defender':
        return {
            'goals': np.random.randint(0, int(3 * price_factor + 1)),
            'assists': np.random.randint(0, int(5 * price_factor + 2)),
            'shots_on_target': np.random.randint(2, int(15 * price_factor + 5)),
            'clean_sheet_prob': 0.2 + (price_factor * 0.5),  # 0.2 to 0.7
            'yellow_cards': np.random.randint(0, 5),
            'red_cards': np.random.randint(0, 1),
            'saves': 0,
            'penalty_saves': 0,
            'minutes_played': np.random.randint(1500, 3000)
        }
    elif pos == 'midfielder':
        return {
            'goals': np.random.randint(0, int(8 * price_factor + 2)),
            'assists': np.random.randint(0, int(10 * price_factor + 3)),
            'shots_on_target': np.random.randint(5, int(25 * price_factor + 10)),
            'clean_sheet_prob': 0.1 + (price_factor * 0.2),  # 0.1 to 0.3
            'yellow_cards': np.random.randint(0, 6),
            'red_cards': np.random.randint(0, 1),
            'saves': 0,
            'penalty_saves': 0,
            'minutes_played': np.random.randint(1500, 3000)
        }
    else:  # forward
        return {
            'goals': np.random.randint(int(5 * price_factor), int(20 * price_factor + 5)),
            'assists': np.random.randint(0, int(8 * price_factor + 2)),
            'shots_on_target': np.random.randint(10, int(50 * price_factor + 20)),
            'clean_sheet_prob': 0.0,  # Forwards don't get clean sheet points
            'yellow_cards': np.random.randint(0, 4),
            'red_cards': np.random.randint(0, 1),
            'saves': 0,
            'penalty_saves': 0,
            'minutes_played': np.random.randint(1500, 3000)
        }

# Apply stats generation
print("Generating realistic stats for all players...")
stats_list = []
for _, row in df.iterrows():
    stats = generate_stats(row)
    stats_list.append(stats)

stats_df = pd.DataFrame(stats_list)
df_with_stats = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

# Compute fantasy points
def compute_football_fpts(row):
    fpts = 0.0
    pos = str(row['Position']).lower()
    
    # Minutes played
    minutes = float(row['minutes_played'])
    fpts += 1.0 if minutes > 0 else 0.0
    fpts += 3.0 if minutes >= 60 else 0.0
    fpts += 1.0 if minutes >= 90 else 0.0
    
    # Goals and assists
    goals = float(row['goals'])
    assists = float(row['assists'])
    shots_on_target = float(row['shots_on_target'])
    
    # Position-based goal scoring
    if pos == 'goalkeeper':
        fpts += goals * 8.0
        fpts += shots_on_target * 1.0
    elif pos == 'defender':
        fpts += goals * 6.0
        fpts += shots_on_target * 0.6
    elif pos == 'midfielder':
        fpts += goals * 5.0
        fpts += shots_on_target * 0.4
    elif pos == 'forward':
        fpts += goals * 4.0
        fpts += shots_on_target * 0.4
    
    # Assists
    fpts += assists * 3.0
    
    # Clean sheets
    clean_sheet_prob = float(row['clean_sheet_prob'])
    if pos in ['goalkeeper', 'defender']:
        fpts += clean_sheet_prob * 4.0
    elif pos == 'midfielder':
        fpts += clean_sheet_prob * 1.0
    
    # Cards (negative)
    yellow_cards = float(row['yellow_cards'])
    red_cards = float(row['red_cards'])
    fpts += yellow_cards * -1.0
    fpts += red_cards * -3.0
    
    # Saves (goalkeepers)
    if pos == 'goalkeeper':
        saves = float(row['saves'])
        fpts += saves * 0.5
        
        penalty_saves = float(row['penalty_saves'])
        fpts += penalty_saves * 5.0
    
    return float(fpts)

print("Computing fantasy points...")
df_with_stats['fpts'] = df_with_stats.apply(compute_football_fpts, axis=1)

# Prepare final output
output_df = df_with_stats[['PlayerID', 'Name', 'Club', 'Position', 'Price', 'fpts']].copy()
output_df = output_df.rename(columns={
    'PlayerID': 'id',
    'Name': 'name',
    'Club': 'team',
    'Position': 'position',
    'Price': 'salary'
})

# Map positions to standard names
pos_mapping = {
    'goalkeeper': 'goalkeeper',
    'defender': 'defender', 
    'midfielder': 'midfielder',
    'forward': 'forward'
}
output_df['position'] = output_df['position'].str.lower().map(pos_mapping).fillna(output_df['position'])

# Add ownership column
output_df['ownership'] = 0.0

# Keep only valid positions
valid_pos = {'goalkeeper', 'defender', 'midfielder', 'forward'}
output_df = output_df[output_df['position'].isin(valid_pos)].reset_index(drop=True)

# Save final output
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {OUTPUT_CSV} with {len(output_df)} players")
print(f"Position breakdown:")
print(output_df['position'].value_counts())
