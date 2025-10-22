Fanteam Optimizers

Multi-sport Streamlit optimizers for Fanteam (basketball, NFL, football). Each sport has its own app entry while sharing utilities in common/.

Repository structure

fanteam/
  common/                (shared helpers: schema, optimizer, io, utils)
  basketball/            (working Streamlit app)
    app.py
    templates/
      projections_template.csv
  nfl/
    single-game/app.py
    multi-game/app.py
  football/
    single-game/app.py
    multi-game/app.py
  requirements.txt       (shared across apps; Cloud installs from root)
  runtime.txt            (python-3.11 for Streamlit Cloud)
  .gitignore
  README.md

Local development (PowerShell on Windows)

cd "C:\Users\thepr\Desktop\fanteam"
python -m venv .venv
 .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Run basketball app

streamlit run .\basketball\app.py

Run NFL multi-game app

streamlit run .\nfl\multi-game\app.py

Run football multi-game app

streamlit run .\football\multi-game\app.py

Future (once single-game apps are ready)

streamlit run .\nfl\single-game\app.py
streamlit run .\football\single-game\app.py

Prepare basketball projections

Convert the Euroleague CSV into an optimizer-ready file with mean and ceiling projections.

cd "C:\Users\thepr\Desktop\fanteam\basketball"
python .\prepare_projections.py

Output: .\basketball\projections_for_app.csv (id,name,team,position,salary,fpts,ceiling,ownership)

Then in the basketball app sidebar:
- Choose Ceiling (or Weighted Tournament Score) as the objective
- Upload projections_for_app.csv

NFL multi-game

The NFL multi-game optimizer is ready to use. Sample data is provided in `nfl/multi-game/data/nfl_sample.csv`.

Features:
- Supports both raw stats (with computed fpts) and pre-computed projections
- Position constraints: QB 1/1, RB 2/3, WR 3/4, TE 1/2, DEF 1/1
- Team cap: up to 3 players per team
- Ownership weighting and exposure controls
- Multiple lineup generation with uniqueness constraints

football multi-game

The football multi-game optimizer is ready to use. Sample data is provided in `football/multi-game/data/football_sample.csv`.

Features:
- Supports both raw stats (with computed fpts) and pre-computed projections
- Position constraints: GK 1/1, DEF 3/5, MID 3/5, FWD 2/4
- Team cap: up to 3 players per team
- Ownership weighting and exposure controls
- Multiple lineup generation with uniqueness constraints

# Activate your venv
& ../.venv/Scripts/Activate.ps1

# Prepare tournament data for optimizer
cd football/multi-game
python scripts/prepare_tournament_data.py


Deploy on Streamlit Community Cloud

1) Push this repo to GitHub.
2) In Streamlit Cloud: New app, select repo/branch.
3) App file path per app, e.g. basketball/app.py.
4) Cloud uses runtime.txt (python-3.11) and installs from root requirements.txt.

Notes

- Keep raw/large datasets out of Git; only tiny samples in templates/.
- Shared logic should live in common/ and be imported from sport apps.
- If a sport needs extra heavy deps, consider keeping them local-only to avoid bloating Cloud deploys.


