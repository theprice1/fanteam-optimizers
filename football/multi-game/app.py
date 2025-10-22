import io
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pulp as pl
import streamlit as st

# Football/soccer specific constants
VALID_POSITIONS = ["goalkeeper", "defender", "midfielder", "forward"]


def compute_football_fpts(row: pd.Series) -> float:
    """Compute fantasy points for football/soccer based on standard scoring"""
    fpts = 0.0
    pos = str(row.get("position", "")).lower()
    
    # Minutes played
    minutes = float(row.get("minutes_played", 0) or 0)
    fpts += 1.0 if minutes > 0 else 0.0
    fpts += 3.0 if minutes >= 60 else 0.0
    fpts += 1.0 if minutes >= 90 else 0.0
    
    # Goals and assists
    goals = float(row.get("goals", 0) or 0)
    assists = float(row.get("assists", 0) or 0)
    shots_on_target = float(row.get("shots_on_target", 0) or 0)
    
    # Position-based goal scoring
    if pos == "goalkeeper":
        fpts += goals * 8.0
        fpts += shots_on_target * 1.0
    elif pos == "defender":
        fpts += goals * 6.0
        fpts += shots_on_target * 0.6
    elif pos == "midfielder":
        fpts += goals * 5.0
        fpts += shots_on_target * 0.4
    elif pos == "forward":
        fpts += goals * 4.0
        fpts += shots_on_target * 0.4
    
    # Assists
    fpts += assists * 3.0
    
    # Clean sheets
    clean_sheet_prob = float(row.get("clean_sheet_prob", 0) or 0)
    if pos in ["goalkeeper", "defender"]:
        fpts += clean_sheet_prob * 4.0
    elif pos == "midfielder":
        fpts += clean_sheet_prob * 1.0
    
    # Cards (negative)
    yellow_cards = float(row.get("yellow_cards", 0) or 0)
    red_cards = float(row.get("red_cards", 0) or 0)
    fpts += yellow_cards * -1.0
    fpts += red_cards * -3.0
    
    # Saves (goalkeepers)
    if pos == "goalkeeper":
        saves = float(row.get("saves", 0) or 0)
        fpts += saves * 0.5
        
        penalty_saves = float(row.get("penalty_saves", 0) or 0)
        fpts += penalty_saves * 5.0
    
    return float(fpts)


def normalize_schema_or_compute(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Normalize input data to optimizer schema or compute fpts from raw stats"""
    original_cols = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in original_cols}
    
    def has(col: str) -> bool:
        return col in cols_lower
    
    # Already in optimizer schema
    if all(has(c) for c in ["id", "name", "team", "position", "salary"]) and (
        has("fpts") or has("proj")
    ):
        out = df.copy()
        out.columns = [c.lower() for c in out.columns]
        if "proj" in out.columns and "fpts" not in out.columns:
            out["fpts"] = pd.to_numeric(out["proj"], errors="coerce")
        out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
        out["fpts"] = pd.to_numeric(out["fpts"], errors="coerce")
        if "ownership" not in out.columns:
            out["ownership"] = 0.0
        out = out[out["position"].str.lower().isin(VALID_POSITIONS)].dropna(subset=["salary", "fpts"]).reset_index(drop=True)
        out["position"] = out["position"].str.lower()
        return out, "Detected optimizer schema with fpts/proj."
    
    # Raw stats schema: compute fpts
    required_like = ["id", "name", "team", "position", "salary"]
    if all(any(c.lower() == r.lower() for c in original_cols) for r in required_like):
        out = df.copy()
        # Normalize names
        rename_map = {c: c.lower() for c in out.columns}
        out.rename(columns=rename_map, inplace=True)
        out["position"] = out["position"].str.lower()
        out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
        
        # Compute fpts per row
        out["fpts"] = out.apply(compute_football_fpts, axis=1)
        if "ownership" in out.columns:
            out["ownership"] = pd.to_numeric(out["ownership"], errors="coerce").fillna(0.0)
        else:
            out["ownership"] = 0.0
        
        out = out[out["position"].isin(VALID_POSITIONS)].dropna(subset=["salary"]).reset_index(drop=True)
        return out, "Computed fpts from raw stats per football scoring."
    
    raise ValueError(f"Unrecognized schema. Columns found: {original_cols}")


def build_lineup_ilp(
    players: pd.DataFrame,
    budget: float,
    lineup_size: int,
    min_per_pos: Dict[str, int],
    max_per_pos: Dict[str, int],
    max_same_team: int,
    ownership_weight: float = 0.0,
    eps_budget_tiebreak: float = 1e-6,
    banned_player_ids: set | None = None,
) -> Tuple[List[int], float, float, float]:
    """Build optimal lineup using integer linear programming"""
    if banned_player_ids is None:
        banned_player_ids = set()
    
    idxs = list(range(len(players)))
    prob = pl.LpProblem("football_multi_optimizer", pl.LpMaximize)
    x = pl.LpVariable.dicts("x", idxs, lowBound=0, upBound=1, cat=pl.LpBinary)
    
    # Bans
    for i in idxs:
        if players.iloc[i]["id"] in banned_player_ids:
            prob += x[i] == 0
    
    # Squad size and budget
    prob += pl.lpSum([x[i] for i in idxs]) == lineup_size
    prob += pl.lpSum([players.iloc[i]["salary"] * x[i] for i in idxs]) <= budget
    
    # Position constraints
    for pos in VALID_POSITIONS:
        pos_idx = [i for i in idxs if str(players.iloc[i]["position"]).lower() == pos]
        if pos_idx:
            prob += pl.lpSum([x[i] for i in pos_idx]) >= min_per_pos.get(pos, 0)
            prob += pl.lpSum([x[i] for i in pos_idx]) <= max_per_pos.get(pos, lineup_size)
        else:
            if min_per_pos.get(pos, 0) > 0:
                return [], 0.0, 0.0, 0.0
    
    # Team cap
    for team, team_group in players.groupby("team"):
        team_idx = [int(i) for i in team_group.index]
        prob += pl.lpSum([x[i] for i in team_idx]) <= max_same_team
    
    total_fpts = pl.lpSum([players.iloc[i]["fpts"] * x[i] for i in idxs])
    total_salary = pl.lpSum([players.iloc[i]["salary"] * x[i] for i in idxs])
    leftover = budget - total_salary
    
    if "ownership" in players.columns and ownership_weight != 0.0:
        total_own = pl.lpSum([players.iloc[i]["ownership"] * x[i] for i in idxs])
        objective = total_fpts - ownership_weight * total_own + eps_budget_tiebreak * leftover
    else:
        objective = total_fpts + eps_budget_tiebreak * leftover
    
    prob += objective
    
    status = prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[status] != "Optimal":
        return [], 0.0, 0.0, 0.0
    
    selected = [i for i in idxs if pl.value(x[i]) > 0.5]
    fpts = float(sum(players.iloc[i]["fpts"] for i in selected))
    salary = float(sum(players.iloc[i]["salary"] for i in selected))
    own = float(sum(players.iloc[i]["ownership"] for i in selected)) if "ownership" in players.columns else 0.0
    return selected, fpts, salary, own


def generate_lineups(
    base_df: pd.DataFrame,
    num_lineups: int,
    budget: float,
    lineup_size: int,
    min_uniques: int,
    max_exposure_pct: float,
    ownership_weight: float,
    min_per_pos: Dict[str, int],
    max_per_pos: Dict[str, int],
    max_same_team: int,
) -> List[Dict[str, Any]]:
    """Generate multiple unique lineups"""
    df = base_df.copy().reset_index(drop=True)
    if "ownership" not in df.columns:
        df["ownership"] = 0.0
    
    exposure_counts: Dict[Any, int] = {}
    max_count = max(1, int(np.floor(max_exposure_pct * num_lineups)))
    results: List[Dict[str, Any]] = []
    
    for k in range(num_lineups):
        banned = {pid for pid, cnt in exposure_counts.items() if cnt >= max_count}
        
        sel, f, s, o = build_lineup_ilp(
            df,
            budget,
            lineup_size,
            min_per_pos,
            max_per_pos,
            max_same_team,
            ownership_weight=ownership_weight,
            eps_budget_tiebreak=1e-6,
            banned_player_ids=banned,
        )
        if not sel:
            break
        
        ids = [df.iloc[i]["id"] for i in sel]
        
        # Uniqueness
        if results and min_uniques > 0:
            tries = 0
            while tries < 30:
                worst_overlap = 0
                for prev in results:
                    overlap = sum(1 for pid in ids if pid in set(prev["player_ids"]))
                    worst_overlap = max(worst_overlap, overlap)
                if (lineup_size - worst_overlap) >= min_uniques:
                    break
                
                overlap_all = set().union(*[set(prev["player_ids"]) for prev in results])
                overlap_curr = [pid for pid in ids if pid in overlap_all]
                if not overlap_curr:
                    break
                ban_pid = df.set_index("id").loc[overlap_curr].sort_values("salary", ascending=False).index[0]
                banned.add(ban_pid)
                sel, f, s, o = build_lineup_ilp(
                    df,
                    budget,
                    lineup_size,
                    min_per_pos,
                    max_per_pos,
                    max_same_team,
                    ownership_weight=ownership_weight,
                    eps_budget_tiebreak=1e-6,
                    banned_player_ids=banned,
                )
                if not sel:
                    break
                ids = [df.iloc[i]["id"] for i in sel]
                tries += 1
        
        if not sel:
            break
        
        results.append({
            "indices": sel,
            "player_ids": ids,
            "fpts": f,
            "salary": s,
            "ownership": o,
        })
        
        for pid in ids:
            exposure_counts[pid] = exposure_counts.get(pid, 0) + 1
    
    return results


def layout_optimizer_tab():
    st.header("Football Multi-Game Optimizer")
    st.caption("Upload projections or raw stats. Supports optimizer schema or raw stats with fpts computed per rules.")
    
    with st.sidebar:
        st.subheader("Settings")
        budget = st.number_input("Budget (M)", value=100.0, min_value=1.0, step=1.0)
        lineup_size = st.number_input("Squad size", value=11, min_value=8, max_value=15, step=1)
        
        min_per_pos = {"goalkeeper": 1, "defender": 3, "midfielder": 3, "forward": 2}
        max_per_pos = {"goalkeeper": 1, "defender": 5, "midfielder": 5, "forward": 3}
        st.write("Position limits:")
        st.write("GK 1/1, DEF 3/5, MID 3/5, FWD 2/3")
        
        max_same_team = st.number_input("Max players per team", value=3, min_value=1, max_value=6, step=1)
        
        num_lineups = st.number_input("Number of lineups", value=10, min_value=1, max_value=500, step=1)
        min_uniques = st.number_input("Minimum uniques between lineups", value=2, min_value=0, max_value=11, step=1)
        max_exposure_pct = st.slider("Max exposure per player (%)", min_value=10, max_value=100, value=60, step=5) / 100.0
        ownership_weight = st.slider("Ownership weight (fpts - weight*ownership)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        
        st.caption("Tie-breaker: if fpts tie, prefers higher remaining budget.")
    
    sample_btn = st.button("Download CSV template")
    if sample_btn:
        # Template with raw stats columns and/or fpts
        sample_csv = (
            "id,name,team,position,salary,minutes_played,goals,assists,shots_on_target,clean_sheet_prob,yellow_cards,red_cards,saves,penalty_saves\n"
            "101,Player GK,TEAM,goalkeeper,10.0,90,0,0,0,0.7,0,0,3,0\n"
            "102,Player DEF,TEAM,defender,9.0,90,1,0,2,0.6,1,0,0,0\n"
            "103,Player MID,TEAM,midfielder,11.0,90,1,2,3,0.3,0,0,0,0\n"
            "104,Player FWD,TEAM,forward,12.0,90,2,1,4,0.0,0,0,0,0\n"
        )
        sample = pd.read_csv(io.StringIO(sample_csv))
        st.download_button(
            label="football_template.csv",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="football_template.csv",
            mime="text/csv",
        )
    
    uploaded = st.file_uploader("Choose projections or stats CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        return
    
    try:
        raw = pd.read_csv(uploaded)
        df, msg = normalize_schema_or_compute(raw)
        st.success(msg)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return
    
    with st.expander("Data preview"):
        st.dataframe(df.head(50))
    
    if st.button("Build lineups"):
        try:
            results = generate_lineups(
                df,
                num_lineups=int(num_lineups),
                budget=float(budget),
                lineup_size=int(lineup_size),
                min_uniques=int(min_uniques),
                max_exposure_pct=float(max_exposure_pct),
                ownership_weight=float(ownership_weight),
                min_per_pos=min_per_pos,
                max_per_pos=max_per_pos,
                max_same_team=int(max_same_team),
            )
        except Exception as e:
            st.error(f"Failed to generate lineups: {e}")
            return
        
        if not results:
            st.error("No feasible lineup found. Check data, budget, team cap, and position availability.")
            return
        
        all_rows = []
        for i, res in enumerate(results, start=1):
            lineup_df = df.iloc[res["indices"]].copy()
            lineup_df = lineup_df[["id", "name", "team", "position", "salary", "fpts", "ownership"]]
            lineup_df["lineup"] = i
            all_rows.append(lineup_df)
        
        out_df = pd.concat(all_rows, ignore_index=True)
        st.success(f"Generated {out_df['lineup'].nunique()} lineups.")
        st.dataframe(out_df)
        
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download lineups CSV", csv_bytes, file_name="football_lineups.csv", mime="text/csv")


def main():
    st.title("Fanteam Football Optimizer - Multi-Game")
    tab1, = st.tabs(["Optimizer"])
    with tab1:
        layout_optimizer_tab()
    
    st.markdown("---")
    st.markdown(
        "- Team cap: up to 3 from the same team.\n"
        "- Position limits: GK 1/1, DEF 3/5, MID 3/5, FWD 2/3.\n"
        "- Budget tie-breaker: if fpts tie, prefers higher remaining budget."
    )


if __name__ == "__main__":
    main()
