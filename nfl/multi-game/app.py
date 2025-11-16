import io
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pulp as pl
import streamlit as st


VALID_POSITIONS = ["QB", "RB", "WR", "TE", "DST"]


def compute_offense_fpts(row: pd.Series) -> float:
    py = float(row.get("passing_yards", 0) or 0)
    ry = float(row.get("rushing_yards", 0) or 0)
    recy = float(row.get("receiving_yards", 0) or 0)
    ptd = float(row.get("passing_td", 0) or 0)
    rtd = float(row.get("rushing_td", 0) or 0)
    rectd = float(row.get("receiving_td", 0) or 0)
    rettd = float(row.get("return_td", 0) or 0)
    ints = float(row.get("interception", 0) or 0)  # thrown
    rec = float(row.get("reception", 0) or 0)
    fuml = float(row.get("fumble_lost", 0) or 0)
    two_pt = float(row.get("two_point", 0) or 0)

    pts = 0.0
    pts += 0.04 * py
    pts += 0.1 * ry
    pts += 0.1 * recy
    pts += 4.0 * ptd
    pts += 6.0 * rtd
    pts += 6.0 * rectd
    pts += 6.0 * rettd
    pts += 1.0 * rec
    pts += 2.0 * two_pt
    pts += -2.0 * ints
    pts += -2.0 * fuml
    return float(pts)


def defense_points_allowed_bonus(points_allowed: float) -> float:
    if points_allowed <= 0:
        return 10.0
    if 1 <= points_allowed <= 6:
        return 7.0
    if 7 <= points_allowed <= 13:
        return 4.0
    if 14 <= points_allowed <= 20:
        return 1.0
    if 21 <= points_allowed <= 27:
        return 0.0
    if 28 <= points_allowed <= 34:
        return -1.0
    return -4.0


def compute_defense_fpts(row: pd.Series) -> float:
    sack = float(row.get("sack", 0) or 0)
    dint = float(row.get("def_interception", 0) or 0)
    fumrec = float(row.get("fumble_recovery", 0) or 0)
    safety = float(row.get("safety", 0) or 0)
    blk = float(row.get("blocked_kick", 0) or 0)
    deftd = float(row.get("def_td", 0) or 0)
    rettd = float(row.get("return_td", 0) or 0)
    pts_allowed = float(row.get("points_allowed", 0) or 0)

    pts = 0.0
    pts += 1.0 * sack
    pts += 2.0 * dint
    pts += 2.0 * fumrec
    pts += 2.0 * safety
    pts += 2.0 * blk
    pts += 6.0 * deftd
    pts += 6.0 * rettd
    pts += defense_points_allowed_bonus(pts_allowed)
    return float(pts)


def normalize_schema_or_compute(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    original_cols = [c.strip() for c in df.columns]
    cols_lower = {c.lower(): c for c in original_cols}

    def has(col: str) -> bool:
        return col in cols_lower

    # Case 1: New schema format (Name, Pos, Team, Salary, Proj, ...)
    if has("name") and has("pos") and has("team") and has("salary") and has("proj"):
        name_col = cols_lower["name"]
        pos_col = cols_lower["pos"]
        team_col = cols_lower["team"]
        salary_col = cols_lower["salary"]
        proj_col = cols_lower["proj"]
        
        # Build base dataframe
        out = pd.DataFrame({
            "name": df[name_col].astype(str).str.strip(),
            "team": df[team_col].astype(str).str.strip(),
            "position": df[pos_col].astype(str).str.strip().str.upper(),
            "salary": pd.to_numeric(df[salary_col], errors="coerce"),
            "fpts": pd.to_numeric(df[proj_col], errors="coerce"),
        })
        
        # Handle ID - prefer Ids column, fallback to index
        if has("ids"):
            ids_col = cols_lower["ids"]
            out["id"] = df[ids_col]
        else:
            out["id"] = range(len(out))
        
        # Handle ownership - not present in this schema, default to 0
        out["ownership"] = 0.0
        
        # Preserve optional columns for weighted scoring
        optional_cols = {
            "value": "value",
            "lineup": "lineup_status",
            "form": "form",
            "lastpoints": "lastpoints",
            "totalpoints": "totalpoints",
        }
        
        for lower_name, target_name in optional_cols.items():
            if has(lower_name):
                orig_col = cols_lower[lower_name]
                out[target_name] = pd.to_numeric(df[orig_col], errors="coerce")
        
        # Reorder columns
        base_cols = ["id", "name", "team", "position", "salary", "fpts", "ownership"]
        extra_cols = [c for c in out.columns if c not in base_cols]
        out = out[base_cols + extra_cols]
        
        # Filter positions
        out = out[out["position"].isin(VALID_POSITIONS)].dropna(subset=["salary", "fpts"]).reset_index(drop=True)
        return out, "Detected new projections format (Name/Pos/Team/Salary/Proj)."

    # Case 2: Already in optimizer schema
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
        out = out[out["position"].str.upper().isin(VALID_POSITIONS)].dropna(subset=["salary", "fpts"]).reset_index(drop=True)
        out["position"] = out["position"].str.upper()
        return out, "Detected optimizer schema with fpts/proj."

    # Case 3: Raw stats schema: we compute fpts per the provided scoring
    required_like = ["id", "name", "team", "position", "salary"]
    if all(any(c.lower() == r.lower() for c in original_cols) for r in required_like):
        out = df.copy()
        # Normalize names
        rename_map = {c: c.lower() for c in out.columns}
        out.rename(columns=rename_map, inplace=True)
        out["position"] = out["position"].str.upper()
        out["salary"] = pd.to_numeric(out["salary"], errors="coerce")

        # Compute fpts per row based on position
        def compute_row(row: pd.Series) -> float:
            pos = str(row.get("position", "")).upper()
            if pos in ["DEF", "DST"]:
                return compute_defense_fpts(row)
            return compute_offense_fpts(row)

        out["fpts"] = out.apply(compute_row, axis=1)
        if "ownership" in out.columns:
            out["ownership"] = pd.to_numeric(out["ownership"], errors="coerce").fillna(0.0)
        else:
            out["ownership"] = 0.0

        out = out[out["position"].isin(VALID_POSITIONS)].dropna(subset=["salary"]).reset_index(drop=True)
        return out, "Computed fpts from raw stats per NFL scoring."

    raise ValueError(f"Unrecognized schema. Columns found: {original_cols}")


def build_lineup_ilp(
    players: pd.DataFrame,
    budget: float,
    lineup_size: int,
    min_per_pos: Dict[str, int],
    max_per_pos: Dict[str, int],
    max_same_team: int,
    *,
    score_col: str = "fpts",
    ownership_weight: float = 0.0,
    eps_budget_tiebreak: float = 1e-6,
    banned_player_ids: set | None = None,
    enforce_stacking_rules: bool = False,
    opponent_map: Dict[str, str] | None = None,
) -> Tuple[List[int], float, float, float]:
    if banned_player_ids is None:
        banned_player_ids = set()
    if opponent_map is None:
        opponent_map = {}

    idxs = list(range(len(players)))
    prob = pl.LpProblem("nfl_multi_optimizer", pl.LpMaximize)
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
        pos_idx = [i for i in idxs if str(players.iloc[i]["position"]).upper() == pos]
        if pos_idx:
            prob += pl.lpSum([x[i] for i in pos_idx]) >= min_per_pos.get(pos, 0)
            prob += pl.lpSum([x[i] for i in pos_idx]) <= max_per_pos.get(pos, lineup_size)
        else:
            if min_per_pos.get(pos, 0) > 0:
                return [], 0.0, 0.0, 0.0

    # Team cap: up to max_same_team players per team
    for team, team_group in players.groupby("team"):
        team_idx = [int(i) for i in team_group.index]
        prob += pl.lpSum([x[i] for i in team_idx]) <= max_same_team

    # STACKING RULES
    if enforce_stacking_rules:
        # Rule 1: No defense against any player in lineup
        # If DST from team A is selected, no offensive players from opponent of A
        dst_indices = [i for i in idxs if str(players.iloc[i]["position"]).upper() in ["DST", "DEF"]]
        for dst_idx in dst_indices:
            dst_team = str(players.iloc[dst_idx]["team"]).upper()
            opponent_team = opponent_map.get(dst_team, "").upper()
            if opponent_team:
                # Find all offensive players from opponent team
                opp_offense = [i for i in idxs 
                             if str(players.iloc[i]["team"]).upper() == opponent_team 
                             and str(players.iloc[i]["position"]).upper() not in ["DST", "DEF"]]
                if opp_offense:
                    # If DST selected, sum of opponent offense must be 0
                    prob += pl.lpSum([x[i] for i in opp_offense]) <= len(opp_offense) * (1 - x[dst_idx])

        # Rule 2: No more than 2 pass catchers (WR/TE) with same QB
        qb_indices = [i for i in idxs if str(players.iloc[i]["position"]).upper() == "QB"]
        for qb_idx in qb_indices:
            qb_team = str(players.iloc[qb_idx]["team"]).upper()
            # Find WR/TE from same team
            pass_catchers = [i for i in idxs 
                           if str(players.iloc[i]["team"]).upper() == qb_team
                           and str(players.iloc[i]["position"]).upper() in ["WR", "TE"]]
            if pass_catchers:
                # If QB selected, max 2 pass catchers from same team
                # We need: sum(pass_catchers) <= 2 when x[qb_idx] = 1
                # And: sum(pass_catchers) can be anything when x[qb_idx] = 0
                # This translates to: sum(pass_catchers) <= 2 + M*(1 - x[qb_idx])
                # where M is large enough (lineup_size works)
                prob += pl.lpSum([x[i] for i in pass_catchers]) <= 2 + lineup_size * (1 - x[qb_idx])

        # Rule 3: Only 1 RB per team
        for team, team_group in players.groupby("team"):
            rb_idx = [int(i) for i in team_group.index 
                     if str(players.iloc[i]["position"]).upper() == "RB"]
            if rb_idx:
                prob += pl.lpSum([x[i] for i in rb_idx]) <= 1

        # Rule 4: Only 1 pass catcher without the QB (orphan pass catcher limit)
        # For each team: if QB not selected, at most 1 WR/TE from that team
        for team, team_group in players.groupby("team"):
            team_qbs = [int(i) for i in team_group.index 
                       if str(players.iloc[i]["position"]).upper() == "QB"]
            team_catchers = [int(i) for i in team_group.index 
                           if str(players.iloc[i]["position"]).upper() in ["WR", "TE"]]
            
            if team_catchers:
                if team_qbs:
                    # If any QB from this team is selected, no limit on catchers (handled by rule 2)
                    # If no QB selected, at most 1 catcher
                    qb_selected = pl.lpSum([x[i] for i in team_qbs])
                    catchers_selected = pl.lpSum([x[i] for i in team_catchers])
                    # catchers_selected <= 1 + M * qb_selected
                    # When qb_selected = 0: catchers <= 1
                    # When qb_selected >= 1: catchers <= 1 + M (effectively no limit, rule 2 applies)
                    prob += catchers_selected <= 1 + lineup_size * qb_selected
                else:
                    # No QB available, limit catchers to 1
                    prob += pl.lpSum([x[i] for i in team_catchers]) <= 1

    total_score = pl.lpSum([players.iloc[i][score_col] * x[i] for i in idxs])
    total_salary = pl.lpSum([players.iloc[i]["salary"] * x[i] for i in idxs])
    leftover = budget - total_salary

    if "ownership" in players.columns and ownership_weight != 0.0:
        total_own = pl.lpSum([players.iloc[i]["ownership"] * x[i] for i in idxs])
        objective = total_score - ownership_weight * total_own + eps_budget_tiebreak * leftover
    else:
        objective = total_score + eps_budget_tiebreak * leftover

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
    *,
    score_col: str = "fpts",
    max_exposure_by_pos: Dict[str, float] | None = None,
    enforce_stacking_rules: bool = False,
    opponent_map: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    df = base_df.copy().reset_index(drop=True)
    if "ownership" not in df.columns:
        df["ownership"] = 0.0

    exposure_counts: Dict[Any, int] = {}
    max_count = max(1, int(np.floor(max_exposure_pct * num_lineups)))
    
    # Per-position exposure limits
    if max_exposure_by_pos is None:
        max_exposure_by_pos = {}
    pos_exposure_counts: Dict[Tuple[Any, str], int] = {}
    
    results: List[Dict[str, Any]] = []

    for k in range(num_lineups):
        banned = {pid for pid, cnt in exposure_counts.items() if cnt >= max_count}
        
        # Add position-specific exposure bans
        for (pid, pos), cnt in pos_exposure_counts.items():
            if pos in max_exposure_by_pos:
                pos_max = max(1, int(np.floor(max_exposure_by_pos[pos] * num_lineups)))
                if cnt >= pos_max:
                    banned.add(pid)

        sel, f, s, o = build_lineup_ilp(
            df,
            budget,
            lineup_size,
            min_per_pos,
            max_per_pos,
            max_same_team,
            score_col=score_col,
            ownership_weight=ownership_weight,
            eps_budget_tiebreak=1e-6,
            banned_player_ids=banned,
            enforce_stacking_rules=enforce_stacking_rules,
            opponent_map=opponent_map,
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
                    score_col=score_col,
                    ownership_weight=ownership_weight,
                    eps_budget_tiebreak=1e-6,
                    banned_player_ids=banned,
                    enforce_stacking_rules=enforce_stacking_rules,
                    opponent_map=opponent_map,
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

        for i in sel:
            pid = df.iloc[i]["id"]
            pos = str(df.iloc[i]["position"]).upper()
            exposure_counts[pid] = exposure_counts.get(pid, 0) + 1
            pos_exposure_counts[(pid, pos)] = pos_exposure_counts.get((pid, pos), 0) + 1

    return results


def layout_optimizer_tab():
    st.header("NFL Multi-Game Optimizer")
    st.caption("Upload projections or raw stats. Supports new format (Name/Pos/Team/Salary/Proj), optimizer schema, or raw stats.")

    with st.sidebar:
        st.subheader("Settings")
        budget = st.number_input("Budget (M)", value=120.0, min_value=1.0, step=1.0)
        lineup_size = st.number_input("Squad size", value=9, min_value=8, max_value=11, step=1)

        min_per_pos = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DEF": 1, "DST": 1}
        max_per_pos = {"QB": 1, "RB": 3, "WR": 4, "TE": 2, "DEF": 1, "DST": 1}
        st.write("Position limits:")
        st.write("QB 1/1, RB 2/3, WR 3/4, TE 1/2, DEF/DST 1/1")

        max_same_team = st.number_input("Max players per team", value=3, min_value=1, max_value=6, step=1)

        st.markdown("---")
        st.markdown("**Stacking Rules**")
        enforce_stacking = st.checkbox("Enable stacking rules", value=False, help="Enforce correlation rules for better lineup construction")
        if enforce_stacking:
            st.info(
                "Active rules:\n"
                "• No defense vs offensive players\n"
                "• Max 2 pass catchers with QB\n"
                "• Max 1 RB per team\n"
                "• Max 1 orphan pass catcher (without QB)"
            )

        st.markdown("---")
        num_lineups = st.number_input("Number of lineups", value=10, min_value=1, max_value=500, step=1)
        min_uniques = st.number_input("Minimum uniques between lineups", value=2, min_value=0, max_value=11, step=1)
        max_exposure_pct = st.slider("Max exposure per player (%)", min_value=10, max_value=100, value=60, step=5) / 100.0
        
        st.markdown("**Position-specific exposure limits**")
        st.caption("Leave at 100% to use overall exposure limit only")
        pos_exp_qb = st.slider("QB max exposure (%)", min_value=10, max_value=100, value=100, step=5) / 100.0
        pos_exp_rb = st.slider("RB max exposure (%)", min_value=10, max_value=100, value=100, step=5) / 100.0
        pos_exp_wr = st.slider("WR max exposure (%)", min_value=10, max_value=100, value=100, step=5) / 100.0
        pos_exp_te = st.slider("TE max exposure (%)", min_value=10, max_value=100, value=100, step=5) / 100.0
        pos_exp_dst = st.slider("DST max exposure (%)", min_value=10, max_value=100, value=100, step=5) / 100.0
        
        ownership_weight = st.slider("Ownership weight (fpts - weight*ownership)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

        st.markdown("**Scoring Mode**")
        score_mode = st.selectbox("Score to optimize", ["Proj (fpts)", "Form", "LastPoints", "TotalPoints", "Weighted Score"], index=0)
        
        st.markdown("Feature weights (used if Weighted Score)")
        w_value = st.slider("Weight: Value", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_form = st.slider("Weight: Form", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_lastpoints = st.slider("Weight: LastPoints", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_totalpoints = st.slider("Weight: TotalPoints", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

        st.caption("Tie-breaker: if fpts tie, prefers higher remaining budget.")

    sample_btn = st.button("Download CSV template")
    if sample_btn:
        sample_csv = """Name,Pos,Team,Salary,Proj,Value,lineup,form,lastPoints,totalPoints,Ids
Lamar Jackson,QB,BAL,19.8,24.3,1.23,possible,23.3,10.7,93.4,4352569
Derrick Henry,RB,BAL,18.2,20.5,1.13,possible,22.1,15.3,87.2,4353210
CeeDee Lamb,WR,DAL,17.5,19.8,1.13,possible,21.4,12.8,92.1,4354123
"""
        sample = pd.read_csv(io.StringIO(sample_csv))
        st.download_button(
            label="nfl_template.csv",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="nfl_template.csv",
            mime="text/csv",
        )
    
    opp_sample_btn = st.button("Download opponent mapping template")
    if opp_sample_btn:
        opp_sample_csv = """Team,Opponent
BAL,CIN
CIN,BAL
KC,LV
LV,KC
DAL,PHI
PHI,DAL
"""
        opp_sample = pd.read_csv(io.StringIO(opp_sample_csv))
        st.download_button(
            label="opponent_template.csv",
            data=opp_sample.to_csv(index=False).encode("utf-8"),
            file_name="opponent_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Choose projections or stats CSV", type=["csv"])
    
    # Optional opponent mapping upload
    opponent_file = None
    opponent_map = {}
    if enforce_stacking:
        st.markdown("**Opponent Mapping (Optional)**")
        st.caption("Upload a CSV with 'Team' and 'Opponent' columns to enable defense vs offense rule")
        opponent_file = st.file_uploader("Choose opponent mapping CSV (optional)", type=["csv"], key="opponent_upload")
        
        if opponent_file:
            try:
                opp_df = pd.read_csv(opponent_file)
                opp_df.columns = [c.strip().lower() for c in opp_df.columns]
                if 'team' in opp_df.columns and 'opponent' in opp_df.columns:
                    opponent_map = dict(zip(
                        opp_df['team'].str.strip().str.upper(),
                        opp_df['opponent'].str.strip().str.upper()
                    ))
                    st.success(f"Loaded opponent mapping for {len(opponent_map)} teams")
                else:
                    st.warning("Opponent CSV must have 'Team' and 'Opponent' columns")
            except Exception as e:
                st.error(f"Failed to parse opponent CSV: {e}")
    
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
            # Prepare score column
            working = df.copy()
            score_col = "fpts"
            
            # Build position exposure dict
            max_exposure_by_pos = {}
            if pos_exp_qb < 1.0:
                max_exposure_by_pos["QB"] = pos_exp_qb
            if pos_exp_rb < 1.0:
                max_exposure_by_pos["RB"] = pos_exp_rb
            if pos_exp_wr < 1.0:
                max_exposure_by_pos["WR"] = pos_exp_wr
            if pos_exp_te < 1.0:
                max_exposure_by_pos["TE"] = pos_exp_te
            if pos_exp_dst < 1.0:
                max_exposure_by_pos["DST"] = pos_exp_dst
                max_exposure_by_pos["DEF"] = pos_exp_dst  # Handle both DST/DEF
            
            if score_mode == "Form":
                if "form" in working.columns:
                    score_col = "form"
                else:
                    st.warning("Form column not found, using fpts instead")
            elif score_mode == "LastPoints":
                if "lastpoints" in working.columns:
                    score_col = "lastpoints"
                else:
                    st.warning("LastPoints column not found, using fpts instead")
            elif score_mode == "TotalPoints":
                if "totalpoints" in working.columns:
                    score_col = "totalpoints"
                else:
                    st.warning("TotalPoints column not found, using fpts instead")
            elif score_mode.startswith("Weighted"):
                # Build weighted score from optional columns
                feats = []
                weights = []
                
                for col_name, w in [("value", w_value), ("form", w_form), ("lastpoints", w_lastpoints), ("totalpoints", w_totalpoints)]:
                    if w <= 0:
                        continue
                    if col_name not in working.columns:
                        continue
                    series = pd.to_numeric(working[col_name], errors="coerce")
                    std = series.std(skipna=True)
                    if std and std > 0:
                        feat = (series - series.mean(skipna=True)) / std
                    else:
                        feat = series.fillna(series.mean(skipna=True))
                    feats.append(feat)
                    weights.append(w)

                if feats:
                    wsum = float(sum(weights))
                    mat = pd.concat(feats, axis=1).fillna(0.0)
                    wnorm = np.array([w / wsum for w in weights])
                    working["weighted_score"] = (mat @ wnorm) + working["fpts"]
                    score_col = "weighted_score"

            results = generate_lineups(
                working,
                num_lineups=int(num_lineups),
                budget=float(budget),
                lineup_size=int(lineup_size),
                min_uniques=int(min_uniques),
                max_exposure_pct=float(max_exposure_pct),
                ownership_weight=float(ownership_weight),
                min_per_pos=min_per_pos,
                max_per_pos=max_per_pos,
                max_same_team=int(max_same_team),
                score_col=score_col,
                max_exposure_by_pos=max_exposure_by_pos,
                enforce_stacking_rules=enforce_stacking,
                opponent_map=opponent_map,
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
        st.download_button("Download lineups CSV", csv_bytes, file_name="nfl_lineups.csv", mime="text/csv")


def main():
    st.title("Fanteam NFL Optimizer - Multi-Game")
    tab1, = st.tabs(["Optimizer"])
    with tab1:
        layout_optimizer_tab()

    st.markdown("---")
    st.markdown(
        "- Team cap: up to 3 from the same team.\n"
        "- Position limits: QB 1/1, RB 2/3, WR 3/4, TE 1/2, DEF/DST 1/1.\n"
        "- Budget tie-breaker: if fpts tie, prefers higher remaining budget.\n"
        "- **Stacking rules** (when enabled):\n"
        "  - No defense against offensive players from opponent\n"
        "  - Max 2 pass catchers (WR/TE) stacked with same QB\n"
        "  - Max 1 RB per team\n"
        "  - Max 1 orphan pass catcher per team (without QB)\n"
    )


if __name__ == "__main__":
    main()