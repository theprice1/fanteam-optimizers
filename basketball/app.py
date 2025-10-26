import io
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pulp as pl
import streamlit as st

VALID_POSITIONS = ["PG", "SG", "SF", "PF", "C"]


def normalize_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Try to coerce an incoming projections CSV into the optimizer schema:
    id, name, team, position, salary, fpts, ownership(optional)
    Returns (normalized_df, message). Raises on fatal schema issues.
    """
    original_cols = [c.strip() for c in df.columns]

    # Lower for detection but keep original for extraction
    cols_lower = {c.lower(): c for c in original_cols}

    def has(col: str) -> bool:
        return col in cols_lower

    # Case 1: Already in optimizer schema
    if all(has(c) for c in ["id", "name", "team", "position", "salary", "fpts"]):
        out = df.copy()
        out.columns = [c.lower() for c in out.columns]
        # Types
        out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
        out["fpts"] = pd.to_numeric(out["fpts"], errors="coerce")
        if "ownership" not in out.columns:
            out["ownership"] = 0.0
        # Filter positions
        out = out[out["position"].isin(VALID_POSITIONS)].dropna(subset=["salary", "fpts"]).reset_index(drop=True)
        return out, "Detected optimizer schema."

    # Case 2: New schema format (Name, Pos, Team, Salary, Proj, ...)
    # Expected headers: Name, Pos, Team, Salary, Proj, Value, FT-Pref, Own%, Form, Floor, Ceil, Std Dev, Boom, Status, Ids
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
            "position": df[pos_col].astype(str).str.strip(),
            "salary": pd.to_numeric(df[salary_col], errors="coerce"),
            "fpts": pd.to_numeric(df[proj_col], errors="coerce"),
        })
        
        # Handle ID - prefer Ids column, fallback to index
        if has("ids"):
            ids_col = cols_lower["ids"]
            out["id"] = df[ids_col]
        else:
            out["id"] = range(len(out))
        
        # Handle ownership - prefer Own% column
        if has("own%"):
            own_col = cols_lower["own%"]
            out["ownership"] = pd.to_numeric(df[own_col].astype(str).str.rstrip('%'), errors="coerce").fillna(0.0)
        else:
            out["ownership"] = 0.0
        
        # Preserve optional columns for weighted scoring
        optional_cols = {
            "value": "value",
            "ft-pref": "ft_pref",
            "form": "form",
            "floor": "floor",
            "ceil": "ceiling",  # Map to "ceiling" for consistency
            "std dev": "std_dev",
            "boom": "boom",
            "status": "status"
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

    # Case 3: Original projections CSV format
    # Expected headers like: Id, Name, Pos, TEAM, ... Proj_adj, Salary, FP ...
    # Prefer Proj_adj for fpts; fallback to FP
    required_like = ["Id", "Name", "Pos", "TEAM", "Salary"]
    if all(any(c.lower() == r.lower() for c in original_cols) for r in required_like):
        id_col = next(c for c in original_cols if c.lower() == "id")
        name_col = next(c for c in original_cols if c.lower() == "name")
        pos_col = next(c for c in original_cols if c.lower() == "pos")
        team_col = next(c for c in original_cols if c.lower() == "team")
        sal_col = next(c for c in original_cols if c.lower() == "salary")
        fpts_col = None
        if any(c.lower() == "proj_adj" for c in original_cols):
            fpts_col = next(c for c in original_cols if c.lower() == "proj_adj")
        elif any(c.lower() == "fp" for c in original_cols):
            fpts_col = next(c for c in original_cols if c.lower() == "fp")

        if fpts_col is None:
            raise ValueError("Could not find a projections column (Proj_adj or FP).")

        out = pd.DataFrame({
            "id": df[id_col],
            "name": df[name_col].astype(str).str.strip(),
            "team": df[team_col].astype(str).str.strip(),
            "position": df[pos_col].astype(str).str.strip(),
            "salary": pd.to_numeric(df[sal_col], errors="coerce"),
            "fpts": pd.to_numeric(df[fpts_col], errors="coerce"),
        })
        if "ownership" in df.columns:
            out["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(0.0)
        else:
            out["ownership"] = 0.0

        out = out[out["position"].isin(VALID_POSITIONS)].dropna(subset=["salary", "fpts"]).reset_index(drop=True)
        return out, "Detected projections CSV (Id/Name/Pos/TEAM/Salary + Proj_adj/FP)."

    raise ValueError(f"Unrecognized schema. Columns found: {original_cols}")


def build_lineup_ilp(
    players: pd.DataFrame,
    budget: float,
    lineup_size: int,
    min_per_pos: Dict[str, int],
    max_per_pos: Dict[str, int],
    *,
    score_col: str = "fpts",
    ownership_weight: float = 0.0,
    eps_budget_tiebreak: float = 1e-6,
    banned_player_ids: set | None = None,
    team_min_stack: Dict[str, int] | None = None,
    max_per_team: int | None = None,
    auto_stack_bonus: float = 0.0,
) -> Tuple[List[int], float, float, float]:
    """
    Returns selected row indices and totals (fpts, salary, ownership).
    """
    if banned_player_ids is None:
        banned_player_ids = set()

    idxs = list(range(len(players)))
    prob = pl.LpProblem("nba_optimizer", pl.LpMaximize)
    x = pl.LpVariable.dicts("x", idxs, lowBound=0, upBound=1, cat=pl.LpBinary)

    # Bans
    for i in idxs:
        if players.iloc[i]["id"] in banned_player_ids:
            prob += x[i] == 0

    # Core constraints
    prob += pl.lpSum([x[i] for i in idxs]) == lineup_size
    prob += pl.lpSum([players.iloc[i]["salary"] * x[i] for i in idxs]) <= budget

    # Position constraints
    for pos in VALID_POSITIONS:
        pos_idx = [i for i in idxs if players.iloc[i]["position"] == pos]
        if pos_idx:
            prob += pl.lpSum([x[i] for i in pos_idx]) >= min_per_pos.get(pos, 0)
            prob += pl.lpSum([x[i] for i in pos_idx]) <= max_per_pos.get(pos, lineup_size)
        else:
            # Infeasible if min > 0 and no eligible players
            if min_per_pos.get(pos, 0) > 0:
                return [], 0.0, 0.0, 0.0

    # Team stacking constraints
    if team_min_stack:
        for team, min_cnt in team_min_stack.items():
            if min_cnt and min_cnt > 0:
                team_idx = [i for i in idxs if str(players.iloc[i]["team"]).upper() == str(team).upper()]
                if team_idx:
                    prob += pl.lpSum([x[i] for i in team_idx]) >= int(min_cnt)
    # Max per team
    if max_per_team is not None and max_per_team > 0:
        for team, grp in players.groupby("team"):
            team_idx = [int(i) for i in grp.index]
            prob += pl.lpSum([x[i] for i in team_idx]) <= int(max_per_team)

    total_score = pl.lpSum([players.iloc[i][score_col] * x[i] for i in idxs])

    # Automatic stacking bonus: add pairwise synergy variables for same-team pairs
    if auto_stack_bonus and auto_stack_bonus > 0:
        # Build pairs per team
        for team, grp in players.groupby("team"):
            team_indices = [int(i) for i in grp.index]
            n = len(team_indices)
            if n < 2:
                continue
            # Create pairwise vars y_{i,j} that are 1 if both players i and j are selected
            for a in range(n):
                i = team_indices[a]
                for b in range(a + 1, n):
                    j = team_indices[b]
                    y = pl.LpVariable(f"y_pair_{team}_{i}_{j}", lowBound=0, upBound=1, cat=pl.LpBinary)
                    # Linking constraints
                    prob += y <= x[i]
                    prob += y <= x[j]
                    prob += y >= x[i] + x[j] - 1
                    total_score += auto_stack_bonus * y
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
    *,
    score_col: str = "fpts",
    team_min_stack: Dict[str, int] | None = None,
    max_per_team: int | None = None,
    auto_stack_bonus: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Sequentially generate lineups with exposure caps and uniqueness.
    """
    df = base_df.copy().reset_index(drop=True)
    if "ownership" not in df.columns:
        df["ownership"] = 0.0

    exposure_counts: Dict[Any, int] = {}
    max_count = max(1, int(np.floor(max_exposure_pct * num_lineups)))
    results: List[Dict[str, Any]] = []

    for k in range(num_lineups):
        banned = {pid for pid, cnt in exposure_counts.items() if cnt >= max_count}

        sel, f, s, o = build_lineup_ilp(
            df, budget, lineup_size, min_per_pos, max_per_pos,
            score_col=score_col,
            ownership_weight=ownership_weight, eps_budget_tiebreak=1e-6,
            banned_player_ids=banned,
            team_min_stack=team_min_stack,
            max_per_team=max_per_team,
            auto_stack_bonus=auto_stack_bonus,
        )
        if not sel:
            break

        ids = [df.iloc[i]["id"] for i in sel]

        # Uniqueness: ensure at least `min_uniques` different players vs any prior lineup
        if results and min_uniques > 0:
            tries = 0
            while tries < 30:
                worst_overlap = 0
                for prev in results:
                    overlap = sum(1 for pid in ids if pid in set(prev["player_ids"]))
                    worst_overlap = max(worst_overlap, overlap)
                if (lineup_size - worst_overlap) >= min_uniques:
                    break

                # Ban one overlapping player (highest salary among overlaps) and re-solve
                overlap_all = set().union(*[set(prev["player_ids"]) for prev in results])
                overlap_curr = [pid for pid in ids if pid in overlap_all]
                if not overlap_curr:
                    break
                ban_pid = df.set_index("id").loc[overlap_curr].sort_values("salary", ascending=False).index[0]
                banned.add(ban_pid)
                sel, f, s, o = build_lineup_ilp(
                    df, budget, lineup_size, min_per_pos, max_per_pos,
                    score_col=score_col,
                    ownership_weight=ownership_weight, eps_budget_tiebreak=1e-6,
                    banned_player_ids=banned,
                    team_min_stack=team_min_stack,
                    max_per_team=max_per_team,
                    auto_stack_bonus=auto_stack_bonus,
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

        # Update exposure
        for pid in ids:
            exposure_counts[pid] = exposure_counts.get(pid, 0) + 1

    return results


def layout_optimizer_tab():
    st.header("Optimizer")
    st.caption("Upload a projections CSV. Accepted formats: optimizer schema, new format (Name/Pos/Team/Salary/Proj), or original (Id/Name/Pos/TEAM/Salary + Proj_adj/FP).")

    # Controls
    with st.sidebar:
        st.subheader("Settings")
        budget = st.number_input("Budget", value=73.0, min_value=1.0, step=0.5)
        lineup_size = 7
        st.write(f"Lineup size: {lineup_size}")

        # Position rules
        min_per_pos = {p: 1 for p in VALID_POSITIONS}
        max_per_pos = {p: 3 for p in VALID_POSITIONS}

        num_lineups = st.number_input("Number of lineups", value=10, min_value=1, max_value=500, step=1)
        min_uniques = st.number_input("Minimum uniques between lineups", value=2, min_value=0, max_value=lineup_size, step=1)
        max_exposure_pct = st.slider("Max exposure per player (%)", min_value=10, max_value=100, value=60, step=5) / 100.0
        ownership_weight = st.slider("Ownership weight (fpts - weight*ownership)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

        st.markdown("**Tournament settings**")
        score_mode = st.selectbox("Score to optimize", ["Mean (fpts)", "Ceiling", "Floor", "Boom", "Weighted Tournament Score"], index=0)
        ceiling_multiplier = st.number_input("Ceiling multiplier (if no ceiling column)", value=1.15, min_value=1.0, max_value=2.0, step=0.01)

        st.markdown("Feature weights (used if Weighted Tournament Score)")
        w_value = st.slider("Weight: Value", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_form = st.slider("Weight: Form", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_boom = st.slider("Weight: Boom", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
        w_floor = st.slider("Weight: Floor", min_value=0.0, max_value=2.0, value=0.0, step=0.05)

        st.markdown("Auto-stacking")
        auto_stack_bonus = st.number_input("Bonus per same-team pair (additive)", value=0.0, min_value=0.0, max_value=5.0, step=0.05)

        st.markdown("**Team stacking**")
        team_a = st.text_input("Team A code for stack (optional)")
        team_a_min = st.number_input("Min from Team A", value=0, min_value=0, max_value=lineup_size, step=1)
        team_b = st.text_input("Team B code for stack (optional)")
        team_b_min = st.number_input("Min from Team B", value=0, min_value=0, max_value=lineup_size, step=1)
        max_from_any_team = st.number_input("Max from any single team (0 = no cap)", value=0, min_value=0, max_value=lineup_size, step=1)

        st.caption("Tie-breaker: if fpts tie, prefers higher remaining budget.")

    # Template download
    sample_btn = st.button("Download CSV template")
    if sample_btn:
        sample_csv = """Name,Pos,Team,Salary,Proj,Value,FT-Pref,Own%,Form,Floor,Ceil,Std Dev,Boom,Status,Ids
Nikola Jokic,C,DEN,19.1,61.4,3.22,6.6,40.70%,59.3,43.1,79.2,14.2,13.40%,expected,4430998
Luka Doncic,PG,DAL,18.5,58.2,3.15,7.2,35.50%,56.8,41.5,75.8,13.8,12.80%,expected,4431002
Anthony Davis,PF,LAL,17.2,54.6,3.17,5.8,28.30%,52.4,38.2,71.4,13.1,11.90%,expected,4431005
"""
        sample = pd.read_csv(io.StringIO(sample_csv))
        st.download_button(
            label="projections_template.csv",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="projections_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Choose projections CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        return

    try:
        raw = pd.read_csv(uploaded)
        df, msg = normalize_schema(raw)
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
            
            if score_mode == "Ceiling":
                if "ceiling" in working.columns:
                    score_col = "ceiling"
                else:
                    working["ceil_calc"] = working["fpts"] * float(ceiling_multiplier)
                    score_col = "ceil_calc"
            elif score_mode == "Floor":
                if "floor" in working.columns:
                    score_col = "floor"
                else:
                    st.warning("Floor column not found, using fpts instead")
            elif score_mode == "Boom":
                if "boom" in working.columns:
                    score_col = "boom"
                else:
                    st.warning("Boom column not found, using fpts instead")
            elif score_mode.startswith("Weighted"):
                # Build weighted score from optional columns
                feats = []
                weights = []
                
                for col_name, w in [("value", w_value), ("form", w_form), ("boom", w_boom), ("floor", w_floor)]:
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
                    working["tourny_weighted"] = (mat @ wnorm) + working["fpts"]
                    score_col = "tourny_weighted"

            # Team stacking map
            team_min_stack = {}
            if team_a and int(team_a_min) > 0:
                team_min_stack[team_a.strip().upper()] = int(team_a_min)
            if team_b and int(team_b_min) > 0:
                team_min_stack[team_b.strip().upper()] = int(team_b_min)
            if not team_min_stack:
                team_min_stack = None

            max_per_team = int(max_from_any_team) if int(max_from_any_team) > 0 else None

            results = generate_lineups(
                working,
                num_lineups=int(num_lineups),
                budget=float(budget),
                lineup_size=lineup_size,
                min_uniques=int(min_uniques),
                max_exposure_pct=float(max_exposure_pct),
                ownership_weight=float(ownership_weight),
                min_per_pos=min_per_pos,
                max_per_pos=max_per_pos,
                score_col=score_col,
                team_min_stack=team_min_stack,
                max_per_team=max_per_team,
                auto_stack_bonus=float(auto_stack_bonus),
            )
        except Exception as e:
            st.error(f"Failed to generate lineups: {e}")
            return

        if not results:
            st.error("No feasible lineup found. Check data, budget, and position availability.")
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
        st.download_button("Download lineups CSV", csv_bytes, file_name="lineups.csv", mime="text/csv")


def layout_prepare_tab():
    st.header("Prepare CSV")
    st.caption("Convert your projections CSV into optimizer schema.")

    uploaded = st.file_uploader("Choose your projections CSV", type=["csv"], key="prepare_uploader")
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        return

    try:
        raw = pd.read_csv(uploaded)
        df, msg = normalize_schema(raw)
        st.success(f"Parsed: {msg}")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return

    with st.expander("Converted preview"):
        st.dataframe(df.head(50))

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download projections_for_app.csv", csv_bytes, file_name="projections_for_app.csv", mime="text/csv")

    st.markdown("- Valid positions required: PG, SG, SF, PF, C")
    st.markdown("- Budget unit must match salary unit (e.g., 73 if salaries are like 12.5).")


def main():
    st.title("Fantasy Basketball Optimizer (7-player)")
    tab1, tab2 = st.tabs(["Optimizer", "Prepare CSV"])
    with tab1:
        layout_optimizer_tab()
    with tab2:
        layout_prepare_tab()

    st.markdown("---")
    st.markdown(
        "- Position limits: 1–3 for each of PG, SG, SF, PF, C; total players = 7.\n"
        "- Budget tie-breaker: if fpts tie, the solver prefers higher remaining budget.\n"
        "- Ownership (optional): add a column 'ownership' or 'Own%' (0–100 scale) to fade chalk via weight.\n"
        "- Multi-lineups: sequential solve with min-uniques and exposure caps."
    )


if __name__ == "__main__":
    main()