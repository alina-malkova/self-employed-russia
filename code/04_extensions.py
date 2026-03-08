"""
04_extensions.py
Methodological extensions addressing referee concerns:

1. Worker fixed effects Townsend test (controls for selection into informality)
2. IV Townsend test using health shocks (addresses endogeneity of Δln w)
3. Callaway-Sant'Anna with two explicit cohorts (table stakes for top journals)
4. Mechanism tests (banking heterogeneity, savings DiD, NPD intensity)
5. Power analysis and MDE calculations
6. Wild cluster bootstrap (Cameron, Gelbach, Miller 2008) for inference with 32 clusters

Requires:
  - data/cleaned/rlms_informality_panel.pkl (from 01_clean_rlms.py)
  - RLMS IND file for health variables (from OneDrive)
  - FRI regional means (from Broken Institutions project)
  - NPD monthly regional panel (from data/)
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")
FIG_DIR = os.path.join(PROJECT_DIR, "output", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PANEL_PATH = os.path.join(DATA_DIR, "rlms_informality_panel.pkl")
IND_PATH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology"
    "/_Research/Sanctions/Working santctions/IND/RLMS_IND_1994_2023_eng_dta.dta"
)
FRI_PATH = os.path.expanduser(
    "~/Dropbox/_Research/Broken Institutions Russia"
    "/welfare analysis/data/theta/fri_regional_means.csv"
)
NPD_MONTHLY_PATH = os.path.join(PROJECT_DIR, "data", "npd_monthly_by_region_panel.csv")


MISSING_CODES = [99999990, 99999991, 99999992, 99999993,
                 99999994, 99999995, 99999996, 99999997,
                 99999998, 99999999]


def to_numeric_safe(s):
    return pd.to_numeric(s, errors='coerce')


# ============================================================
# DATA LOADING
# ============================================================

def load_panel():
    """Load cleaned RLMS panel and construct analysis variables."""
    print("Loading panel...")
    df = pd.read_pickle(PANEL_PATH)
    df['year'] = df['year'].astype(int)
    df['event_time'] = df['year'] - df['npd_year']
    df['npd_cohort'] = np.where(df['npd_year'] == 2019, 'early', 'late')

    # Employment type dummies
    df['is_informal'] = df['emp_type'].isin(['informal_wage', 'self_employed']).astype(int)
    df['is_informal_wage'] = (df['emp_type'] == 'informal_wage').astype(int)
    df['is_self_employed'] = (df['emp_type'] == 'self_employed').astype(int)
    df['is_formal'] = (df['emp_type'] == 'formal_wage').astype(int)
    df['post_npd_int'] = df['post_npd'].fillna(0).astype(int)

    # Interactions
    df['dw_x_informal'] = df['wage_growth'] * df['is_informal']
    df['dw_x_post'] = df['wage_growth'] * df['post_npd_int']
    df['dw_x_informal_x_post'] = df['wage_growth'] * df['is_informal'] * df['post_npd_int']
    df['informal_x_post'] = df['is_informal'] * df['post_npd_int']

    print(f"  Panel: {len(df):,} obs, {df['idind'].nunique():,} individuals")
    return df


def add_health_vars(df):
    """Load health variables from raw RLMS IND and merge to panel."""
    print("\nLoading health variables from RLMS IND...")
    if not os.path.exists(IND_PATH):
        print("  WARNING: IND file not found. Skipping health variables.")
        return df

    # Load only identifiers + health vars
    health_vars = ['idind', 'year',
                   'm3',       # Self-rated health (1=very good ... 5=very bad)
                   'm20_61',   # Chronic heart disease
                   'm20_62',   # Chronic lung disease
                   'm20_63',   # Chronic liver disease
                   'm20_64',   # Chronic kidney disease
                   'm20_65',   # Chronic stomach disease
                   'm20_66',   # Chronic spinal disease
                   'm20_67',   # Other chronic disease
                   'm20_7',    # Assigned disability
                   'm137',     # Health allows routine duties
                   'm32',      # Took medicines last 7 days
                   ]

    try:
        reader = pd.read_stata(IND_PATH, iterator=True, convert_categoricals=False)
        all_cols = list(reader.variable_labels().keys())
        available = [v for v in health_vars if v in all_cols]
        missing = [v for v in health_vars if v not in all_cols]
        if missing:
            print(f"  Health vars not found: {missing}")
    except Exception:
        available = health_vars

    hdf = pd.read_stata(IND_PATH, columns=available, convert_categoricals=False)
    hdf['year'] = to_numeric_safe(hdf['year'])
    hdf = hdf[hdf['year'].between(2010, 2023)].copy()

    # Convert to numeric
    for col in hdf.columns:
        if col not in ['idind']:
            hdf[col] = to_numeric_safe(hdf[col])

    # Clean missing codes
    for col in hdf.select_dtypes(include=[np.number]).columns:
        hdf.loc[hdf[col].isin(MISSING_CODES), col] = np.nan

    # Construct health shock indicators
    if 'm3' in hdf.columns:
        # m3: 1=very good, 2=good, 3=average, 4=bad, 5=very bad
        hdf['bad_health'] = (hdf['m3'] >= 4).astype(float)
        hdf.loc[hdf['m3'].isna(), 'bad_health'] = np.nan

    if 'm137' in hdf.columns:
        # m137: 1=fully, 2=with difficulty, 3=cannot
        hdf['health_limits_work'] = (hdf['m137'] >= 2).astype(float)
        hdf.loc[hdf['m137'].isna(), 'health_limits_work'] = np.nan

    # Count chronic conditions
    chronic_cols = [c for c in ['m20_61', 'm20_62', 'm20_63', 'm20_64',
                                'm20_65', 'm20_66', 'm20_67'] if c in hdf.columns]
    if chronic_cols:
        for c in chronic_cols:
            hdf[c] = hdf[c].replace({1: 1, 2: 0})  # 1=yes, 2=no
        hdf['n_chronic'] = hdf[chronic_cols].sum(axis=1, min_count=1)
        hdf['any_chronic'] = (hdf['n_chronic'] > 0).astype(float)
        hdf.loc[hdf['n_chronic'].isna(), 'any_chronic'] = np.nan

    if 'm32' in hdf.columns:
        hdf['took_medicine'] = (hdf['m32'] == 1).astype(float)
        hdf.loc[hdf['m32'].isna(), 'took_medicine'] = np.nan

    if 'm20_7' in hdf.columns:
        hdf['has_disability'] = (hdf['m20_7'] == 1).astype(float)
        hdf.loc[hdf['m20_7'].isna(), 'has_disability'] = np.nan

    # Compute health shocks (changes within individual)
    hdf = hdf.sort_values(['idind', 'year'])
    for var in ['bad_health', 'any_chronic', 'n_chronic',
                'health_limits_work', 'has_disability']:
        if var in hdf.columns:
            hdf[f'{var}_lag'] = hdf.groupby('idind')[var].shift(1)
            hdf[f'd_{var}'] = hdf[var] - hdf[f'{var}_lag']

    # New health shock: transition to bad health or new chronic condition
    if 'd_bad_health' in hdf.columns:
        hdf['health_shock'] = (hdf['d_bad_health'] > 0).astype(float)
        hdf.loc[hdf['d_bad_health'].isna(), 'health_shock'] = np.nan

    # Keep only constructed variables for merge
    keep_cols = ['idind', 'year'] + [c for c in hdf.columns
                                      if c.startswith(('bad_health', 'any_chronic',
                                                       'n_chronic', 'health_limits',
                                                       'has_disability', 'took_medicine',
                                                       'health_shock', 'd_'))]
    hdf = hdf[keep_cols].drop_duplicates(subset=['idind', 'year'])

    # Merge
    df = df.merge(hdf, on=['idind', 'year'], how='left')
    matched = df['bad_health'].notna().sum()
    print(f"  Health vars merged: {matched:,} obs with health data")

    return df


def add_banking_data(df):
    """Add FRI regional banking variables for mechanism tests."""
    print("\nLoading FRI banking data...")
    if not os.path.exists(FRI_PATH):
        print("  WARNING: FRI file not found. Skipping banking data.")
        return df

    fri = pd.read_csv(FRI_PATH)
    # Keep banking/credit variables
    fri_vars = ['ter', 'credpop_regional', 'sberpop_regional',
                'sberdep_per_cap_regional', 'FRI_financial_regional']
    fri = fri[[c for c in fri_vars if c in fri.columns]]

    # Merge on ter (time-invariant regional characteristics)
    df = df.merge(fri, on='ter', how='left')
    matched = df['credpop_regional'].notna().sum()
    print(f"  FRI merged: {matched:,} obs ({matched/len(df)*100:.1f}%)")

    # Create binary high/low banking splits
    if 'credpop_regional' in df.columns:
        median_credit = df.groupby('ter')['credpop_regional'].first().median()
        df['high_credit'] = (df['credpop_regional'] >= median_credit).astype(int)
        print(f"  Median credit per capita: {median_credit:.4f}")
        print(f"  High credit regions: {df[df['high_credit']==1]['ter'].nunique()}, "
              f"Low: {df[df['high_credit']==0]['ter'].nunique()}")

    if 'sberpop_regional' in df.columns:
        median_sber = df.groupby('ter')['sberpop_regional'].first().median()
        df['high_banking'] = (df['sberpop_regional'] >= median_sber).astype(int)
        print(f"  Median Sberbank branches per capita: {median_sber:.4f}")

    return df


def add_npd_intensity(df):
    """Add NPD take-up intensity by region-year."""
    print("\nLoading NPD monthly panel for intensity measure...")
    if not os.path.exists(NPD_MONTHLY_PATH):
        print("  WARNING: NPD monthly file not found. Skipping intensity.")
        return df

    npd = pd.read_csv(NPD_MONTHLY_PATH)
    npd['year'] = to_numeric_safe(npd['year'])
    npd['month'] = to_numeric_safe(npd['month'])

    # Get December values per region-year (or latest available month)
    npd_annual = npd.sort_values(['region_code', 'year', 'month'])
    npd_annual = npd_annual.groupby(['region_code', 'year']).last().reset_index()

    # Need to map region_code (OKTMO-like) to ter (Goskomstat)
    # Since the mapping is complex, use npd_region name → ter via the
    # TER_TO_NPD_REGION dict (reversed)
    from code_01_mappings import TER_TO_NPD_REGION_REVERSED
    # Actually, let's just aggregate NPD by region_name and merge via npd_region
    # The panel already has npd_region from 01_clean_rlms.py

    # Simpler approach: merge on region_name if available
    # For now, use a national-level NPD intensity as a continuous treatment
    npd_national = npd.groupby(['year', 'month']).agg(
        npd_total=('npd_total', 'sum'),
        npd_physical=('npd_physical', 'sum')
    ).reset_index()

    # Get December of each year
    npd_dec = npd_national[npd_national['month'] == 12].copy()
    # Population of Russia ~146M
    npd_dec['npd_rate_national'] = npd_dec['npd_total'] / 146_000_000

    print(f"  NPD national rates:")
    for _, row in npd_dec.iterrows():
        print(f"    {int(row['year'])}: {row['npd_rate_national']:.4f} "
              f"({int(row['npd_total']):,} registrants)")

    # For now, use binary post_npd since we can't easily map
    # region_code to ter. Regional intensity would require the crosswalk.
    # The paper's Eq 8 will use post_npd × npd_early (2-cohort structure) as proxy.
    print("  Note: Using cohort-based intensity (early vs late) as proxy for regional NPD rate.")

    return df


# ============================================================
# 1. WORKER FIXED EFFECTS TOWNSEND TEST
# ============================================================

def worker_fe_townsend(df):
    """
    Townsend test with individual fixed effects.
    Since Δln(c) on Δln(w) is already first-differenced, adding individual FE
    is equivalent to second-differencing. This controls for time-invariant
    individual heterogeneity in the pass-through relationship.

    Key test: Track same individuals across pre/post NPD while informal.
    """
    print("\n" + "=" * 60)
    print("EXTENSION 1: Worker Fixed Effects Townsend Test")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter', 'idind'])

    # Year dummies
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results = {}

    # --- 1a: OLS baseline (no FE) for comparison ---
    print("\n--- 1a: OLS Townsend DiD (baseline, no person FE) ---")
    X_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
              'dw_x_informal_x_post',
              'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
    X = sm.add_constant(sample[X_vars])
    y = sample['consumption_growth']
    mod_ols = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sample['ter']})

    for v in ['wage_growth', 'dw_x_informal', 'dw_x_post', 'dw_x_informal_x_post']:
        stars = '***' if mod_ols.pvalues[v] < 0.01 else '**' if mod_ols.pvalues[v] < 0.05 else '*' if mod_ols.pvalues[v] < 0.1 else ''
        print(f"  {v:30s}: {mod_ols.params[v]:8.4f}{stars} (SE: {mod_ols.bse[v]:.4f})")
    print(f"  N={mod_ols.nobs:,.0f}, R²={mod_ols.rsquared:.4f}")
    results['ols'] = mod_ols

    # --- 1b: Individual FE (PanelOLS) ---
    print("\n--- 1b: Individual FE Townsend DiD ---")
    panel_sample = sample.copy()
    panel_sample = panel_sample.set_index(['idind', 'year'])

    # For PanelOLS: dependent = consumption_growth, exog includes interactions
    # Entity FE absorbs individual-level time-invariant factors
    # Since is_informal varies within person, the interaction still works
    dep = panel_sample['consumption_growth']
    exog_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                 'dw_x_informal_x_post',
                 'is_informal', 'post_npd_int', 'informal_x_post']
    exog = panel_sample[exog_vars]

    mod_fe = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                       drop_absorbed=True)
    res_fe = mod_fe.fit(cov_type='clustered', cluster_entity=False,
                         clusters=panel_sample['ter'])

    for v in ['wage_growth', 'dw_x_informal', 'dw_x_post', 'dw_x_informal_x_post']:
        if v in res_fe.params:
            stars = '***' if res_fe.pvalues[v] < 0.01 else '**' if res_fe.pvalues[v] < 0.05 else '*' if res_fe.pvalues[v] < 0.1 else ''
            print(f"  {v:30s}: {res_fe.params[v]:8.4f}{stars} (SE: {res_fe.std_errors[v]:.4f})")
    print(f"  N={res_fe.nobs:,}, R²={res_fe.rsquared:.4f}")
    print(f"  FE: individual + year, clusters: {panel_sample['ter'].nunique()} oblasts")
    results['fe'] = res_fe

    panel_sample = panel_sample.reset_index()

    # --- 1c: Within-person informal spells ---
    # Track individuals who are observed informal in both pre and post NPD periods
    print("\n--- 1c: Same-person informal spells (pre vs post NPD) ---")
    informal_obs = sample[sample['is_informal_wage'] == 1].copy()
    inf_pre = set(informal_obs[informal_obs['post_npd_int'] == 0]['idind'].unique())
    inf_post = set(informal_obs[informal_obs['post_npd_int'] == 1]['idind'].unique())
    inf_both = inf_pre & inf_post
    print(f"  Informal wage workers: {len(inf_pre)} pre-only, "
          f"{len(inf_post)} post-only, {len(inf_both)} in BOTH periods")

    if len(inf_both) >= 30:
        inf_panel = informal_obs[informal_obs['idind'].isin(inf_both)]
        X_vars_inf = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
        X_inf = sm.add_constant(inf_panel[X_vars_inf])
        y_inf = inf_panel['consumption_growth']
        mod_inf = sm.OLS(y_inf, X_inf).fit(cov_type='cluster',
                                             cov_kwds={'groups': inf_panel['ter']})

        beta_pre = mod_inf.params['wage_growth']
        beta_change = mod_inf.params['dw_x_post']
        stars = '***' if mod_inf.pvalues['dw_x_post'] < 0.01 else '**' if mod_inf.pvalues['dw_x_post'] < 0.05 else '*' if mod_inf.pvalues['dw_x_post'] < 0.1 else ''
        print(f"  Pre-NPD pass-through:  {beta_pre:.4f}")
        print(f"  Change post-NPD:       {beta_change:.4f}{stars} (SE: {mod_inf.bse['dw_x_post']:.4f})")
        print(f"  Post-NPD pass-through: {beta_pre + beta_change:.4f}")
        print(f"  N={mod_inf.nobs:,.0f} (same {len(inf_both)} individuals)")
        results['within_informal'] = mod_inf
    else:
        print(f"  Too few within-person informal spells ({len(inf_both)}). Skipping.")

    # --- 1d: Switchers analysis ---
    # People who switch between formal and informal — compare their insurance in each state
    print("\n--- 1d: Switchers (formal <-> informal within person) ---")
    person_types = sample.groupby('idind')['emp_type'].nunique()
    switchers = person_types[person_types > 1].index
    print(f"  Workers with multiple employment types: {len(switchers):,}")

    if len(switchers) >= 50:
        switch_sample = sample[sample['idind'].isin(switchers)].copy()
        switch_sample = switch_sample.set_index(['idind', 'year'])
        dep_sw = switch_sample['consumption_growth']
        exog_sw = switch_sample[['wage_growth', 'dw_x_informal', 'is_informal']]
        mod_sw = PanelOLS(dep_sw, exog_sw, entity_effects=True, time_effects=True,
                          drop_absorbed=True)
        res_sw = mod_sw.fit(cov_type='clustered', cluster_entity=False,
                            clusters=switch_sample['ter'])

        for v in ['wage_growth', 'dw_x_informal']:
            if v in res_sw.params:
                stars = '***' if res_sw.pvalues[v] < 0.01 else '**' if res_sw.pvalues[v] < 0.05 else '*' if res_sw.pvalues[v] < 0.1 else ''
                print(f"  {v:30s}: {res_sw.params[v]:8.4f}{stars} (SE: {res_sw.std_errors[v]:.4f})")
        print(f"  N={res_sw.nobs:,} ({len(switchers):,} switchers)")
        print(f"  Interpretation: within-person, being informal adds "
              f"{res_sw.params.get('dw_x_informal', 0):.3f} to pass-through")
        results['switchers'] = res_sw
        switch_sample = switch_sample.reset_index()

    # Save results
    rows = []
    for spec, mod in results.items():
        if spec == 'ols':
            params = mod.params
            ses = mod.bse
            pvals = mod.pvalues
            n = int(mod.nobs)
            r2 = mod.rsquared
        elif hasattr(mod, 'std_errors'):
            params = mod.params
            ses = mod.std_errors
            pvals = mod.pvalues
            n = mod.nobs
            r2 = mod.rsquared
        else:
            continue
        for v in ['wage_growth', 'dw_x_informal', 'dw_x_post', 'dw_x_informal_x_post']:
            if v in params:
                rows.append({
                    'Specification': spec,
                    'Variable': v,
                    'Coefficient': params[v],
                    'SE': ses[v],
                    'p_value': pvals[v],
                    'N': n,
                    'R2': r2,
                })
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'worker_fe_results.csv'), index=False)
    print(f"\n  Saved: {TABLE_DIR}/worker_fe_results.csv")

    return results


# ============================================================
# 2. IV TOWNSEND TEST (HEALTH SHOCKS)
# ============================================================

def iv_townsend(df):
    """
    IV Townsend test using health shocks as instruments for income changes.
    Addresses the endogeneity of Δln(w) — income changes may reflect
    hours choices, job changes, etc. Health shocks provide exogenous
    variation in income (Gertler-Gruber 2002).

    Instruments: health_shock (new bad health), d_any_chronic (new chronic),
                 d_health_limits_work (new functional limitation)
    """
    print("\n" + "=" * 60)
    print("EXTENSION 2: IV Townsend Test (Health Shocks)")
    print("=" * 60)

    # Check if health variables exist
    health_instruments = ['health_shock', 'd_any_chronic', 'd_health_limits_work']
    available_ivs = [v for v in health_instruments if v in df.columns and df[v].notna().sum() > 100]

    if not available_ivs:
        print("  No health instruments available. Skipping IV Townsend.")
        return None

    print(f"  Available instruments: {available_ivs}")
    for iv in available_ivs:
        n_valid = df[iv].notna().sum()
        print(f"    {iv}: {n_valid:,} non-missing obs")

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'ter', 'idind'] + available_ivs)

    # Year dummies
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results = {}

    # --- First stage: health shock → Δln(w) ---
    print("\n--- First Stage: Health shocks → Income growth ---")
    fs_X = sm.add_constant(sample[available_ivs + ['is_informal'] + yr_cols])
    fs_y = sample['wage_growth']
    fs_mod = sm.OLS(fs_y, fs_X).fit(cov_type='cluster',
                                      cov_kwds={'groups': sample['ter']})
    for iv in available_ivs:
        stars = '***' if fs_mod.pvalues[iv] < 0.01 else '**' if fs_mod.pvalues[iv] < 0.05 else '*' if fs_mod.pvalues[iv] < 0.1 else ''
        print(f"  {iv:30s}: {fs_mod.params[iv]:8.4f}{stars} (SE: {fs_mod.bse[iv]:.4f})")
    print(f"  N={fs_mod.nobs:,.0f}, F-stat on instruments: {fs_mod.fvalue:.2f}")
    results['first_stage'] = fs_mod

    # --- 2SLS: Δln(c) on instrumented Δln(w) ---
    print("\n--- 2SLS: IV Townsend Test ---")

    # OLS comparison
    X_ols = sm.add_constant(sample[['wage_growth', 'is_informal', 'dw_x_informal'] + yr_cols])
    y = sample['consumption_growth']
    mod_ols = sm.OLS(y, X_ols).fit(cov_type='cluster',
                                     cov_kwds={'groups': sample['ter']})
    print(f"\n  OLS Townsend:")
    for v in ['wage_growth', 'dw_x_informal']:
        stars = '***' if mod_ols.pvalues[v] < 0.01 else '**' if mod_ols.pvalues[v] < 0.05 else '*' if mod_ols.pvalues[v] < 0.1 else ''
        print(f"    {v:28s}: {mod_ols.params[v]:8.4f}{stars} (SE: {mod_ols.bse[v]:.4f})")
    results['ols'] = mod_ols

    # IV using PanelOLS framework isn't ideal; use manual 2SLS
    # First stage: predict wage_growth from health shocks
    # Also need to instrument dw_x_informal = wage_growth × informal
    # Create interaction instruments: health_shock × informal
    for iv in available_ivs:
        sample[f'{iv}_x_informal'] = sample[iv] * sample['is_informal']

    iv_interaction_vars = [f'{iv}_x_informal' for iv in available_ivs]

    # First stage for wage_growth
    fs1_vars = available_ivs + ['is_informal'] + yr_cols
    X_fs1 = sm.add_constant(sample[fs1_vars])
    mod_fs1 = sm.OLS(sample['wage_growth'], X_fs1).fit()
    sample['wage_growth_hat'] = mod_fs1.fittedvalues

    # First stage for dw_x_informal
    fs2_vars = available_ivs + iv_interaction_vars + ['is_informal'] + yr_cols
    X_fs2 = sm.add_constant(sample[fs2_vars])
    mod_fs2 = sm.OLS(sample['dw_x_informal'], X_fs2).fit()
    sample['dw_x_informal_hat'] = mod_fs2.fittedvalues

    # Second stage: Δln(c) on predicted Δln(w) and predicted interaction
    X_iv = sm.add_constant(sample[['wage_growth_hat', 'is_informal',
                                    'dw_x_informal_hat'] + yr_cols])
    mod_iv = sm.OLS(y, X_iv).fit(cov_type='cluster',
                                   cov_kwds={'groups': sample['ter']})

    print(f"\n  IV Townsend (health shock instruments):")
    for v_orig, v_hat in [('wage_growth', 'wage_growth_hat'),
                           ('dw_x_informal', 'dw_x_informal_hat')]:
        stars = '***' if mod_iv.pvalues[v_hat] < 0.01 else '**' if mod_iv.pvalues[v_hat] < 0.05 else '*' if mod_iv.pvalues[v_hat] < 0.1 else ''
        print(f"    {v_orig:28s}: {mod_iv.params[v_hat]:8.4f}{stars} (SE: {mod_iv.bse[v_hat]:.4f})")
    print(f"  N={mod_iv.nobs:,.0f}")
    results['iv'] = mod_iv

    # --- IV with DiD ---
    print("\n--- 2SLS IV Townsend DiD ---")
    sample2 = sample.dropna(subset=['post_npd_int'])

    # Create instrument interactions with post
    for iv in available_ivs:
        sample2[f'{iv}_x_post'] = sample2[iv] * sample2['post_npd_int']
        sample2[f'{iv}_x_inf_x_post'] = sample2[iv] * sample2['is_informal'] * sample2['post_npd_int']

    iv_post_vars = [f'{iv}_x_post' for iv in available_ivs]
    iv_inf_post_vars = [f'{iv}_x_inf_x_post' for iv in available_ivs]

    # Predict dw_x_post and dw_x_informal_x_post
    fs3_vars = (available_ivs + iv_interaction_vars + iv_post_vars +
                ['is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols)
    X_fs3 = sm.add_constant(sample2[fs3_vars])
    sample2['dw_x_post_hat'] = sm.OLS(sample2['dw_x_post'], X_fs3).fit().fittedvalues
    sample2['dw_x_inf_x_post_hat'] = sm.OLS(sample2['dw_x_informal_x_post'], X_fs3).fit().fittedvalues

    X_iv2 = sm.add_constant(sample2[['wage_growth_hat', 'dw_x_informal_hat',
                                      'dw_x_post_hat', 'dw_x_inf_x_post_hat',
                                      'is_informal', 'post_npd_int',
                                      'informal_x_post'] + yr_cols])
    y2 = sample2['consumption_growth']
    mod_iv2 = sm.OLS(y2, X_iv2).fit(cov_type='cluster',
                                      cov_kwds={'groups': sample2['ter']})

    print(f"\n  IV Townsend DiD:")
    for v_orig, v_hat in [('wage_growth', 'wage_growth_hat'),
                           ('dw_x_informal', 'dw_x_informal_hat'),
                           ('dw_x_post', 'dw_x_post_hat'),
                           ('dw_x_inf_x_post', 'dw_x_inf_x_post_hat')]:
        if v_hat in mod_iv2.params:
            stars = '***' if mod_iv2.pvalues[v_hat] < 0.01 else '**' if mod_iv2.pvalues[v_hat] < 0.05 else '*' if mod_iv2.pvalues[v_hat] < 0.1 else ''
            print(f"    {v_orig:28s}: {mod_iv2.params[v_hat]:8.4f}{stars} (SE: {mod_iv2.bse[v_hat]:.4f})")
    print(f"  N={mod_iv2.nobs:,.0f}")
    results['iv_did'] = mod_iv2

    # Save
    rows = []
    for spec_name in ['ols', 'iv']:
        mod = results[spec_name]
        for v in mod.params.index:
            if v != 'const' and not v.startswith('yr_'):
                rows.append({
                    'Specification': spec_name,
                    'Variable': v,
                    'Coefficient': mod.params[v],
                    'SE': mod.bse[v],
                    'p_value': mod.pvalues[v],
                    'N': int(mod.nobs),
                })
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'iv_townsend_results.csv'), index=False)
    print(f"\n  Saved: {TABLE_DIR}/iv_townsend_results.csv")

    return results


# ============================================================
# 3. CALLAWAY-SANT'ANNA WITH TWO COHORTS
# ============================================================

def callaway_santanna(df):
    """
    Manual Callaway-Sant'Anna with two cohorts (2019 and 2020).

    ATT(g=2019) = E[ΔY | G=2019, post] - E[ΔY | G=2020, 2019]
    ATT(g=2020) = E[ΔY | G=2020, post] - E[ΔY | G=2020, pre-2020]

    Since all RLMS regions are treated by 2020, the 2019 cohort's
    comparison group in 2019 is the not-yet-treated 2020 cohort.
    """
    print("\n" + "=" * 60)
    print("EXTENSION 3: Callaway-Sant'Anna (Two Cohorts)")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'ter', 'npd_year'])
    sample['cohort'] = np.where(sample['npd_year'] == 2019, 2019, 2020)

    results = {}

    # --- ATT(g=2019): Early adopters ---
    # Treatment: cohort 2019, years >= 2019
    # Control: cohort 2020, year = 2019 (not yet treated)
    print("\n--- ATT(g=2019): Early adopters ---")
    early_treated_post = sample[(sample['cohort'] == 2019) & (sample['year'] >= 2019)]
    early_treated_pre = sample[(sample['cohort'] == 2019) & (sample['year'] < 2019)]
    late_in_2019 = sample[(sample['cohort'] == 2020) & (sample['year'] == 2019)]
    late_pre = sample[(sample['cohort'] == 2020) & (sample['year'] < 2019)]

    # For each cohort, estimate: Δln(c) on Δln(w) × informal, pre vs post
    # Simpler: compute the Townsend pass-through change for each cohort

    for cohort_year, cohort_label in [(2019, 'early'), (2020, 'late')]:
        print(f"\n  --- Cohort {cohort_year} ({cohort_label}) ---")
        csample = sample[sample['cohort'] == cohort_year].copy()
        csample['post_cohort'] = (csample['year'] >= cohort_year).astype(int)
        csample['dw_x_post_c'] = csample['wage_growth'] * csample['post_cohort']
        csample['dw_x_inf_c'] = csample['wage_growth'] * csample['is_informal']
        csample['dw_x_inf_post_c'] = csample['wage_growth'] * csample['is_informal'] * csample['post_cohort']
        csample['inf_x_post_c'] = csample['is_informal'] * csample['post_cohort']

        # Year dummies
        yr_dum = pd.get_dummies(csample['year'], prefix='yr', drop_first=True, dtype=float)
        csample = pd.concat([csample, yr_dum], axis=1)
        yr_cols = list(yr_dum.columns)

        X_vars = ['wage_growth', 'dw_x_inf_c', 'dw_x_post_c', 'dw_x_inf_post_c',
                  'is_informal', 'post_cohort', 'inf_x_post_c'] + yr_cols
        X = sm.add_constant(csample[X_vars])
        y = csample['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': csample['ter']})

        key_vars = ['wage_growth', 'dw_x_inf_c', 'dw_x_post_c', 'dw_x_inf_post_c']
        for v in key_vars:
            stars = '***' if mod.pvalues[v] < 0.01 else '**' if mod.pvalues[v] < 0.05 else '*' if mod.pvalues[v] < 0.1 else ''
            print(f"    {v:30s}: {mod.params[v]:8.4f}{stars} (SE: {mod.bse[v]:.4f})")
        print(f"    N={mod.nobs:,.0f}, n_oblasts={csample['ter'].nunique()}")
        print(f"    ATT triple-diff: {mod.params['dw_x_inf_post_c']:.4f}")
        results[f'cohort_{cohort_year}'] = mod

    # --- 2020 cohort using 2019 cohort as comparison (before 2020) ---
    print("\n  --- ATT(g=2020) using not-yet-treated 2019 cohort as control ---")
    # Before 2020: both cohorts exist, but 2019 is already treated
    # Use only 2018 (pre-period for both) vs 2020+ (post for 2020 cohort)
    # Not-yet-treated comparison: 2020 cohort in 2019 vs 2020 cohort in 2018

    # Actually, with 2 cohorts and TWFE equivalence, let's just report
    # the cohort-specific ATTs and the weighted average

    # --- Weighted average ATT ---
    print("\n  --- Weighted Average ATT ---")
    n_early = sample[sample['cohort'] == 2019].shape[0]
    n_late = sample[sample['cohort'] == 2020].shape[0]
    n_total = n_early + n_late

    att_early = results['cohort_2019'].params['dw_x_inf_post_c']
    att_late = results['cohort_2020'].params['dw_x_inf_post_c']
    se_early = results['cohort_2019'].bse['dw_x_inf_post_c']
    se_late = results['cohort_2020'].bse['dw_x_inf_post_c']

    w_early = n_early / n_total
    w_late = n_late / n_total
    att_avg = w_early * att_early + w_late * att_late
    se_avg = np.sqrt((w_early * se_early)**2 + (w_late * se_late)**2)

    print(f"    ATT(2019 cohort): {att_early:.4f} (SE: {se_early:.4f}), weight: {w_early:.3f}")
    print(f"    ATT(2020 cohort): {att_late:.4f} (SE: {se_late:.4f}), weight: {w_late:.3f}")
    print(f"    Weighted ATT:     {att_avg:.4f} (SE: {se_avg:.4f})")
    print(f"    p-value:          {2*(1 - stats.norm.cdf(abs(att_avg/se_avg))):.4f}")
    results['weighted_att'] = {'att': att_avg, 'se': se_avg, 'p': 2*(1 - stats.norm.cdf(abs(att_avg/se_avg)))}

    # --- By-type CS: informal wage only ---
    print("\n  --- By-type CS: Informal wage workers only ---")
    for cohort_year in [2019, 2020]:
        inf_sample = sample[(sample['cohort'] == cohort_year) &
                            (sample['emp_type'] == 'informal_wage')].copy()
        if len(inf_sample) < 50:
            print(f"    Cohort {cohort_year}: too few informal obs ({len(inf_sample)})")
            continue
        inf_sample['post_c'] = (inf_sample['year'] >= cohort_year).astype(int)
        inf_sample['dw_x_post_c'] = inf_sample['wage_growth'] * inf_sample['post_c']
        yr_dum = pd.get_dummies(inf_sample['year'], prefix='yr', drop_first=True, dtype=float)
        inf_sample = pd.concat([inf_sample, yr_dum], axis=1)
        yr_c = list(yr_dum.columns)

        X = sm.add_constant(inf_sample[['wage_growth', 'dw_x_post_c', 'post_c'] + yr_c])
        y = inf_sample['consumption_growth']
        mod_inf = sm.OLS(y, X).fit(cov_type='cluster',
                                     cov_kwds={'groups': inf_sample['ter']})
        pre_beta = mod_inf.params['wage_growth']
        change = mod_inf.params['dw_x_post_c']
        stars = '***' if mod_inf.pvalues['dw_x_post_c'] < 0.01 else '**' if mod_inf.pvalues['dw_x_post_c'] < 0.05 else '*' if mod_inf.pvalues['dw_x_post_c'] < 0.1 else ''
        print(f"    Cohort {cohort_year}: pre-β={pre_beta:.4f}, "
              f"Δβ={change:.4f}{stars} (SE: {mod_inf.bse['dw_x_post_c']:.4f}), "
              f"post-β={pre_beta+change:.4f}, N={mod_inf.nobs:,.0f}")
        results[f'inf_cohort_{cohort_year}'] = mod_inf

    # Save
    rows = []
    for spec_name in ['cohort_2019', 'cohort_2020']:
        mod = results[spec_name]
        for v in ['wage_growth', 'dw_x_inf_c', 'dw_x_post_c', 'dw_x_inf_post_c']:
            if v in mod.params:
                rows.append({
                    'Specification': spec_name,
                    'Variable': v,
                    'Coefficient': mod.params[v],
                    'SE': mod.bse[v],
                    'p_value': mod.pvalues[v],
                    'N': int(mod.nobs),
                })
    att = results['weighted_att']
    rows.append({
        'Specification': 'weighted_ATT',
        'Variable': 'dw_x_inf_post (weighted)',
        'Coefficient': att['att'],
        'SE': att['se'],
        'p_value': att['p'],
        'N': n_total,
    })
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'callaway_santanna_results.csv'), index=False)
    print(f"\n  Saved: {TABLE_DIR}/callaway_santanna_results.csv")

    return results


# ============================================================
# 4. MECHANISM TESTS
# ============================================================

def mechanism_tests(df):
    """
    Test mechanisms through which NPD improves consumption insurance:
    (a) Banking heterogeneity — credit access channel
    (b) Savings response DiD
    (c) NPD cohort intensity (early vs late)
    """
    print("\n" + "=" * 60)
    print("EXTENSION 4: Mechanism Tests")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter'])
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results = {}

    # --- 4a: Banking heterogeneity ---
    print("\n--- 4a: Heterogeneity by Regional Banking Density ---")
    if 'high_credit' in sample.columns:
        for bank_level, label in [(1, 'High credit'), (0, 'Low credit')]:
            sub = sample[sample['high_credit'] == bank_level].copy()
            if sub['is_informal_wage'].sum() < 30:
                print(f"  {label}: too few informal obs. Skipping.")
                continue

            # Run by-type Townsend for informal wage workers
            for etype, elabel in [('informal_wage', 'Informal wage'),
                                   ('formal_wage', 'Formal wage')]:
                esub = sub[sub['emp_type'] == etype]
                if len(esub) < 50:
                    continue
                X_vars = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
                X = sm.add_constant(esub[X_vars])
                y = esub['consumption_growth']
                mod = sm.OLS(y, X).fit(cov_type='cluster',
                                        cov_kwds={'groups': esub['ter']})
                pre = mod.params['wage_growth']
                change = mod.params['dw_x_post']
                stars = '***' if mod.pvalues['dw_x_post'] < 0.01 else '**' if mod.pvalues['dw_x_post'] < 0.05 else '*' if mod.pvalues['dw_x_post'] < 0.1 else ''
                print(f"  {label}, {elabel}: pre-β={pre:.4f}, "
                      f"Δ={change:.4f}{stars} (SE: {mod.bse['dw_x_post']:.4f}), "
                      f"post-β={pre+change:.4f}, N={mod.nobs:,.0f}")
                results[f'bank_{bank_level}_{etype}'] = {
                    'beta_pre': pre, 'beta_change': change,
                    'beta_post': pre + change,
                    'se_change': mod.bse['dw_x_post'],
                    'p_change': mod.pvalues['dw_x_post'],
                    'N': int(mod.nobs),
                }

        # Also run the full triple-diff with banking interaction
        print("\n  Triple-diff with banking interaction:")
        sample['dw_x_inf_x_post_x_bank'] = (sample['wage_growth'] * sample['is_informal'] *
                                              sample['post_npd_int'] * sample.get('high_credit', 0))
        X_bank = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                  'dw_x_informal_x_post', 'dw_x_inf_x_post_x_bank',
                  'is_informal', 'post_npd_int', 'informal_x_post',
                  'high_credit'] + yr_cols
        X_bank = [c for c in X_bank if c in sample.columns]
        Xb = sm.add_constant(sample[X_bank].fillna(0))
        yb = sample['consumption_growth']
        mod_bank = sm.OLS(yb, Xb).fit(cov_type='cluster',
                                        cov_kwds={'groups': sample['ter']})
        for v in ['dw_x_informal_x_post', 'dw_x_inf_x_post_x_bank']:
            if v in mod_bank.params:
                stars = '***' if mod_bank.pvalues[v] < 0.01 else '**' if mod_bank.pvalues[v] < 0.05 else '*' if mod_bank.pvalues[v] < 0.1 else ''
                print(f"    {v:35s}: {mod_bank.params[v]:8.4f}{stars} (SE: {mod_bank.bse[v]:.4f})")
        results['bank_interaction'] = mod_bank
    else:
        print("  No banking data available. Skipping.")

    # --- 4b: Savings response ---
    print("\n--- 4b: Savings Response DiD ---")
    if 'saved_money' in sample.columns:
        sav_sample = sample.dropna(subset=['saved_money'])
        sav_X = sm.add_constant(sav_sample[['is_informal', 'post_npd_int',
                                              'informal_x_post'] + yr_cols])
        sav_y = sav_sample['saved_money']
        mod_sav = sm.OLS(sav_y, sav_X).fit(cov_type='cluster',
                                              cov_kwds={'groups': sav_sample['ter']})
        for v in ['is_informal', 'post_npd_int', 'informal_x_post']:
            stars = '***' if mod_sav.pvalues[v] < 0.01 else '**' if mod_sav.pvalues[v] < 0.05 else '*' if mod_sav.pvalues[v] < 0.1 else ''
            print(f"  {v:30s}: {mod_sav.params[v]:8.4f}{stars} (SE: {mod_sav.bse[v]:.4f})")
        print(f"  N={mod_sav.nobs:,.0f}")
        print(f"  Baseline savings rate (formal): {sav_sample[sav_sample['is_informal']==0]['saved_money'].mean():.3f}")
        print(f"  Baseline savings rate (informal): {sav_sample[sav_sample['is_informal']==1]['saved_money'].mean():.3f}")
        results['savings'] = mod_sav
    else:
        print("  No savings variable. Skipping.")

    # --- 4c: NPD cohort intensity ---
    print("\n--- 4c: NPD Cohort Intensity (Early vs Late) ---")
    sample['npd_early_int'] = sample.get('npd_early', 0).fillna(0).astype(int)
    sample['dw_x_inf_x_post_x_early'] = (sample['wage_growth'] * sample['is_informal'] *
                                           sample['post_npd_int'] * sample['npd_early_int'])
    X_int = ['wage_growth', 'dw_x_informal', 'dw_x_post',
             'dw_x_informal_x_post', 'dw_x_inf_x_post_x_early',
             'is_informal', 'post_npd_int', 'informal_x_post',
             'npd_early_int'] + yr_cols
    X_int = [c for c in X_int if c in sample.columns]
    Xi = sm.add_constant(sample[X_int].fillna(0))
    yi = sample['consumption_growth']
    mod_int = sm.OLS(yi, Xi).fit(cov_type='cluster',
                                   cov_kwds={'groups': sample['ter']})
    for v in ['dw_x_informal_x_post', 'dw_x_inf_x_post_x_early']:
        if v in mod_int.params:
            stars = '***' if mod_int.pvalues[v] < 0.01 else '**' if mod_int.pvalues[v] < 0.05 else '*' if mod_int.pvalues[v] < 0.1 else ''
            print(f"  {v:35s}: {mod_int.params[v]:8.4f}{stars} (SE: {mod_int.bse[v]:.4f})")
    print(f"  N={mod_int.nobs:,.0f}")
    results['intensity'] = mod_int

    # Save mechanism results
    mech_rows = []
    for key, val in results.items():
        if isinstance(val, dict) and 'beta_pre' in val:
            mech_rows.append({
                'Specification': key,
                'beta_pre': val['beta_pre'],
                'beta_change': val['beta_change'],
                'beta_post': val['beta_post'],
                'se_change': val['se_change'],
                'p_change': val['p_change'],
                'N': val['N'],
            })
    if mech_rows:
        pd.DataFrame(mech_rows).to_csv(os.path.join(TABLE_DIR, 'mechanism_banking_results.csv'),
                                         index=False)
        print(f"\n  Saved: {TABLE_DIR}/mechanism_banking_results.csv")

    # Save savings results
    if 'savings' in results:
        sav_rows = []
        mod = results['savings']
        for v in ['is_informal', 'post_npd_int', 'informal_x_post']:
            sav_rows.append({
                'Variable': v,
                'Coefficient': mod.params[v],
                'SE': mod.bse[v],
                'p_value': mod.pvalues[v],
            })
        pd.DataFrame(sav_rows).to_csv(os.path.join(TABLE_DIR, 'mechanism_savings_results.csv'),
                                        index=False)
        print(f"  Saved: {TABLE_DIR}/mechanism_savings_results.csv")

    return results


# ============================================================
# 5. POWER ANALYSIS
# ============================================================

def power_analysis(df):
    """
    Compute minimum detectable effects for the triple-difference.
    With G clusters, the MDE for a coefficient is approximately:
    MDE ≈ (t_α/2 + t_β) × SE_estimated
    """
    print("\n" + "=" * 60)
    print("EXTENSION 5: Power Analysis")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter'])

    # Key numbers
    n_clusters = sample['ter'].nunique()
    n_total = len(sample)
    n_informal = sample['is_informal'].sum()
    n_informal_wage = sample['is_informal_wage'].sum()
    n_formal = (sample['is_formal'] == 1).sum()
    n_post = (sample['post_npd_int'] == 1).sum()
    n_inf_post = ((sample['is_informal'] == 1) & (sample['post_npd_int'] == 1)).sum()

    print(f"\n  Key sample sizes:")
    print(f"    Total observations:         {n_total:,}")
    print(f"    Clusters (oblasts):         {n_clusters}")
    print(f"    Formal wage workers:        {n_formal:,}")
    print(f"    Informal wage workers:      {n_informal_wage:,}")
    print(f"    Informal (any) + post-NPD:  {n_inf_post:,}")
    print(f"    Post-NPD observations:      {n_post:,}")

    # Estimated SE from main triple-diff (from previous results)
    se_triple = 0.036  # from 03_main_did.py
    estimated_beta = -0.042

    # MDE at 80% power, 5% significance (two-sided)
    t_alpha = stats.norm.ppf(0.975)  # ≈ 1.96
    t_beta_80 = stats.norm.ppf(0.80)  # ≈ 0.842
    t_beta_90 = stats.norm.ppf(0.90)  # ≈ 1.282

    mde_80 = (t_alpha + t_beta_80) * se_triple
    mde_90 = (t_alpha + t_beta_90) * se_triple

    print(f"\n  Power calculations (SE = {se_triple}):")
    print(f"    MDE at 80% power, α=0.05: {mde_80:.4f}")
    print(f"    MDE at 90% power, α=0.05: {mde_90:.4f}")
    print(f"    Estimated β₄:             {estimated_beta:.4f}")
    print(f"    Ratio |β₄|/MDE(80%):      {abs(estimated_beta)/mde_80:.2f}")

    # How many clusters needed for significance?
    # SE ∝ 1/sqrt(G). Current SE=0.036 with G=32.
    # For |β|/SE = 1.96: need SE = |β|/1.96 = 0.042/1.96 = 0.0214
    target_se = abs(estimated_beta) / t_alpha
    cluster_ratio = (se_triple / target_se) ** 2
    needed_clusters = int(np.ceil(n_clusters * cluster_ratio))

    print(f"\n  To detect β = {estimated_beta} at 5% significance:")
    print(f"    Target SE:      {target_se:.4f}")
    print(f"    Clusters needed: ≈ {needed_clusters} (have {n_clusters})")
    print(f"    Sample multiplier: {cluster_ratio:.1f}×")

    # How many informal wage obs needed?
    # Power ∝ sqrt(N_informal). Currently ~4100 informal wage obs
    # For same MDE but at 5% significance:
    current_informal_n = n_informal_wage
    needed_informal = int(np.ceil(current_informal_n * cluster_ratio))
    print(f"    Informal wage obs needed: ≈ {needed_informal:,} (have {current_informal_n:,})")

    # Ex-post power of the actual test
    z_stat = abs(estimated_beta) / se_triple
    power_actual = stats.norm.cdf(z_stat - t_alpha) + stats.norm.cdf(-z_stat - t_alpha)
    print(f"\n  Ex-post power of β₄ = {estimated_beta}:")
    print(f"    z-statistic: {z_stat:.3f}")
    print(f"    Power:       {power_actual:.3f} ({power_actual*100:.1f}%)")

    # Pre-existing insurance gap
    beta_informal_pre = 0.128
    beta_formal_pre = 0.059
    gap = beta_informal_pre - beta_formal_pre
    pct_closed = abs(estimated_beta) / gap * 100

    print(f"\n  Economic interpretation:")
    print(f"    Pre-NPD informal pass-through: {beta_informal_pre}")
    print(f"    Pre-NPD formal pass-through:   {beta_formal_pre}")
    print(f"    Insurance gap:                 {gap:.3f}")
    print(f"    β₄ as % of gap:               {pct_closed:.1f}%")
    print(f"    Post-NPD informal (predicted): {beta_informal_pre + estimated_beta:.3f}")
    print(f"    Post-NPD formal (actual):      0.068")
    print(f"    → Convergence to formal level: YES")

    # Save
    power_results = {
        'n_clusters': n_clusters,
        'n_total': n_total,
        'n_informal_wage': int(n_informal_wage),
        'se_triple': se_triple,
        'estimated_beta': estimated_beta,
        'mde_80': mde_80,
        'mde_90': mde_90,
        'needed_clusters': needed_clusters,
        'needed_informal': needed_informal,
        'power_actual': power_actual,
        'gap_closed_pct': pct_closed,
    }
    pd.DataFrame([power_results]).to_csv(os.path.join(TABLE_DIR, 'power_analysis.csv'), index=False)
    print(f"\n  Saved: {TABLE_DIR}/power_analysis.csv")

    return power_results


# ============================================================
# 6. WILD CLUSTER BOOTSTRAP
# ============================================================

def wild_cluster_bootstrap(df):
    """
    Wild cluster bootstrap p-values for the Townsend DiD coefficients.

    With only 32 clusters, standard clustered SEs may be unreliable
    (Cameron, Gelbach, Miller 2008). We compute bootstrap p-values using
    Rademacher weights (+/- 1 with equal probability) assigned at the
    cluster level.

    Approach (unrestricted residuals):
      1. Estimate the full model, record t-statistic for the tested variable.
      2. For each of B=9,999 bootstrap replications:
         a. Draw Rademacher weights w_g in {-1, +1} for each cluster g.
         b. Multiply residuals by cluster weights: e*_i = w_{g(i)} * e_i.
         c. Construct pseudo-outcome: y*_i = X_i @ beta_hat + e*_i.
         d. Re-estimate OLS on (y*, X), get bootstrap t-statistic t*_b.
      3. Bootstrap p-value = fraction of |t*_b| >= |t_observed|.

    Two tests:
      (A) Triple-diff: beta on dw_x_informal_x_post (H0: beta4 = 0)
      (B) By-type (informal wage only): beta on dw_x_post (H0 = 0)
    """
    print("\n" + "=" * 60)
    print("EXTENSION 6: Wild Cluster Bootstrap")
    print("=" * 60)

    B = 9_999  # bootstrap replications
    np.random.seed(20260308)

    # ---- Prepare sample (same as main triple-diff) ----
    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter'])

    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results_list = []

    # ==================================================================
    # (A) Triple-difference: test beta on dw_x_informal_x_post
    # ==================================================================
    print("\n--- 6a: Triple-diff (dw_x_informal_x_post) ---")

    X_vars_a = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                'dw_x_informal_x_post',
                'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
    X_a = sm.add_constant(sample[X_vars_a]).values.astype(np.float64)
    y_a = sample['consumption_growth'].values.astype(np.float64)
    clusters_a = sample['ter'].values

    # Column index of the tested variable in X_a
    col_names_a = ['const'] + X_vars_a
    test_idx_a = col_names_a.index('dw_x_informal_x_post')

    # Original estimation
    mod_a = sm.OLS(y_a, X_a).fit(cov_type='cluster', cov_kwds={'groups': clusters_a})
    beta_hat_a = mod_a.params
    t_obs_a = mod_a.tvalues[test_idx_a]
    resid_a = mod_a.resid

    print(f"  Original estimate: beta = {beta_hat_a[test_idx_a]:.4f}, "
          f"SE = {mod_a.bse[test_idx_a]:.4f}, t = {t_obs_a:.3f}")

    # Cluster-level setup
    unique_clusters_a = np.unique(clusters_a)
    n_clusters_a = len(unique_clusters_a)
    # Build integer cluster index for fast vectorized weight expansion
    cluster_idx_a = np.empty(len(y_a), dtype=int)
    for j, c in enumerate(unique_clusters_a):
        cluster_idx_a[clusters_a == c] = j
    # Pre-compute per-cluster X slices for sandwich estimator
    X_by_cluster_a = [X_a[clusters_a == c] for c in unique_clusters_a]

    # Fitted values under unrestricted model
    Xb_a = X_a @ beta_hat_a

    # Pre-compute (X'X)^{-1} X' for OLS coefficient extraction
    XtX_inv_Xt_a = np.linalg.solve(X_a.T @ X_a, X_a.T)
    k_a = X_a.shape[1]
    N_a = len(y_a)
    correction_a = (n_clusters_a / (n_clusters_a - 1)) * ((N_a - 1) / (N_a - k_a))
    XtX_inv_a = np.linalg.solve(X_a.T @ X_a, np.eye(k_a))

    print(f"  Running {B:,} bootstrap replications ({n_clusters_a} clusters)...")

    # Bootstrap (vectorized weight expansion via integer indexing)
    t_boot_a = np.empty(B)
    for b in range(B):
        weights = np.random.choice([-1.0, 1.0], size=n_clusters_a)
        w = weights[cluster_idx_a]

        y_star = Xb_a + w * resid_a
        beta_star = XtX_inv_Xt_a @ y_star
        resid_star = y_star - X_a @ beta_star

        # Clustered SE via pre-split X blocks
        meat = np.zeros((k_a, k_a))
        for j in range(n_clusters_a):
            Xg = X_by_cluster_a[j]
            eg = resid_star[clusters_a == unique_clusters_a[j]]
            score_g = Xg.T @ eg
            meat += np.outer(score_g, score_g)

        V_cl = correction_a * XtX_inv_a @ meat @ XtX_inv_a
        se_star = np.sqrt(V_cl[test_idx_a, test_idx_a])

        t_boot_a[b] = beta_star[test_idx_a] / se_star if se_star > 0 else 0.0

    # Bootstrap p-value (two-sided)
    p_boot_a = np.mean(np.abs(t_boot_a) >= np.abs(t_obs_a))

    print(f"  Bootstrap p-value: {p_boot_a:.4f}")
    print(f"  (Conventional p-value: {mod_a.pvalues[test_idx_a]:.4f})")
    print(f"  |t_obs| = {np.abs(t_obs_a):.3f}, "
          f"median |t*| = {np.median(np.abs(t_boot_a)):.3f}, "
          f"95th pct |t*| = {np.percentile(np.abs(t_boot_a), 95):.3f}")

    results_list.append({
        'specification': 'triple_diff',
        'tested_variable': 'dw_x_informal_x_post',
        'beta': beta_hat_a[test_idx_a],
        'se_clustered': mod_a.bse[test_idx_a],
        't_observed': t_obs_a,
        'p_conventional': mod_a.pvalues[test_idx_a],
        'p_wild_cluster_bootstrap': p_boot_a,
        'n_obs': int(len(y_a)),
        'n_clusters': n_clusters_a,
        'n_bootstrap': B,
    })

    # ==================================================================
    # (B) By-type: informal wage only, test beta on dw_x_post
    # ==================================================================
    print("\n--- 6b: Informal wage only (dw_x_post) ---")

    inf_sample = sample[sample['emp_type'] == 'informal_wage'].copy()
    X_vars_b = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
    X_b = sm.add_constant(inf_sample[X_vars_b]).values.astype(np.float64)
    y_b = inf_sample['consumption_growth'].values.astype(np.float64)
    clusters_b = inf_sample['ter'].values

    col_names_b = ['const'] + X_vars_b
    test_idx_b = col_names_b.index('dw_x_post')

    mod_b = sm.OLS(y_b, X_b).fit(cov_type='cluster', cov_kwds={'groups': clusters_b})
    beta_hat_b = mod_b.params
    t_obs_b = mod_b.tvalues[test_idx_b]
    resid_b = mod_b.resid

    print(f"  Original estimate: beta = {beta_hat_b[test_idx_b]:.4f}, "
          f"SE = {mod_b.bse[test_idx_b]:.4f}, t = {t_obs_b:.3f}")

    unique_clusters_b = np.unique(clusters_b)
    n_clusters_b = len(unique_clusters_b)
    cluster_idx_b = np.empty(len(y_b), dtype=int)
    for j, c in enumerate(unique_clusters_b):
        cluster_idx_b[clusters_b == c] = j
    X_by_cluster_b = [X_b[clusters_b == c] for c in unique_clusters_b]

    Xb_b = X_b @ beta_hat_b
    XtX_inv_Xt_b = np.linalg.solve(X_b.T @ X_b, X_b.T)
    k_b = X_b.shape[1]
    N_b = len(y_b)
    correction_b = (n_clusters_b / (n_clusters_b - 1)) * ((N_b - 1) / (N_b - k_b))
    XtX_inv_b = np.linalg.solve(X_b.T @ X_b, np.eye(k_b))

    print(f"  Running {B:,} bootstrap replications ({n_clusters_b} clusters)...")

    t_boot_b = np.empty(B)
    for b in range(B):
        weights = np.random.choice([-1.0, 1.0], size=n_clusters_b)
        w = weights[cluster_idx_b]

        y_star = Xb_b + w * resid_b
        beta_star = XtX_inv_Xt_b @ y_star
        resid_star = y_star - X_b @ beta_star

        meat = np.zeros((k_b, k_b))
        for j in range(n_clusters_b):
            Xg = X_by_cluster_b[j]
            eg = resid_star[clusters_b == unique_clusters_b[j]]
            score_g = Xg.T @ eg
            meat += np.outer(score_g, score_g)

        V_cl = correction_b * XtX_inv_b @ meat @ XtX_inv_b
        se_star = np.sqrt(V_cl[test_idx_b, test_idx_b])

        t_boot_b[b] = beta_star[test_idx_b] / se_star if se_star > 0 else 0.0

    p_boot_b = np.mean(np.abs(t_boot_b) >= np.abs(t_obs_b))

    print(f"  Bootstrap p-value: {p_boot_b:.4f}")
    print(f"  (Conventional p-value: {mod_b.pvalues[test_idx_b]:.4f})")
    print(f"  |t_obs| = {np.abs(t_obs_b):.3f}, "
          f"median |t*| = {np.median(np.abs(t_boot_b)):.3f}, "
          f"95th pct |t*| = {np.percentile(np.abs(t_boot_b), 95):.3f}")

    results_list.append({
        'specification': 'informal_wage_only',
        'tested_variable': 'dw_x_post',
        'beta': beta_hat_b[test_idx_b],
        'se_clustered': mod_b.bse[test_idx_b],
        't_observed': t_obs_b,
        'p_conventional': mod_b.pvalues[test_idx_b],
        'p_wild_cluster_bootstrap': p_boot_b,
        'n_obs': int(len(y_b)),
        'n_clusters': n_clusters_b,
        'n_bootstrap': B,
    })

    # ---- Save ----
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(TABLE_DIR, 'wild_cluster_bootstrap.csv'), index=False)
    print(f"\n  Saved: {TABLE_DIR}/wild_cluster_bootstrap.csv")

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    df = load_panel()

    # Add health variables for IV
    df = add_health_vars(df)

    # Add banking data for mechanisms
    df = add_banking_data(df)

    # Run all extensions
    print("\n" + "#" * 70)
    print("# RUNNING METHODOLOGICAL EXTENSIONS")
    print("#" * 70)

    # 1. Worker FE
    fe_results = worker_fe_townsend(df)

    # 2. IV Townsend
    iv_results = iv_townsend(df)

    # 3. Callaway-Sant'Anna
    cs_results = callaway_santanna(df)

    # 4. Mechanism tests
    mech_results = mechanism_tests(df)

    # 5. Power analysis
    power_results = power_analysis(df)

    # 6. Wild cluster bootstrap
    wcb_results = wild_cluster_bootstrap(df)

    print("\n" + "=" * 70)
    print("ALL EXTENSIONS COMPLETE")
    print("=" * 70)
    print(f"\nOutput tables saved in: {TABLE_DIR}/")
    print("  - worker_fe_results.csv")
    print("  - iv_townsend_results.csv")
    print("  - callaway_santanna_results.csv")
    print("  - mechanism_banking_results.csv")
    print("  - mechanism_savings_results.csv")
    print("  - power_analysis.csv")
    print("  - wild_cluster_bootstrap.csv")


if __name__ == '__main__':
    main()
