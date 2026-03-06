"""
01_clean_rlms.py
Build RLMS individual panel for labor informality & welfare cost analysis.
Merges HH-level consumption with IND-level labor/informality variables.

Requires: RLMS IND and HH data files (Stata .dta format) on OneDrive.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
ONEDRIVE_BASE = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-FloridaInstituteofTechnology"
    "/_Research/Sanctions/Working santctions"
)
IND_PATH = os.path.join(ONEDRIVE_BASE, "IND", "RLMS_IND_1994_2023_eng_dta.dta")
HH_PATH = os.path.join(ONEDRIVE_BASE, "HH", "RLMS_HH_1994_2023_eng_dta.dta")

# Credit market project — PSU→oblast mapping
CREDIT_BASE = os.path.expanduser(
    "~/Library/CloudStorage/Dropbox/Credit market (1)"
)
PSU_PATH = os.path.join(CREDIT_BASE, "Data", "RLMSsites_pubuse.dta")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
NPD_PATH = os.path.join(PROJECT_DIR, "data", "npd_rollout_by_region.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# VARIABLE LISTS
# ============================================================
IND_VARS = [
    # Identifiers
    'idind', 'id_h', 'year', 'psu', 'region',
    # Demographics
    'h5', 'h6', 'age', 'educ', 'marst',
    # Employment
    'j1', 'j4_1', 'j8', 'j10', 'j11',
    # Informality indicators
    'j11_1',        # Officially employed? (1=yes, 2=no)
    'j10_1',        # % of wages officially registered
    'j10_3',        # All money transferred officially? (post-2015)
    'j13',          # Number of employees in enterprise (firm size)
    'j59_1',        # By contract? — informal economic activity
]

HH_VARS = [
    'id_h', 'id_w', 'psu', 'region',
    # Consumption — food
    'e4',           # Cost of eating, last 30 days
    # Consumption — clothing (3 months)
    'e5',           # Clothing/shoes bought? (1=yes, 2=no)
    'e6_1',         # Clothing/shoes cost — adult (3 months)
    'e6_2',         # Clothing/shoes cost — child (3 months)
    # Consumption — utilities
    'e10',          # Rent/utilities paid for? (1=yes, 2=no)
    'e11',          # Rent/utilities cost (last 30 days)
    # Savings
    'e16',          # Saved money (1=yes, 2=no)
    'e17',          # Amount saved
    # Transfers
    'e18',          # Given money/goods to others? (1=yes, 2=no)
]

MISSING_CODES = [99999990, 99999991, 99999992, 99999993,
                 99999994, 99999995, 99999996, 99999997,
                 99999998, 99999999]

# ter (4-digit Goskomstat) → NPD region name crosswalk
# Maps RLMS oblast codes to names used in npd_rollout_by_region.csv
TER_TO_NPD_REGION = {
    1101: 'Altai Krai',
    1103: 'Krasnodar Krai',
    1104: 'Krasnoyarsk Krai',
    1105: 'Primorsky Krai',
    1107: 'Stavropol Krai',
    1110: 'Amur Oblast',
    1118: 'Volgograd Oblast',
    1122: 'Nizhny Novgorod Oblast',
    1128: 'Tver Oblast',
    1129: 'Kaluga Oblast',
    1137: 'Kurgan Oblast',
    1140: 'St. Petersburg',
    1141: 'Leningrad Oblast',
    1142: 'Lipetsk Oblast',
    1145: 'Moscow',
    1146: 'Moscow Oblast',
    1150: 'Novosibirsk Oblast',
    1153: 'Orenburg Oblast',
    1156: 'Penza Oblast',
    1157: 'Perm Krai',
    1160: 'Rostov Oblast',
    1163: 'Saratov Oblast',
    1166: 'Smolensk Oblast',
    1168: 'Tambov Oblast',
    1169: 'Tomsk Oblast',
    1170: 'Tula Oblast',
    1171: 'Tyumen Oblast',
    1175: 'Chelyabinsk Oblast',
    1183: 'Kabardino-Balkaria',
    1187: 'Republic of Komi',
    1192: 'Republic of Tatarstan',
    1194: 'Udmurt Republic',
    1197: 'Chuvash Republic',
}

YEAR_MIN, YEAR_MAX = 2010, 2023


def to_numeric_safe(series):
    """Convert categorical or other types to numeric safely."""
    return pd.to_numeric(series, errors='coerce')


def load_ind(path):
    """Load and clean individual-level RLMS data."""
    print(f"Loading IND file: {path}")

    # Check which requested variables exist
    available_vars = []
    try:
        reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
        all_cols = list(reader.variable_labels().keys())
        available_vars = [v for v in IND_VARS if v in all_cols]
        missing_vars = [v for v in IND_VARS if v not in all_cols]
        if missing_vars:
            print(f"  Variables not found in data: {missing_vars}")
    except Exception:
        available_vars = IND_VARS

    # CRITICAL: convert_categoricals=False avoids crash on duplicate value labels
    df = pd.read_stata(path, columns=available_vars, convert_categoricals=False)
    print(f"  Loaded {len(df):,} observations, {len(df.columns)} variables")

    # Filter years
    df['year'] = to_numeric_safe(df['year'])
    df = df[df['year'].between(YEAR_MIN, YEAR_MAX)].copy()
    print(f"  After filtering {YEAR_MIN}-{YEAR_MAX}: {len(df):,} observations")

    # Convert categorical columns to numeric
    numeric_cols = ['j10', 'j8', 'j10_1', 'j13', 'h6', 'age',
                    'j1', 'j4_1', 'j11', 'j11_1', 'j10_3',
                    'j59_1', 'h5', 'educ', 'marst']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # Clean missing codes
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col].isin(MISSING_CODES), col] = np.nan

    # --- Demographics ---
    if 'h5' in df.columns:
        df['female'] = (df['h5'] == 2).astype(int)
    if 'h6' in df.columns:
        df['birth_year'] = df['h6']

    # --- Employment ---
    if 'j1' in df.columns:
        df['employed'] = (df['j1'] == 1).astype(int)
    if 'j10' in df.columns:
        df['wage'] = df['j10'].where(df['j10'] > 0)
        df['ln_wage'] = np.log(df['wage'])
    if 'j8' in df.columns:
        df['hours_worked'] = df['j8'].where(df['j8'].between(1, 100))

    # --- Informality classification ---
    # 1. Not officially employed (legalistic definition)
    if 'j11_1' in df.columns:
        df['no_contract'] = (df['j11_1'] == 2).astype(float)
        df.loc[df['j11_1'].isna(), 'no_contract'] = np.nan

    # 2. Envelope wages (% paid officially)
    if 'j10_1' in df.columns:
        df['pct_official_wage'] = df['j10_1'].where(df['j10_1'].between(0, 100))
        df['has_envelope_wage'] = (df['pct_official_wage'] < 100).astype(float)
        df.loc[df['pct_official_wage'].isna(), 'has_envelope_wage'] = np.nan

    # Alternative envelope wage variable (post-2015)
    if 'j10_3' in df.columns:
        if 'has_envelope_wage' not in df.columns:
            df['has_envelope_wage'] = np.nan
        fill_mask = df['has_envelope_wage'].isna() & df['j10_3'].notna()
        if fill_mask.any():
            df.loc[fill_mask & (df['j10_3'] == 2), 'has_envelope_wage'] = 1.0
            df.loc[fill_mask & (df['j10_3'] == 1), 'has_envelope_wage'] = 0.0

    # 3. Firm size
    if 'j13' in df.columns:
        df['firm_size'] = df['j13'].where(df['j13'] > 0)
        df['small_firm'] = (df['firm_size'] <= 5).astype(float)
        df.loc[df['firm_size'].isna(), 'small_firm'] = np.nan

    # 4. Employment type classification: formal wage / informal wage / self-employed
    #    j4_1: employment type (1=regular employee, 2=temp, 3=casual, 4=self-employed, etc.)
    #    j11: employment status
    df['emp_type'] = np.nan
    employed_mask = df.get('employed', pd.Series(0, index=df.index)) == 1

    if 'j4_1' in df.columns:
        # Self-employed: j4_1 in (4, 5) — own business, freelance
        self_emp_mask = employed_mask & df['j4_1'].isin([4, 5])
        df.loc[self_emp_mask, 'emp_type'] = 'self_employed'

        # Wage workers: distinguish formal vs informal
        wage_mask = employed_mask & ~df['j4_1'].isin([4, 5])
        if 'j11_1' in df.columns:
            df.loc[wage_mask & (df['j11_1'] == 1), 'emp_type'] = 'formal_wage'
            df.loc[wage_mask & (df['j11_1'] == 2), 'emp_type'] = 'informal_wage'
            # If j11_1 missing but employed and not self-employed, leave as NaN
        else:
            df.loc[wage_mask, 'emp_type'] = 'formal_wage'  # default if no contract info

    # 5. Composite informality indicator
    informality_cols = [c for c in ['no_contract', 'has_envelope_wage']
                        if c in df.columns]
    if informality_cols:
        df['informal'] = df[informality_cols].max(axis=1)

    return df


def load_hh(path):
    """Load and clean household-level RLMS data (consumption variables)."""
    print(f"Loading HH file: {path}")

    available_vars = []
    try:
        reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
        all_cols = list(reader.variable_labels().keys())
        available_vars = [v for v in HH_VARS if v in all_cols]
        missing_vars = [v for v in HH_VARS if v not in all_cols]
        if missing_vars:
            print(f"  Variables not found in data: {missing_vars}")
    except Exception:
        available_vars = HH_VARS

    df = pd.read_stata(path, columns=available_vars, convert_categoricals=False)
    print(f"  Loaded {len(df):,} observations")

    # Convert all numeric-looking columns
    for col in df.columns:
        if col not in ['id_h', 'psu', 'region']:
            df[col] = to_numeric_safe(df[col])

    # Create year from wave identifier
    if 'id_w' in df.columns and 'year' not in df.columns:
        wave_year = {
            19: 2010, 20: 2011, 21: 2012, 22: 2013, 23: 2014,
            24: 2015, 25: 2016, 26: 2017, 27: 2018, 28: 2019,
            29: 2020, 30: 2021, 31: 2022, 32: 2023
        }
        df['year'] = df['id_w'].map(wave_year)
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
    elif 'year' in df.columns:
        df['year'] = to_numeric_safe(df['year'])

    df = df[df['year'].between(YEAR_MIN, YEAR_MAX)].copy()
    print(f"  After filtering {YEAR_MIN}-{YEAR_MAX}: {len(df):,} observations")

    # Clean missing codes
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col].isin(MISSING_CODES), col] = np.nan

    # --- Construct consumption aggregates ---
    # Food: e4 is monthly (last 30 days)
    if 'e4' in df.columns:
        df['food_exp'] = df['e4'].where(df['e4'] > 0)

    # Clothing: e6_1 (adult) + e6_2 (child), 3-month recall → monthly
    # e5 is yes/no indicator; e6 is empty in recent waves
    clothing_cols = [c for c in ['e6_1', 'e6_2'] if c in df.columns]
    if clothing_cols:
        for c in clothing_cols:
            df[c] = df[c].where(df[c] > 0)
        df['clothing_exp'] = df[clothing_cols].sum(axis=1, min_count=1) / 3.0

    # Utilities: e11 (rent/utilities cost, 30 days)
    # e10 is yes/no indicator
    if 'e11' in df.columns:
        df['utility_exp'] = df['e11'].where(df['e11'] > 0)

    # Total consumption (monthly): food + clothing + utilities
    cons_cols = [c for c in ['food_exp', 'clothing_exp', 'utility_exp']
                 if c in df.columns]
    if cons_cols:
        df['total_consumption'] = df[cons_cols].sum(axis=1, min_count=1)
        df['ln_consumption'] = np.log(df['total_consumption'].where(
            df['total_consumption'] > 0))

    # Savings
    if 'e16' in df.columns:
        df['saved_money'] = (df['e16'] == 1).astype(float)
        df.loc[df['e16'].isna(), 'saved_money'] = np.nan
    if 'e17' in df.columns:
        df['savings_amount'] = df['e17'].where(df['e17'] > 0)

    # Transfers given (e18 is yes/no indicator)
    if 'e18' in df.columns:
        df['gave_transfers'] = (df['e18'] == 1).astype(float)
        df.loc[df['e18'].isna(), 'gave_transfers'] = np.nan

    # Rename to avoid conflicts in merge (prefix hh_ for raw vars)
    raw_hh = [c for c in ['e4', 'e5', 'e6_1', 'e6_2', 'e10', 'e11',
                           'e16', 'e17', 'e18']
              if c in df.columns]
    rename_cols = {c: f'hh_{c}' for c in raw_hh}
    df = df.rename(columns=rename_cols)

    return df


def load_psu_mapping():
    """Load PSU → oblast (ter) mapping from credit market project."""
    if not os.path.exists(PSU_PATH):
        print(f"WARNING: PSU mapping not found at {PSU_PATH}")
        return None

    psu = pd.read_stata(PSU_PATH)
    mapping = psu.groupby('psu').agg({
        'ter': 'first',
        'ter_lab': 'first',
        'okrug': 'first'
    }).reset_index()

    mapping['psu'] = to_numeric_safe(mapping['psu'])
    print(f"  PSU mapping: {len(mapping)} PSUs → {mapping['ter'].nunique()} oblasts")
    return mapping


def load_npd_rollout():
    """Load NPD rollout dates and create region → rollout year mapping."""
    if not os.path.exists(NPD_PATH):
        print(f"WARNING: NPD rollout file not found at {NPD_PATH}")
        return None

    npd = pd.read_csv(NPD_PATH)
    npd['npd_start_date'] = pd.to_datetime(npd['npd_start_date'])
    npd['npd_year'] = npd['npd_start_date'].dt.year
    print(f"  NPD rollout: {len(npd)} regions, "
          f"waves {npd['wave'].unique()}")
    return npd


def merge_ind_hh(ind_df, hh_df):
    """Merge individual and household data on id_h + year."""
    print("Merging IND and HH data...")
    merged = ind_df.merge(hh_df, on=['id_h', 'year'], how='left',
                          suffixes=('', '_hh'))
    print(f"  Merged panel: {len(merged):,} observations, "
          f"{merged['idind'].nunique():,} individuals, "
          f"{merged['year'].nunique()} years")
    return merged


def add_region_info(df, psu_mapping):
    """Add oblast-level identifiers via PSU → ter mapping."""
    if psu_mapping is None:
        return df

    df['psu'] = to_numeric_safe(df['psu'])
    df = df.merge(psu_mapping[['psu', 'ter', 'ter_lab', 'okrug']],
                  on='psu', how='left')
    matched = df['ter'].notna().sum()
    print(f"  PSU → Oblast match: {matched:,} of {len(df):,} "
          f"({matched/len(df)*100:.1f}%)")

    # Add broad RLMS region labels
    region_map = {
        1: 'Metropolitan',
        2: 'Northern/NorthWestern',
        3: 'Central/CentralBlackEarth',
        4: 'Volga-Vyatski/VolgaBasin',
        5: 'NorthCaucasian',
        6: 'Ural',
        7: 'WesternSiberian',
        8: 'EasternSiberian/FarEastern',
    }
    if 'region' in df.columns:
        df['region'] = to_numeric_safe(df['region'])
        df['region_name'] = df['region'].map(region_map)

    return df


def add_npd_treatment(df, npd_rollout):
    """Add NPD treatment indicators by mapping ter → NPD region → rollout date."""
    if npd_rollout is None or 'ter' not in df.columns:
        print("  Skipping NPD treatment (missing data)")
        return df

    # Map ter codes to NPD region names
    df['npd_region'] = df['ter'].map(TER_TO_NPD_REGION)
    matched_regions = df['npd_region'].notna().sum()
    print(f"  ter → NPD region match: {matched_regions:,} of {len(df):,} "
          f"({matched_regions/len(df)*100:.1f}%)")

    # Merge NPD rollout dates
    npd_slim = npd_rollout[['region', 'npd_start_date', 'npd_year', 'wave']].copy()
    npd_slim = npd_slim.rename(columns={'region': 'npd_region',
                                         'wave': 'npd_wave'})
    df = df.merge(npd_slim, on='npd_region', how='left')

    # Construct treatment indicators
    if 'npd_year' in df.columns:
        df['post_npd'] = (df['year'] >= df['npd_year']).astype(float)
        df.loc[df['npd_year'].isna(), 'post_npd'] = np.nan

        # Early adopter (wave 1: 4 pilots, 2019)
        df['npd_early'] = (df['npd_wave'] == '1').astype(float)
        df.loc[df['npd_wave'].isna(), 'npd_early'] = np.nan

    npd_matched = df['npd_year'].notna().sum()
    print(f"  NPD rollout matched: {npd_matched:,} obs "
          f"({npd_matched/len(df)*100:.1f}%)")

    return df


def add_consumption_growth(df):
    """Compute consumption growth (Δ ln c) within individuals over time."""
    if 'ln_consumption' not in df.columns or 'idind' not in df.columns:
        return df

    df = df.sort_values(['idind', 'year'])
    df['ln_consumption_lag'] = df.groupby('idind')['ln_consumption'].shift(1)
    df['consumption_growth'] = df['ln_consumption'] - df['ln_consumption_lag']

    # Also compute income growth for Townsend test
    if 'ln_wage' in df.columns:
        df['ln_wage_lag'] = df.groupby('idind')['ln_wage'].shift(1)
        df['wage_growth'] = df['ln_wage'] - df['ln_wage_lag']

    valid = df['consumption_growth'].notna().sum()
    print(f"  Consumption growth computed: {valid:,} obs with valid Δ ln c")

    return df


def report_summary(df):
    """Print summary statistics for the cleaned panel."""
    print("\n" + "=" * 60)
    print("PANEL SUMMARY")
    print("=" * 60)
    print(f"Observations: {len(df):,}")
    print(f"Individuals:  {df['idind'].nunique():,}")
    print(f"Households:   {df['id_h'].nunique():,}")
    print(f"Years:        {int(df['year'].min())}-{int(df['year'].max())}")
    if 'region' in df.columns:
        print(f"Regions:      {df['region'].nunique()} broad regions")
    if 'ter' in df.columns:
        print(f"Oblasts:      {df['ter'].nunique()} federal subjects")

    # Observations by year
    print("\n--- Observations by Year ---")
    year_counts = df.groupby('year').size()
    print(year_counts.to_string())

    # Consumption
    if 'total_consumption' in df.columns:
        print("\n--- Consumption (monthly, RUB) ---")
        cons_stats = df.groupby('year')['total_consumption'].agg(['mean', 'median', 'count'])
        cons_stats.columns = ['Mean', 'Median', 'N_nonmissing']
        print(cons_stats.round(0).to_string())

    # Informality
    if 'emp_type' in df.columns:
        print("\n--- Employment Type (among employed) ---")
        emp_dist = df[df['employed'] == 1].groupby('year')['emp_type'].value_counts(
            normalize=True).unstack(fill_value=0).round(3)
        print(emp_dist.to_string())

    if 'informal' in df.columns:
        print(f"\nInformality rate (overall): {df['informal'].mean():.3f}")
        print("Informality by year:")
        print(df.groupby('year')['informal'].mean().round(3).to_string())

    # Consumption growth
    if 'consumption_growth' in df.columns:
        print("\n--- Consumption Growth (Δ ln c) ---")
        cg = df.groupby('year')['consumption_growth'].agg(['mean', 'std', 'count'])
        cg.columns = ['Mean', 'Std', 'N']
        print(cg.round(4).to_string())

    # NPD coverage
    if 'npd_year' in df.columns:
        print("\n--- NPD Treatment Coverage ---")
        print(f"  Obs with NPD match: {df['npd_year'].notna().sum():,}")
        if 'post_npd' in df.columns:
            print(f"  Post-NPD observations: {(df['post_npd'] == 1).sum():,}")

    # Variable coverage
    key_vars = ['employed', 'ln_wage', 'informal', 'no_contract',
                'has_envelope_wage', 'total_consumption', 'ln_consumption',
                'consumption_growth', 'food_exp', 'clothing_exp',
                'utility_exp', 'min_living_cost', 'ter', 'npd_year']
    avail_vars = [v for v in key_vars if v in df.columns]
    print("\n--- Variable Coverage by Year ---")
    coverage = df.groupby('year')[avail_vars].apply(
        lambda x: x.notna().mean()
    ).round(3)
    print(coverage.to_string())


def main():
    # Check data files exist
    for path, label in [(IND_PATH, "IND"), (HH_PATH, "HH")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found at {path}")
            print("Please update the path or ensure OneDrive is synced.")
            sys.exit(1)

    # Load PSU mapping
    print("\n=== Loading Geographic Data ===")
    psu_mapping = load_psu_mapping()
    npd_rollout = load_npd_rollout()

    # Load and clean
    print("\n=== Loading Individual Data ===")
    ind_df = load_ind(IND_PATH)

    print("\n=== Loading Household Data ===")
    hh_df = load_hh(HH_PATH)

    # Merge IND + HH
    print("\n=== Merging Individual and Household Data ===")
    panel = merge_ind_hh(ind_df, hh_df)

    # Add region info
    print("\n=== Adding Regional Variables ===")
    panel = add_region_info(panel, psu_mapping)

    # Add NPD treatment
    print("\n=== Adding NPD Treatment ===")
    panel = add_npd_treatment(panel, npd_rollout)

    # Compute consumption growth
    print("\n=== Computing Consumption Growth ===")
    panel = add_consumption_growth(panel)

    # Report
    report_summary(panel)

    # Drop raw Stata variables to keep file manageable
    drop_cols = [c for c in panel.columns if c.startswith('hh_')]
    drop_cols += [c for c in panel.columns
                  if c.startswith('j') and c not in ['j1', 'j4_1', 'j10', 'j8', 'j13']]
    panel = panel.drop(columns=[c for c in drop_cols if c in panel.columns],
                       errors='ignore')

    # Save
    out_csv = os.path.join(DATA_DIR, 'rlms_informality_panel.csv')
    out_pkl = os.path.join(DATA_DIR, 'rlms_informality_panel.pkl')
    panel.to_csv(out_csv, index=False)
    panel.to_pickle(out_pkl)
    print(f"\nSaved CSV: {out_csv} ({os.path.getsize(out_csv) / 1e6:.1f} MB)")
    print(f"Saved PKL: {out_pkl} ({os.path.getsize(out_pkl) / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
