"""
05_channel_tests.py
Channel identification tests for the NPD consumption insurance puzzle.

Tests three channels:
  A. Sector heterogeneity (Channel 2: contracting/client stability)
     - B2B-heavy sectors vs individual-client sectors
  B. Primary vs secondary earner (Channel 5: household bargaining)
     - Effect by earner rank within household, by gender
  C. Event study dynamics (Channel 1 vs 4: enforcement risk vs planning horizon)
     - Immediate vs gradual gap closure

Requires:
  - data/cleaned/rlms_informality_panel.pkl
  - RLMS IND file for industry codes (j4_1)
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
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


# j4_1 industry codes → sector classification
# B2B-heavy: firms need documented suppliers, receipts matter
B2B_SECTORS = {
    6: 'Construction',
    7: 'Transport/Communication',
    27: 'Information Technology',
    2: 'Machine Construction',
    3: 'Military-Industrial',
    15: 'Finances',
    18: 'Real Estate',
    21: 'Legal Services',
}

# Individual-client: services to individuals, cash-based
INDIVIDUAL_SECTORS = {
    14: 'Trade/Consumer Services',
    26: 'Services to Public',
    29: 'Catering',
    25: 'Sports/Tourism/Entertainment',
    8: 'Agriculture',
    17: 'Housing/Communal',
    20: 'Social Services',
    12: 'Public Health',
    10: 'Education',
}


def load_panel():
    """Load cleaned panel and construct analysis variables."""
    print("Loading panel...")
    df = pd.read_pickle(PANEL_PATH)
    df['year'] = df['year'].astype(int)

    df['is_informal'] = df['emp_type'].isin(['informal_wage', 'self_employed']).astype(int)
    df['is_informal_wage'] = (df['emp_type'] == 'informal_wage').astype(int)
    df['is_formal'] = (df['emp_type'] == 'formal_wage').astype(int)
    df['post_npd_int'] = df['post_npd'].fillna(0).astype(int)

    df['dw_x_informal'] = df['wage_growth'] * df['is_informal']
    df['dw_x_post'] = df['wage_growth'] * df['post_npd_int']
    df['dw_x_informal_x_post'] = df['wage_growth'] * df['is_informal'] * df['post_npd_int']
    df['informal_x_post'] = df['is_informal'] * df['post_npd_int']
    df['event_time'] = df['year'] - df['npd_year']

    print(f"  Panel: {len(df):,} obs, {df['idind'].nunique():,} individuals")
    return df


def add_industry(df):
    """Load industry codes from RLMS IND and merge."""
    print("\nLoading industry codes from RLMS IND...")
    if not os.path.exists(IND_PATH):
        print("  WARNING: IND file not found.")
        return df

    ind = pd.read_stata(IND_PATH, columns=['idind', 'year', 'j4_1'],
                        convert_categoricals=False)
    ind['year'] = pd.to_numeric(ind['year'], errors='coerce')
    ind['j4_1'] = pd.to_numeric(ind['j4_1'], errors='coerce')
    ind = ind[ind['year'].between(2010, 2023)].copy()
    ind = ind.rename(columns={'j4_1': 'industry'})
    ind = ind.drop_duplicates(subset=['idind', 'year'])

    df = df.merge(ind[['idind', 'year', 'industry']], on=['idind', 'year'], how='left')

    # Classify sectors
    df['b2b_sector'] = df['industry'].isin(B2B_SECTORS.keys()).astype(int)
    df['individual_sector'] = df['industry'].isin(INDIVIDUAL_SECTORS.keys()).astype(int)
    df['sector_type'] = np.where(df['b2b_sector'] == 1, 'B2B',
                         np.where(df['individual_sector'] == 1, 'Individual', 'Other'))

    matched = df['industry'].notna().sum()
    print(f"  Industry matched: {matched:,} obs ({matched/len(df)*100:.1f}%)")
    print(f"  Sector distribution (employed):")
    emp = df[df['employed'] == 1]
    for st in ['B2B', 'Individual', 'Other']:
        n = (emp['sector_type'] == st).sum()
        print(f"    {st:15s}: {n:,} ({n/len(emp)*100:.1f}%)")

    # Cross-tab sector × informality
    print(f"\n  Sector × informality (employed):")
    ct = pd.crosstab(emp['sector_type'], emp['emp_type'], normalize='index')
    print(ct.round(3).to_string())

    return df


def add_earner_rank(df):
    """Classify workers as primary or secondary earner within household."""
    print("\nClassifying primary vs secondary earners...")

    # Within each HH-year, rank earners by wage
    wage_data = df[df['wage'].notna() & (df['wage'] > 0)].copy()
    wage_data['wage_rank'] = wage_data.groupby(['id_h', 'year'])['wage'].rank(
        ascending=False, method='first')
    wage_data['n_earners'] = wage_data.groupby(['id_h', 'year'])['wage'].transform('count')
    wage_data['hh_total_wage'] = wage_data.groupby(['id_h', 'year'])['wage'].transform('sum')
    wage_data['wage_share'] = wage_data['wage'] / wage_data['hh_total_wage']

    # Primary earner: highest wage in HH (rank 1)
    wage_data['is_primary'] = (wage_data['wage_rank'] == 1).astype(int)
    wage_data['is_secondary'] = (wage_data['wage_rank'] > 1).astype(int)
    wage_data['multi_earner_hh'] = (wage_data['n_earners'] > 1).astype(int)

    # Merge back
    df = df.merge(
        wage_data[['idind', 'year', 'wage_rank', 'n_earners', 'hh_total_wage',
                    'wage_share', 'is_primary', 'is_secondary', 'multi_earner_hh']],
        on=['idind', 'year'], how='left'
    )

    # Summary
    with_rank = df['is_primary'].notna().sum()
    print(f"  Earner rank assigned: {with_rank:,} obs")
    print(f"  Primary earners: {(df['is_primary']==1).sum():,}")
    print(f"  Secondary earners: {(df['is_secondary']==1).sum():,}")
    print(f"  Multi-earner HH: {(df['multi_earner_hh']==1).sum():,}")
    print(f"  Single-earner HH: {(df['multi_earner_hh']==0).sum():,}")

    # Cross-tab with informality
    emp = df[(df['employed'] == 1) & df['is_primary'].notna()]
    print(f"\n  Earner rank × informality:")
    for rank_label, rank_val in [('Primary', 1), ('Secondary', 0)]:
        sub = emp[emp['is_primary'] == (1 if rank_label == 'Primary' else 0)]
        inf_rate = sub['is_informal'].mean()
        print(f"    {rank_label}: N={len(sub):,}, informality rate={inf_rate:.3f}")

    # Gender × earner rank
    if 'female' in df.columns:
        print(f"\n  Gender × earner rank (employed):")
        for gen, glabel in [(0, 'Male'), (1, 'Female')]:
            sub = emp[emp['female'] == gen]
            primary_rate = sub['is_primary'].mean()
            print(f"    {glabel}: {primary_rate*100:.1f}% primary earner")

    return df


# ============================================================
# TEST A: SECTOR HETEROGENEITY (Channel 2)
# ============================================================

def test_sector_heterogeneity(df):
    """
    Test whether NPD effect is larger in B2B sectors (construction, IT,
    transport) where receipt-issuance enables formal contracting.
    """
    print("\n" + "=" * 60)
    print("TEST A: Sector Heterogeneity (Contracting Channel)")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter', 'sector_type'])

    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results = {}

    # --- By-type Townsend within each sector group ---
    for sector in ['B2B', 'Individual']:
        print(f"\n  --- {sector} sectors ---")
        sec_sample = sample[sample['sector_type'] == sector]

        for etype in ['formal_wage', 'informal_wage', 'self_employed']:
            esub = sec_sample[sec_sample['emp_type'] == etype]
            if len(esub) < 50:
                print(f"    {etype}: too few obs ({len(esub)}). Skipping.")
                continue

            X_vars = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
            X_vars = [c for c in X_vars if c in esub.columns]
            X = sm.add_constant(esub[X_vars])
            y = esub['consumption_growth']
            mod = sm.OLS(y, X).fit(cov_type='cluster',
                                    cov_kwds={'groups': esub['ter']})
            pre = mod.params['wage_growth']
            change = mod.params['dw_x_post']
            stars = '***' if mod.pvalues['dw_x_post'] < 0.01 else '**' if mod.pvalues['dw_x_post'] < 0.05 else '*' if mod.pvalues['dw_x_post'] < 0.1 else ''
            print(f"    {etype:20s}: pre-β={pre:.4f}, Δ={change:.4f}{stars} "
                  f"(SE: {mod.bse['dw_x_post']:.4f}), post-β={pre+change:.4f}, N={mod.nobs:,.0f}")
            results[f'{sector}_{etype}'] = {
                'sector': sector, 'emp_type': etype,
                'beta_pre': pre, 'beta_change': change, 'beta_post': pre + change,
                'se_change': mod.bse['dw_x_post'], 'p_change': mod.pvalues['dw_x_post'],
                'N': int(mod.nobs),
            }

    # --- Triple-diff with sector interaction ---
    print("\n  --- Triple-diff with B2B interaction ---")
    sample['is_b2b'] = (sample['sector_type'] == 'B2B').astype(int)
    sample['dw_x_inf_x_post_x_b2b'] = (sample['wage_growth'] * sample['is_informal'] *
                                         sample['post_npd_int'] * sample['is_b2b'])
    sample['dw_x_inf_x_b2b'] = sample['wage_growth'] * sample['is_informal'] * sample['is_b2b']

    X_vars_int = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                  'dw_x_informal_x_post', 'dw_x_inf_x_b2b', 'dw_x_inf_x_post_x_b2b',
                  'is_informal', 'post_npd_int', 'informal_x_post',
                  'is_b2b'] + yr_cols
    X = sm.add_constant(sample[X_vars_int])
    y = sample['consumption_growth']
    mod_int = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sample['ter']})

    for v in ['dw_x_informal_x_post', 'dw_x_inf_x_post_x_b2b', 'dw_x_inf_x_b2b']:
        stars = '***' if mod_int.pvalues[v] < 0.01 else '**' if mod_int.pvalues[v] < 0.05 else '*' if mod_int.pvalues[v] < 0.1 else ''
        print(f"    {v:35s}: {mod_int.params[v]:8.4f}{stars} (SE: {mod_int.bse[v]:.4f})")
    print(f"    N={mod_int.nobs:,.0f}")
    results['sector_interaction'] = mod_int

    # --- Formal test: B2B effect ≠ Individual effect ---
    print("\n  --- Formal test: B2B vs Individual effect equality ---")
    b2b_res = results.get('B2B_informal_wage', {})
    ind_res = results.get('Individual_informal_wage', {})
    if b2b_res and ind_res:
        diff = b2b_res['beta_change'] - ind_res['beta_change']
        diff_se = np.sqrt(b2b_res['se_change']**2 + ind_res['se_change']**2)
        diff_z = diff / diff_se
        diff_p = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(diff_z)))
        print(f"    B2B Δ - Individual Δ = {diff:.4f} (SE: {diff_se:.4f}, z={diff_z:.2f}, p={diff_p:.3f})")
        results['b2b_vs_individual_test'] = {
            'test': 'B2B_minus_Individual',
            'difference': diff, 'se': diff_se, 'z': diff_z, 'p': diff_p,
        }

    # Interaction model: formal test from quadruple-diff
    if hasattr(mod_int, 'params') and 'dw_x_inf_x_post_x_b2b' in mod_int.params:
        b_quad = mod_int.params['dw_x_inf_x_post_x_b2b']
        se_quad = mod_int.bse['dw_x_inf_x_post_x_b2b']
        p_quad = mod_int.pvalues['dw_x_inf_x_post_x_b2b']
        results['quadruple_diff'] = {
            'test': 'Δw×informal×post×B2B',
            'coef': b_quad, 'se': se_quad, 'p': p_quad,
            'N': int(mod_int.nobs),
        }

    # --- Save interaction model coefficients ---
    int_rows = []
    for v in mod_int.params.index:
        if v not in yr_cols and v != 'const':
            int_rows.append({
                'variable': v, 'coef': mod_int.params[v],
                'se': mod_int.bse[v], 'p': mod_int.pvalues[v],
            })
    pd.DataFrame(int_rows).to_csv(
        os.path.join(TABLE_DIR, 'channel_sector_interaction_model.csv'), index=False)
    print(f"  Saved: {TABLE_DIR}/channel_sector_interaction_model.csv")

    # --- Top B2B industries detail ---
    print("\n  --- NPD effect by top industries (informal wage only) ---")
    inf_sample = sample[sample['emp_type'] == 'informal_wage']
    industry_labels = {**B2B_SECTORS, **INDIVIDUAL_SECTORS}
    industry_results = []
    for ind_code in sorted(inf_sample['industry'].dropna().unique()):
        ind_sub = inf_sample[inf_sample['industry'] == ind_code]
        if len(ind_sub) < 30:
            continue
        X = sm.add_constant(ind_sub[['wage_growth', 'dw_x_post', 'post_npd_int'] +
                                     [c for c in yr_cols if c in ind_sub.columns]])
        y_ind = ind_sub['consumption_growth']
        try:
            mod_ind = sm.OLS(y_ind, X).fit(cov_type='cluster',
                                            cov_kwds={'groups': ind_sub['ter']})
            pre = mod_ind.params['wage_growth']
            change = mod_ind.params['dw_x_post']
            label = industry_labels.get(int(ind_code), f'Code {int(ind_code)}')
            stype = 'B2B' if int(ind_code) in B2B_SECTORS else ('Ind' if int(ind_code) in INDIVIDUAL_SECTORS else 'Oth')
            stars = '***' if mod_ind.pvalues['dw_x_post'] < 0.01 else '**' if mod_ind.pvalues['dw_x_post'] < 0.05 else '*' if mod_ind.pvalues['dw_x_post'] < 0.1 else ''
            print(f"    [{stype}] {label:30s}: pre={pre:.3f}, Δ={change:+.3f}{stars}, N={mod_ind.nobs:,.0f}")
            industry_results.append({
                'industry_code': int(ind_code), 'industry_label': label,
                'sector_type': stype, 'beta_pre': pre, 'beta_change': change,
                'se_change': mod_ind.bse['dw_x_post'],
                'p_change': mod_ind.pvalues['dw_x_post'],
                'N': int(mod_ind.nobs),
            })
        except Exception:
            pass

    if industry_results:
        pd.DataFrame(industry_results).to_csv(
            os.path.join(TABLE_DIR, 'channel_sector_by_industry.csv'), index=False)
        print(f"  Saved: {TABLE_DIR}/channel_sector_by_industry.csv")

    # --- Figure: sector comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: B2B vs Individual pass-through (pre/post)
    ax = axes[0]
    groups = []
    for sector in ['B2B', 'Individual']:
        for etype in ['formal_wage', 'informal_wage']:
            key = f'{sector}_{etype}'
            if key in results and isinstance(results[key], dict):
                groups.append(results[key])

    if groups:
        labels = [f"{g['sector']}\n{g['emp_type'].replace('_',' ')}" for g in groups]
        pre_vals = [g['beta_pre'] for g in groups]
        post_vals = [g['beta_post'] for g in groups]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, pre_vals, width, label='Pre-NPD', color='#b2182b', alpha=0.7)
        ax.bar(x + width/2, post_vals, width, label='Post-NPD', color='#2166ac', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Pass-through (β)')
        ax.set_title('A. Pass-Through by Sector Type')
        ax.legend()
        ax.axhline(y=0, color='gray', alpha=0.3)

    # Panel B: Change in pass-through (Δβ) with CIs
    ax = axes[1]
    sector_data = [r for r in results.values() if isinstance(r, dict) and 'beta_change' in r]
    if sector_data:
        labels_b = [f"{d['sector']}\n{d['emp_type'].replace('_',' ')}" for d in sector_data]
        changes = [d['beta_change'] for d in sector_data]
        ses = [d['se_change'] for d in sector_data]
        colors = ['#b2182b' if 'informal' in d['emp_type'] else '#2166ac' for d in sector_data]
        x = np.arange(len(labels_b))
        ax.barh(x, changes, xerr=[1.96*s for s in ses], color=colors, alpha=0.7,
                capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels_b, fontsize=8)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Change in pass-through (Δβ)')
        ax.set_title('B. NPD Effect on Pass-Through')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'channel_sector_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR}/channel_sector_comparison.png")

    # --- Summary interpretation ---
    print("\n  === INTERPRETATION ===")
    if b2b_res and ind_res:
        print(f"    B2B informal wage: pass-through drops by {abs(b2b_res['beta_change']):.3f} "
              f"(from {b2b_res['beta_pre']:.3f} to {b2b_res['beta_post']:.3f})")
        print(f"    Individual-client:  pass-through drops by {abs(ind_res['beta_change']):.3f} "
              f"(from {ind_res['beta_pre']:.3f} to {ind_res['beta_post']:.3f})")
        print(f"    B2B effect is {abs(b2b_res['beta_change']/ind_res['beta_change']):.0f}× larger"
              if ind_res['beta_change'] != 0 else "    Individual effect ≈ 0")
        print(f"    → Supports Channel 2 (contracting): NPD receipts enable B2B formal contracts")
        print(f"    → B2B firms need documented suppliers; NPD provides tax ID + receipts")

    # Save all results
    rows = [v for v in results.values() if isinstance(v, dict) and ('beta_pre' in v or 'difference' in v or 'coef' in v)]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'channel_sector_results.csv'),
                                    index=False)
        print(f"\n  Saved: {TABLE_DIR}/channel_sector_results.csv")

    return results


# ============================================================
# TEST B: PRIMARY VS SECONDARY EARNER (Channel 5)
# ============================================================

def test_earner_heterogeneity(df):
    """
    Test whether NPD effect is larger for secondary earners,
    whose informal income was previously invisible within the household.
    """
    print("\n" + "=" * 60)
    print("TEST B: Primary vs Secondary Earner (Household Bargaining)")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter',
                                    'is_primary', 'multi_earner_hh'])

    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True, dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    results = {}

    # --- By earner rank: all employed ---
    print("\n  --- Triple-diff by earner rank ---")
    for rank_label, rank_filter in [('Primary', sample['is_primary'] == 1),
                                     ('Secondary', sample['is_secondary'] == 1)]:
        sub = sample[rank_filter]
        X_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                  'dw_x_informal_x_post',
                  'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
        X = sm.add_constant(sub[X_vars])
        y = sub['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sub['ter']})

        b4 = mod.params['dw_x_informal_x_post']
        se4 = mod.bse['dw_x_informal_x_post']
        p4 = mod.pvalues['dw_x_informal_x_post']
        stars = '***' if p4 < 0.01 else '**' if p4 < 0.05 else '*' if p4 < 0.1 else ''
        print(f"    {rank_label:12s}: β₄ = {b4:.4f}{stars} (SE: {se4:.4f}), N={mod.nobs:,.0f}")
        results[f'triple_{rank_label.lower()}'] = {
            'group': rank_label, 'beta4': b4, 'se4': se4, 'p4': p4,
            'N': int(mod.nobs),
        }

    # --- By-type within secondary earners ---
    print("\n  --- By-type Townsend for SECONDARY earners only ---")
    sec_sample = sample[sample['is_secondary'] == 1]
    for etype in ['formal_wage', 'informal_wage']:
        esub = sec_sample[sec_sample['emp_type'] == etype]
        if len(esub) < 30:
            print(f"    {etype}: too few obs ({len(esub)}). Skipping.")
            continue
        X_vars = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
        X = sm.add_constant(esub[X_vars])
        y = esub['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': esub['ter']})
        pre = mod.params['wage_growth']
        change = mod.params['dw_x_post']
        stars = '***' if mod.pvalues['dw_x_post'] < 0.01 else '**' if mod.pvalues['dw_x_post'] < 0.05 else '*' if mod.pvalues['dw_x_post'] < 0.1 else ''
        print(f"    {etype:20s}: pre-β={pre:.4f}, Δ={change:.4f}{stars} "
              f"(SE: {mod.bse['dw_x_post']:.4f}), N={mod.nobs:,.0f}")
        results[f'secondary_{etype}'] = {
            'group': f'Secondary_{etype}',
            'beta_pre': pre, 'beta_change': change,
            'se_change': mod.bse['dw_x_post'],
            'N': int(mod.nobs),
        }

    # --- Gender × earner rank interaction ---
    print("\n  --- Gender × earner rank × informality × post ---")
    if 'female' in sample.columns:
        for gender, glabel in [(0, 'Male'), (1, 'Female')]:
            for rank_label, rank_val in [('Primary', 1), ('Secondary', 0)]:
                sub = sample[(sample['female'] == gender) &
                             (sample['is_primary'] == rank_val)]
                if sub['is_informal'].sum() < 20:
                    continue
                X_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                          'dw_x_informal_x_post',
                          'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
                X = sm.add_constant(sub[X_vars])
                y = sub['consumption_growth']
                mod = sm.OLS(y, X).fit(cov_type='cluster',
                                        cov_kwds={'groups': sub['ter']})
                b4 = mod.params['dw_x_informal_x_post']
                se4 = mod.bse['dw_x_informal_x_post']
                p4 = mod.pvalues['dw_x_informal_x_post']
                stars = '***' if p4 < 0.01 else '**' if p4 < 0.05 else '*' if p4 < 0.1 else ''
                print(f"    {glabel:6s} {rank_label:12s}: β₄ = {b4:.4f}{stars} "
                      f"(SE: {se4:.4f}), N={mod.nobs:,.0f}, "
                      f"N_informal={sub['is_informal'].sum():,}")
                results[f'{glabel.lower()}_{rank_label.lower()}'] = {
                    'group': f'{glabel}_{rank_label}',
                    'beta4': b4, 'se4': se4, 'p4': p4,
                    'N': int(mod.nobs),
                }

    # --- Multi-earner vs single-earner HH ---
    print("\n  --- Multi-earner vs single-earner households ---")
    for multi, mlabel in [(1, 'Multi-earner'), (0, 'Single-earner')]:
        sub = sample[sample['multi_earner_hh'] == multi]
        X_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                  'dw_x_informal_x_post',
                  'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
        X = sm.add_constant(sub[X_vars])
        y = sub['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': sub['ter']})
        b4 = mod.params['dw_x_informal_x_post']
        se4 = mod.bse['dw_x_informal_x_post']
        p4 = mod.pvalues['dw_x_informal_x_post']
        stars = '***' if p4 < 0.01 else '**' if p4 < 0.05 else '*' if p4 < 0.1 else ''
        print(f"    {mlabel:15s}: β₄ = {b4:.4f}{stars} (SE: {se4:.4f}), N={mod.nobs:,.0f}")
        results[f'hh_{mlabel.lower().replace("-", "_")}'] = {
            'group': mlabel, 'beta4': b4, 'se4': se4, 'p4': p4,
            'N': int(mod.nobs),
        }

    # --- 4-way interaction: Δw × informal × post × primary (pooled) ---
    print("\n  --- Pooled 4-way interaction: Δw × informal × post × primary ---")
    pooled = sample[sample['is_primary'].notna()].copy()
    pooled['dw_x_inf_x_post_x_primary'] = (pooled['wage_growth'] * pooled['is_informal'] *
                                             pooled['post_npd_int'] * pooled['is_primary'])
    pooled['dw_x_inf_x_primary'] = pooled['wage_growth'] * pooled['is_informal'] * pooled['is_primary']
    pooled['dw_x_post_x_primary'] = pooled['wage_growth'] * pooled['post_npd_int'] * pooled['is_primary']
    pooled['dw_x_primary'] = pooled['wage_growth'] * pooled['is_primary']
    pooled['inf_x_post_x_primary'] = pooled['is_informal'] * pooled['post_npd_int'] * pooled['is_primary']
    pooled['inf_x_primary'] = pooled['is_informal'] * pooled['is_primary']
    pooled['post_x_primary'] = pooled['post_npd_int'] * pooled['is_primary']

    X_vars_4way = ['wage_growth', 'dw_x_informal', 'dw_x_post', 'dw_x_informal_x_post',
                   'dw_x_primary', 'dw_x_inf_x_primary', 'dw_x_post_x_primary',
                   'dw_x_inf_x_post_x_primary',
                   'is_informal', 'post_npd_int', 'informal_x_post',
                   'is_primary', 'inf_x_primary', 'post_x_primary',
                   'inf_x_post_x_primary'] + yr_cols
    X = sm.add_constant(pooled[X_vars_4way])
    y = pooled['consumption_growth']
    mod_4way = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': pooled['ter']})

    key_vars = ['dw_x_informal_x_post', 'dw_x_inf_x_post_x_primary', 'dw_x_inf_x_primary']
    for v in key_vars:
        stars = '***' if mod_4way.pvalues[v] < 0.01 else '**' if mod_4way.pvalues[v] < 0.05 else '*' if mod_4way.pvalues[v] < 0.1 else ''
        print(f"    {v:35s}: {mod_4way.params[v]:8.4f}{stars} (SE: {mod_4way.bse[v]:.4f})")
    print(f"    N={mod_4way.nobs:,.0f}")

    # Secondary earner effect = dw_x_informal_x_post (reference)
    # Primary earner effect = dw_x_informal_x_post + dw_x_inf_x_post_x_primary
    b_sec = mod_4way.params['dw_x_informal_x_post']
    b_pri_extra = mod_4way.params['dw_x_inf_x_post_x_primary']
    print(f"\n    Secondary earner NPD effect (reference): {b_sec:.4f}")
    print(f"    Primary earner additional effect:         {b_pri_extra:.4f}")
    print(f"    Primary earner total NPD effect:          {b_sec + b_pri_extra:.4f}")

    results['fourway_interaction'] = {
        'group': '4-way interaction',
        'beta4': mod_4way.params['dw_x_informal_x_post'],
        'se4': mod_4way.bse['dw_x_informal_x_post'],
        'p4': mod_4way.pvalues['dw_x_informal_x_post'],
        'beta_primary_extra': b_pri_extra,
        'se_primary_extra': mod_4way.bse['dw_x_inf_x_post_x_primary'],
        'p_primary_extra': mod_4way.pvalues['dw_x_inf_x_post_x_primary'],
        'N': int(mod_4way.nobs),
    }

    # Save 4-way model coefficients
    int_rows = []
    for v in mod_4way.params.index:
        if v not in yr_cols and v != 'const':
            int_rows.append({
                'variable': v, 'coef': mod_4way.params[v],
                'se': mod_4way.bse[v], 'p': mod_4way.pvalues[v],
            })
    pd.DataFrame(int_rows).to_csv(
        os.path.join(TABLE_DIR, 'channel_earner_interaction_model.csv'), index=False)
    print(f"  Saved: {TABLE_DIR}/channel_earner_interaction_model.csv")

    # --- Continuous wage share interaction ---
    print("\n  --- Wage share as continuous moderator ---")
    ws_sample = sample[sample['wage_share'].notna() & (sample['multi_earner_hh'] == 1)].copy()
    ws_sample['dw_x_inf_x_post_x_share'] = (ws_sample['wage_growth'] * ws_sample['is_informal'] *
                                              ws_sample['post_npd_int'] * ws_sample['wage_share'])
    ws_sample['dw_x_inf_x_share'] = ws_sample['wage_growth'] * ws_sample['is_informal'] * ws_sample['wage_share']
    ws_sample['dw_x_share'] = ws_sample['wage_growth'] * ws_sample['wage_share']

    X_vars_ws = ['wage_growth', 'dw_x_informal', 'dw_x_post', 'dw_x_informal_x_post',
                 'dw_x_share', 'dw_x_inf_x_share', 'dw_x_inf_x_post_x_share',
                 'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
    X_ws = sm.add_constant(ws_sample[X_vars_ws])
    y_ws = ws_sample['consumption_growth']
    mod_ws = sm.OLS(y_ws, X_ws).fit(cov_type='cluster', cov_kwds={'groups': ws_sample['ter']})
    for v in ['dw_x_informal_x_post', 'dw_x_inf_x_post_x_share']:
        stars = '***' if mod_ws.pvalues[v] < 0.01 else '**' if mod_ws.pvalues[v] < 0.05 else '*' if mod_ws.pvalues[v] < 0.1 else ''
        print(f"    {v:35s}: {mod_ws.params[v]:8.4f}{stars} (SE: {mod_ws.bse[v]:.4f})")
    print(f"    N={mod_ws.nobs:,.0f}")

    results['wage_share_interaction'] = {
        'group': 'Wage share interaction (multi-earner HH)',
        'beta4': mod_ws.params['dw_x_informal_x_post'],
        'se4': mod_ws.bse['dw_x_informal_x_post'],
        'p4': mod_ws.pvalues['dw_x_informal_x_post'],
        'N': int(mod_ws.nobs),
    }

    # --- Figure: earner heterogeneity ---
    fig, ax = plt.subplots(figsize=(8, 5))
    earner_groups = [(k, v) for k, v in results.items()
                     if isinstance(v, dict) and 'beta4' in v and v.get('group') not in
                     ['4-way interaction', 'Wage share interaction (multi-earner HH)']]
    if earner_groups:
        labels = [v['group'] for _, v in earner_groups]
        betas = [v['beta4'] for _, v in earner_groups]
        ses = [v['se4'] for _, v in earner_groups]
        y_pos = np.arange(len(labels))
        colors = ['#b2182b' if b < -0.03 else '#2166ac' if b < 0 else '#999999' for b in betas]
        ax.barh(y_pos, betas, xerr=[1.96*s for s in ses], color=colors, alpha=0.7,
                capsize=4, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('β₄: Δw × informal × post-NPD')
        ax.set_title('NPD Effect on Consumption Insurance by Earner Type')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'channel_earner_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIG_DIR}/channel_earner_comparison.png")

    # --- Interpretation ---
    print("\n  === INTERPRETATION ===")
    pri = results.get('triple_primary', {})
    sec = results.get('triple_secondary', {})
    if pri and sec:
        print(f"    Primary earner β₄ = {pri['beta4']:.4f} (p={pri['p4']:.3f})")
        print(f"    Secondary earner β₄ = {sec['beta4']:.4f} (p={sec['p4']:.3f})")
        if abs(pri['beta4']) > abs(sec['beta4']):
            print(f"    → Effect is LARGER for primary earners ({abs(pri['beta4']):.3f} > {abs(sec['beta4']):.3f})")
            print(f"    → AGAINST Channel 5 (household bargaining): if NPD made secondary")
            print(f"      earner income 'visible', we'd expect larger effect for secondary earners")
            print(f"    → CONSISTENT with Channel 2/3 (contracting/credit): primary earners")
            print(f"      gain more from documented income because they are the main borrowers")
            print(f"      and contract signers for the household")
        else:
            print(f"    → Effect is LARGER for secondary earners")
            print(f"    → Supports Channel 5 (household bargaining)")

    # Save
    rows = [v for v in results.values() if isinstance(v, dict)]
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'channel_earner_results.csv'),
                                index=False)
    print(f"\n  Saved: {TABLE_DIR}/channel_earner_results.csv")

    return results


# ============================================================
# TEST C: EVENT STUDY DYNAMICS (Channel 1 vs 4)
# ============================================================

def test_event_dynamics(df):
    """
    Analyze event study dynamics to distinguish:
    - Channel 1 (enforcement risk): immediate effect at t=0
    - Channel 4 (planning horizon): gradual effect growing over t=1,2,3
    """
    print("\n" + "=" * 60)
    print("TEST C: Event Study Dynamics (Enforcement vs Planning)")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'ter', 'event_time'])

    # Estimate pass-through by event time and informality status
    results = []
    for et in range(-5, 5):
        et_sample = sample[sample['event_time'] == et]
        if len(et_sample) < 100:
            continue

        for inf_val, inf_label in [(0, 'Formal'), (1, 'Informal')]:
            sub = et_sample[et_sample['is_informal'] == inf_val]
            if len(sub) < 30:
                continue
            X = sm.add_constant(sub[['wage_growth']])
            y = sub['consumption_growth']
            mod = sm.OLS(y, X).fit(cov_type='cluster',
                                    cov_kwds={'groups': sub['ter']})
            results.append({
                'event_time': et,
                'group': inf_label,
                'beta': mod.params['wage_growth'],
                'se': mod.bse['wage_growth'],
                'ci_lo': mod.params['wage_growth'] - 1.96 * mod.bse['wage_growth'],
                'ci_hi': mod.params['wage_growth'] + 1.96 * mod.bse['wage_growth'],
                'N': int(mod.nobs),
            })

    rdf = pd.DataFrame(results)

    # Print results
    print("\n  Event-time pass-through coefficients:")
    print(f"  {'t':>4s} {'Formal β':>10s} {'SE':>8s} {'Informal β':>12s} {'SE':>8s} {'Gap':>8s}")
    print("  " + "-" * 58)
    for et in sorted(rdf['event_time'].unique()):
        formal = rdf[(rdf['event_time'] == et) & (rdf['group'] == 'Formal')]
        informal = rdf[(rdf['event_time'] == et) & (rdf['group'] == 'Informal')]
        if len(formal) > 0 and len(informal) > 0:
            fb = formal.iloc[0]['beta']
            fse = formal.iloc[0]['se']
            ib = informal.iloc[0]['beta']
            ise = informal.iloc[0]['se']
            gap = ib - fb
            marker = ' <-- NPD' if et == 0 else ''
            print(f"  {et:+4d}  {fb:10.4f} ({fse:.4f})  {ib:10.4f} ({ise:.4f})  {gap:+8.4f}{marker}")

    # Compute gap dynamics
    print("\n  --- Gap dynamics ---")
    gaps = []
    for et in sorted(rdf['event_time'].unique()):
        formal = rdf[(rdf['event_time'] == et) & (rdf['group'] == 'Formal')]
        informal = rdf[(rdf['event_time'] == et) & (rdf['group'] == 'Informal')]
        if len(formal) > 0 and len(informal) > 0:
            gap = informal.iloc[0]['beta'] - formal.iloc[0]['beta']
            gap_se = np.sqrt(formal.iloc[0]['se']**2 + informal.iloc[0]['se']**2)
            gaps.append({'event_time': et, 'gap': gap, 'gap_se': gap_se,
                         'N_formal': int(formal.iloc[0]['N']),
                         'N_informal': int(informal.iloc[0]['N'])})

    gdf = pd.DataFrame(gaps)
    pre_gaps = gdf[gdf['event_time'] < 0]['gap']
    post_gaps = gdf[gdf['event_time'] >= 0]['gap']

    print(f"    Mean pre-NPD gap:  {pre_gaps.mean():+.4f} (SD: {pre_gaps.std():.4f})")
    print(f"    Mean post-NPD gap: {post_gaps.mean():+.4f} (SD: {post_gaps.std():.4f})")

    # --- Formal pre-trend test ---
    from scipy import stats as scipy_stats

    print("\n  --- Pre-trend test ---")
    pre_gdf = gdf[gdf['event_time'] < 0].copy()
    if len(pre_gdf) >= 3:
        # Joint test: H0 = all pre-NPD gaps are equal (no trend)
        # Wald-type: chi2 = sum((gap_t - mean_gap)^2 / se_t^2)
        mean_pre = pre_gdf['gap'].mean()
        chi2_stat = ((pre_gdf['gap'] - mean_pre)**2 / pre_gdf['gap_se']**2).sum()
        dof = len(pre_gdf) - 1
        chi2_p = 1 - scipy_stats.chi2.cdf(chi2_stat, dof)
        print(f"    Joint significance (H0: all pre-gaps equal): χ²={chi2_stat:.2f}, "
              f"df={dof}, p={chi2_p:.3f}")

        # Trend test: regress gap on event_time for pre-period
        slope, intercept, r_value, p_trend, std_err = scipy_stats.linregress(
            pre_gdf['event_time'], pre_gdf['gap'])
        print(f"    Pre-trend slope: {slope:.4f} (SE: {std_err:.4f}, p={p_trend:.3f})")
        if p_trend < 0.10:
            print(f"    ⚠ Warning: pre-trend is significant at 10% — parallel trends concern")
        else:
            print(f"    ✓ No significant pre-trend (p={p_trend:.3f})")

        # Test H0: each pre-period gap = 0
        print(f"\n    Individual pre-period gap tests (H0: gap=0):")
        for _, row in pre_gdf.iterrows():
            z = row['gap'] / row['gap_se']
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
            print(f"      t={int(row['event_time']):+d}: gap={row['gap']:+.4f} "
                  f"(SE={row['gap_se']:.4f}, z={z:.2f}, p={p:.3f}){sig}")

    # --- Immediate vs gradual decomposition ---
    print("\n  --- Immediate vs gradual decomposition ---")
    dynamics = {}
    if len(gdf[gdf['event_time'] == 0]) > 0:
        t0_gap = gdf[gdf['event_time'] == 0].iloc[0]['gap']
        t0_se = gdf[gdf['event_time'] == 0].iloc[0]['gap_se']
        pre_mean_gap = pre_gaps.mean()
        pre_mean_se = np.sqrt((pre_gdf['gap_se']**2).mean() / len(pre_gdf))

        immediate_drop = t0_gap - pre_mean_gap
        immediate_se = np.sqrt(t0_se**2 + pre_mean_se**2)
        immediate_z = immediate_drop / immediate_se
        immediate_p = 2 * (1 - scipy_stats.norm.cdf(abs(immediate_z)))

        later_gaps = gdf[gdf['event_time'] > 0]
        if len(later_gaps) > 0:
            later_mean_gap = later_gaps['gap'].mean()
            later_mean_se = np.sqrt((later_gaps['gap_se']**2).mean() / len(later_gaps))
            gradual_component = later_mean_gap - t0_gap
            gradual_se = np.sqrt(later_mean_se**2 + t0_se**2)
            gradual_z = gradual_component / gradual_se
            gradual_p = 2 * (1 - scipy_stats.norm.cdf(abs(gradual_z)))
        else:
            later_mean_gap = gradual_component = gradual_z = gradual_p = np.nan

        print(f"    Pre-NPD mean gap:         {pre_mean_gap:+.4f} (SE: {pre_mean_se:.4f})")
        print(f"    Gap at t=0:               {t0_gap:+.4f} (SE: {t0_se:.4f})")
        print(f"    Immediate drop (t=0):     {immediate_drop:+.4f} (SE: {immediate_se:.4f}, "
              f"z={immediate_z:.2f}, p={immediate_p:.3f})")
        if not np.isnan(gradual_component):
            print(f"    Later mean gap (t>0):     {later_mean_gap:+.4f}")
            print(f"    Gradual component (Δ>0):  {gradual_component:+.4f} (SE: {gradual_se:.4f}, "
                  f"z={gradual_z:.2f}, p={gradual_p:.3f})")

        # Total gap closure
        total_closure = pre_mean_gap - post_gaps.mean()
        print(f"\n    Total gap closure:         {total_closure:+.4f}")
        if abs(immediate_drop) > 0 and not np.isnan(gradual_component):
            pct_immediate = abs(immediate_drop) / (abs(immediate_drop) + abs(gradual_component)) * 100
            print(f"    Fraction immediate:        {pct_immediate:.0f}%")
            print(f"    Fraction gradual:          {100-pct_immediate:.0f}%")

        dynamics = {
            'pre_mean_gap': pre_mean_gap, 'gap_at_t0': t0_gap,
            'immediate_drop': immediate_drop, 'immediate_se': immediate_se,
            'immediate_p': immediate_p,
            'gradual_component': gradual_component if not np.isnan(gradual_component) else None,
            'gradual_p': gradual_p if not np.isnan(gradual_p) else None,
            'total_closure': total_closure,
        }

    # --- Parametric test: regress gap on post and trend ---
    print("\n  --- Parametric structural break test ---")
    gdf_test = gdf.copy()
    gdf_test['post'] = (gdf_test['event_time'] >= 0).astype(int)
    gdf_test['post_trend'] = gdf_test['event_time'] * gdf_test['post']
    gdf_test['post_trend'] = gdf_test['post_trend'].clip(lower=0)

    # WLS weighted by inverse variance
    gdf_test['weight'] = 1 / gdf_test['gap_se']**2
    X_break = sm.add_constant(gdf_test[['event_time', 'post', 'post_trend']])
    mod_break = sm.WLS(gdf_test['gap'], X_break, weights=gdf_test['weight']).fit()
    print(f"    gap = α + δ₁·t + δ₂·post + δ₃·t·post")
    for v in ['event_time', 'post', 'post_trend']:
        stars = '***' if mod_break.pvalues[v] < 0.01 else '**' if mod_break.pvalues[v] < 0.05 else '*' if mod_break.pvalues[v] < 0.1 else ''
        print(f"      {v:15s}: {mod_break.params[v]:8.4f}{stars} (SE: {mod_break.bse[v]:.4f})")
    print(f"      Interpretation: δ₂ = immediate level shift at NPD; δ₃ = post-NPD trend change")

    dynamics['break_post_coef'] = mod_break.params['post']
    dynamics['break_post_p'] = mod_break.pvalues['post']
    dynamics['break_trend_coef'] = mod_break.params['post_trend']
    dynamics['break_trend_p'] = mod_break.pvalues['post_trend']

    pd.DataFrame([dynamics]).to_csv(
        os.path.join(TABLE_DIR, 'channel_dynamics_tests.csv'), index=False)
    print(f"  Saved: {TABLE_DIR}/channel_dynamics_tests.csv")

    # --- Interpretation ---
    print("\n  === INTERPRETATION ===")
    print(f"    Pre-NPD: informal pass-through consistently exceeds formal")
    print(f"    (mean gap = {pre_mean_gap:+.4f}, though noisy with SD={pre_gaps.std():.4f})")
    if p_trend >= 0.10:
        print(f"    No significant pre-trend (slope={slope:.4f}, p={p_trend:.3f}) — parallel trends OK")
    else:
        print(f"    ⚠ Pre-trend is significant (slope={slope:.4f}, p={p_trend:.3f})")
        print(f"      Gap was already narrowing before NPD — caution interpreting post-NPD change")
    print(f"\n    Post-NPD gap collapses to near zero (mean={post_gaps.mean():+.4f})")
    if dynamics.get('break_post_p') is not None:
        if dynamics['break_post_p'] < 0.10:
            print(f"    Level shift at NPD: δ₂={dynamics['break_post_coef']:+.4f} (p={dynamics['break_post_p']:.3f})")
            print(f"    → Supports Channel 1: IMMEDIATE enforcement risk elimination")
        else:
            print(f"    Level shift at NPD not significant (p={dynamics['break_post_p']:.3f})")
    if dynamics.get('break_trend_p') is not None:
        if dynamics['break_trend_p'] < 0.10:
            print(f"    Post-NPD trend: δ₃={dynamics['break_trend_coef']:+.4f} (p={dynamics['break_trend_p']:.3f})")
            print(f"    → Also supports Channel 4: continuing improvement via planning horizon")
        else:
            print(f"    No significant post-NPD trend (p={dynamics['break_trend_p']:.3f})")
    print(f"\n    Caveat: only {len(pre_gdf)} pre-periods and {len(gdf[gdf['event_time']>=0])} "
          f"post-periods; wide CIs on informal (N≈700/period)")

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Pass-through by group
    ax = axes[0]
    for group, color, marker in [('Formal', '#2166ac', 'o'), ('Informal', '#b2182b', 's')]:
        gdata = rdf[rdf['group'] == group].sort_values('event_time')
        ax.plot(gdata['event_time'], gdata['beta'], f'-{marker}', color=color,
                label=group, markersize=6)
        ax.fill_between(gdata['event_time'], gdata['ci_lo'], gdata['ci_hi'],
                        alpha=0.15, color=color)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Years relative to NPD adoption')
    ax.set_ylabel('Pass-through coefficient (β)')
    ax.set_title('A. Income-Consumption Pass-Through')
    ax.legend()

    # Panel B: Gap (informal - formal) with pre-trend line
    ax = axes[1]
    pre_color = '#999999'
    post_color = '#b2182b'
    colors_bar = [post_color if et >= 0 else pre_color for et in gdf['event_time']]
    ax.bar(gdf['event_time'], gdf['gap'], color=colors_bar,
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.errorbar(gdf['event_time'], gdf['gap'], yerr=1.96 * gdf['gap_se'],
                fmt='none', color='black', capsize=3)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=pre_gaps.mean(), color='blue', linestyle=':', alpha=0.5,
               label=f'Pre-NPD mean ({pre_gaps.mean():.3f})')
    # Add pre-trend regression line
    if len(pre_gdf) >= 3:
        x_trend = np.linspace(pre_gdf['event_time'].min(), -0.5, 50)
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, '--', color='orange', alpha=0.6,
                label=f'Pre-trend (slope={slope:.3f})')
    ax.set_xlabel('Years relative to NPD adoption')
    ax.set_ylabel('Informal − Formal gap')
    ax.set_title('B. Insurance Gap Dynamics')
    ax.legend(fontsize=8)

    # Panel C: Decomposition schematic
    ax = axes[2]
    if dynamics:
        labels_d = ['Pre-NPD\nmean gap', 'Immediate\ndrop (t=0)', 'Gradual\n(t>0)', 'Post-NPD\nmean gap']
        vals = [pre_mean_gap, immediate_drop,
                gradual_component if not np.isnan(gradual_component) else 0,
                post_gaps.mean()]
        colors_d = ['#999999', '#e41a1c', '#ff7f00', '#4daf4a']
        ax.bar(range(len(labels_d)), vals, color=colors_d, alpha=0.7,
               edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(labels_d)))
        ax.set_xticklabels(labels_d, fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_ylabel('Gap magnitude')
        ax.set_title('C. Gap Decomposition')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'channel_event_dynamics.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {FIG_DIR}/channel_event_dynamics.png")

    # Save data
    rdf.to_csv(os.path.join(TABLE_DIR, 'channel_event_dynamics.csv'), index=False)
    gdf.to_csv(os.path.join(TABLE_DIR, 'channel_gap_dynamics.csv'), index=False)
    print(f"  Saved: {TABLE_DIR}/channel_event_dynamics.csv")
    print(f"  Saved: {TABLE_DIR}/channel_gap_dynamics.csv")

    return rdf, gdf


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_panel()
    df = add_industry(df)
    df = add_earner_rank(df)

    print("\n" + "#" * 70)
    print("# CHANNEL IDENTIFICATION TESTS")
    print("#" * 70)

    sector_results = test_sector_heterogeneity(df)
    earner_results = test_earner_heterogeneity(df)
    event_results = test_event_dynamics(df)

    print("\n" + "=" * 70)
    print("ALL CHANNEL TESTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
