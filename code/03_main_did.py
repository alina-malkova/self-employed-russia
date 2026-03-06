"""
03_main_did.py
Main difference-in-differences estimation for labor informality & welfare cost.

Uses NPD staggered rollout (2019-2020) across regions to identify:
  1. Whether NPD reduces informality
  2. Whether NPD improves consumption insurance (Townsend DiD)
  3. Whether consumption volatility falls for informal workers post-NPD

Key specification (Townsend DiD — triple-difference):
  Δln(c)_it = β₁ Δln(w) + β₂ Δln(w)×informal + β₃ Δln(w)×post_NPD
              + β₄ Δln(w)×informal×post_NPD  ← KEY COEFFICIENT
              + controls + year FE + ε_it
  β₄ < 0 means NPD improves consumption insurance for informal workers.

Requires: data/cleaned/rlms_informality_panel.pkl (from 01_clean_rlms.py)
"""

import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")
FIG_DIR = os.path.join(PROJECT_DIR, "output", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PANEL_PATH = os.path.join(DATA_DIR, "rlms_informality_panel.pkl")


def load_and_prepare():
    """Load panel and construct DiD variables."""
    print("Loading panel...")
    df = pd.read_pickle(PANEL_PATH)
    df['year'] = df['year'].astype(int)

    # Event time relative to NPD adoption
    df['event_time'] = df['year'] - df['npd_year']

    # NPD cohort: early (2019 pilots) vs late (2020 bulk)
    df['npd_cohort'] = np.where(df['npd_year'] == 2019, 'early', 'late')

    # Binary informality (employed only)
    df['is_informal'] = df['emp_type'].isin(['informal_wage', 'self_employed']).astype(int)
    df['is_informal_wage'] = (df['emp_type'] == 'informal_wage').astype(int)
    df['is_self_employed'] = (df['emp_type'] == 'self_employed').astype(int)

    # Interactions
    df['post_npd_int'] = df['post_npd'].fillna(0).astype(int)
    df['dw_x_informal'] = df['wage_growth'] * df['is_informal']
    df['dw_x_post'] = df['wage_growth'] * df['post_npd_int']
    df['dw_x_informal_x_post'] = df['wage_growth'] * df['is_informal'] * df['post_npd_int']
    df['informal_x_post'] = df['is_informal'] * df['post_npd_int']

    # Squared consumption growth (for variance analysis)
    df['cg_squared'] = df['consumption_growth'] ** 2

    print(f"  Panel: {len(df):,} obs, {df['year'].min()}-{df['year'].max()}")
    return df


def cluster_se(model, df, cluster_var):
    """Compute cluster-robust standard errors."""
    clusters = df[cluster_var].values
    return model.get_robustcov_results(cov_type='cluster',
                                        groups=clusters)


def format_coef(beta, se, p):
    """Format coefficient with stars."""
    stars = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    return f"{beta:.4f}{stars}", f"({se:.4f})"


# ============================================================
# 1. Informality rate DiD (does NPD reduce informality?)
# ============================================================
def informality_did(df):
    """Individual FE regression: informal_it = α_i + δ_t + β post_NPD_rt + ε."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 1: Informality Rate DiD")
    print("=" * 60)

    # Sample: employed workers with informality classification
    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed']) &
                df['post_npd'].notna()].copy()
    sample = sample.dropna(subset=['is_informal', 'post_npd_int', 'ter', 'idind'])
    sample['year_cat'] = pd.Categorical(sample['year'])

    print(f"  Sample: {len(sample):,} obs, {sample['idind'].nunique():,} individuals")

    # PanelOLS with individual + year FE
    sample = sample.set_index(['idind', 'year'])
    y = sample['is_informal']
    X = sample[['post_npd_int']]

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type='clustered', cluster_entity=False,
                  clusters=sample['ter'])

    print(f"\n  Dependent variable: is_informal (0/1)")
    print(f"  post_NPD coefficient: {res.params['post_npd_int']:.4f} "
          f"(SE: {res.std_errors['post_npd_int']:.4f}, "
          f"p: {res.pvalues['post_npd_int']:.4f})")
    print(f"  N={res.nobs:,}, R²={res.rsquared:.4f}")
    print(f"  FE: individual + year, clusters: {sample['ter'].nunique()} oblasts")

    # Also by type
    for outcome, label in [('is_informal_wage', 'Informal wage'),
                            ('is_self_employed', 'Self-employed')]:
        y_sub = sample[outcome]
        mod_sub = PanelOLS(y_sub, X, entity_effects=True, time_effects=True)
        res_sub = mod_sub.fit(cov_type='clustered', cluster_entity=False,
                              clusters=sample['ter'])
        print(f"  {label}: β={res_sub.params['post_npd_int']:.4f} "
              f"(SE: {res_sub.std_errors['post_npd_int']:.4f}, "
              f"p: {res_sub.pvalues['post_npd_int']:.4f})")

    sample = sample.reset_index()
    return res


# ============================================================
# 2. Townsend DiD (main specification)
# ============================================================
def townsend_did(df):
    """
    Triple-difference: Δln(c) = β₁ Δln(w) + β₂ Δln(w)×informal
                                + β₃ Δln(w)×post_NPD
                                + β₄ Δln(w)×informal×post_NPD  ← KEY
                                + controls + year FE + ε
    β₄ < 0 means NPD improves consumption insurance for informal workers.
    """
    print("\n" + "=" * 60)
    print("SPECIFICATION 2: Townsend Consumption Insurance DiD")
    print("=" * 60)

    # Sample: employed workers with both growth rates
    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter'])

    print(f"  Sample: {len(sample):,} obs, {sample['idind'].nunique():,} individuals")

    # Year dummies (consumption growth already differences out individual FE)
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True,
                                   dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = [c for c in year_dummies.columns]

    results = {}

    # --- Spec 2a: Baseline Townsend (no DiD) ---
    X_vars_a = ['wage_growth', 'dw_x_informal', 'is_informal'] + yr_cols
    X_a = sm.add_constant(sample[X_vars_a])
    y_a = sample['consumption_growth']
    mod_a = sm.OLS(y_a, X_a).fit(cov_type='cluster',
                                   cov_kwds={'groups': sample['ter']})
    print("\n--- Spec 2a: Baseline Townsend (pooled) ---")
    for v in ['wage_growth', 'dw_x_informal']:
        print(f"  {v:30s}: {mod_a.params[v]:8.4f} "
              f"(SE: {mod_a.bse[v]:.4f}, p: {mod_a.pvalues[v]:.4f})")
    print(f"  N={mod_a.nobs:,.0f}, R²={mod_a.rsquared:.4f}")
    results['baseline'] = mod_a

    # --- Spec 2b: Townsend DiD (main) ---
    X_vars_b = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                'dw_x_informal_x_post',
                'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
    X_b = sm.add_constant(sample[X_vars_b])
    y_b = sample['consumption_growth']
    mod_b = sm.OLS(y_b, X_b).fit(cov_type='cluster',
                                   cov_kwds={'groups': sample['ter']})

    print("\n--- Spec 2b: Townsend DiD (main result) ---")
    key_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                'dw_x_informal_x_post',
                'is_informal', 'post_npd_int', 'informal_x_post']
    for v in key_vars:
        stars = '***' if mod_b.pvalues[v] < 0.01 else '**' if mod_b.pvalues[v] < 0.05 else '*' if mod_b.pvalues[v] < 0.1 else ''
        print(f"  {v:30s}: {mod_b.params[v]:8.4f}{stars:3s} "
              f"(SE: {mod_b.bse[v]:.4f})")
    print(f"  N={mod_b.nobs:,.0f}, R²={mod_b.rsquared:.4f}")
    print(f"  Clusters: {sample['ter'].nunique()} oblasts")
    results['main'] = mod_b

    # --- Spec 2c: With individual controls ---
    controls = []
    for c in ['age', 'female']:
        if c in sample.columns and sample[c].notna().sum() > 0:
            controls.append(c)

    X_vars_c = X_vars_b + controls
    sample_c = sample.dropna(subset=controls)
    X_c = sm.add_constant(sample_c[X_vars_c])
    y_c = sample_c['consumption_growth']
    mod_c = sm.OLS(y_c, X_c).fit(cov_type='cluster',
                                   cov_kwds={'groups': sample_c['ter']})

    print("\n--- Spec 2c: With individual controls ---")
    for v in key_vars:
        stars = '***' if mod_c.pvalues[v] < 0.01 else '**' if mod_c.pvalues[v] < 0.05 else '*' if mod_c.pvalues[v] < 0.1 else ''
        print(f"  {v:30s}: {mod_c.params[v]:8.4f}{stars:3s} "
              f"(SE: {mod_c.bse[v]:.4f})")
    print(f"  N={mod_c.nobs:,.0f}, R²={mod_c.rsquared:.4f}")
    results['controls'] = mod_c

    # --- Save coefficient table ---
    coef_rows = []
    for spec_name, mod in results.items():
        for v in key_vars:
            if v in mod.params:
                coef_rows.append({
                    'Specification': spec_name,
                    'Variable': v,
                    'Coefficient': mod.params[v],
                    'SE': mod.bse[v],
                    'p-value': mod.pvalues[v],
                    'N': int(mod.nobs),
                    'R2': mod.rsquared,
                })
    coef_table = pd.DataFrame(coef_rows)
    coef_table.to_csv(os.path.join(TABLE_DIR, 'townsend_did_results.csv'), index=False)

    return results


# ============================================================
# 3. Townsend DiD by employment type (3-way split)
# ============================================================
def townsend_by_type(df):
    """Run Townsend test separately for each employment type."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 3: Townsend DiD by Employment Type")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'post_npd_int', 'ter'])
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True,
                                   dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    rows = []
    for etype in ['formal_wage', 'informal_wage', 'self_employed']:
        sub = sample[sample['emp_type'] == etype]
        X_vars = ['wage_growth', 'dw_x_post', 'post_npd_int'] + yr_cols
        X = sm.add_constant(sub[X_vars])
        y = sub['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster',
                                cov_kwds={'groups': sub['ter']})

        beta_wg = mod.params['wage_growth']
        beta_dw_post = mod.params['dw_x_post']
        se_wg = mod.bse['wage_growth']
        se_dw_post = mod.bse['dw_x_post']
        p_wg = mod.pvalues['wage_growth']
        p_dw_post = mod.pvalues['dw_x_post']

        stars_wg = '***' if p_wg < 0.01 else '**' if p_wg < 0.05 else '*' if p_wg < 0.1 else ''
        stars_post = '***' if p_dw_post < 0.01 else '**' if p_dw_post < 0.05 else '*' if p_dw_post < 0.1 else ''

        print(f"\n  {etype} (N={len(sub):,}):")
        print(f"    Δln(w) pass-through:     {beta_wg:.4f}{stars_wg} (SE: {se_wg:.4f})")
        print(f"    Δln(w) × post_NPD:       {beta_dw_post:.4f}{stars_post} (SE: {se_dw_post:.4f})")
        print(f"    Pre-NPD pass-through:     {beta_wg:.4f}")
        print(f"    Post-NPD pass-through:    {beta_wg + beta_dw_post:.4f}")

        rows.append({
            'emp_type': etype,
            'N': len(sub),
            'beta_wg': beta_wg, 'se_wg': se_wg, 'p_wg': p_wg,
            'beta_dw_post': beta_dw_post, 'se_dw_post': se_dw_post, 'p_dw_post': p_dw_post,
            'pre_passthrough': beta_wg,
            'post_passthrough': beta_wg + beta_dw_post,
        })

    type_table = pd.DataFrame(rows)
    type_table.to_csv(os.path.join(TABLE_DIR, 'townsend_did_by_type.csv'), index=False)

    # --- Figure: pre vs post pass-through by type ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rows))
    width = 0.35
    colors_pre = ['#2166ac', '#b2182b', '#4daf4a']
    colors_post = ['#67a9cf', '#ef8a62', '#a6d96a']

    for i, row in enumerate(rows):
        ax.bar(i - width/2, row['pre_passthrough'], width, color=colors_pre[i],
               label=f"Pre-NPD" if i == 0 else None, alpha=0.8)
        ax.bar(i + width/2, row['post_passthrough'], width, color=colors_post[i],
               label=f"Post-NPD" if i == 0 else None, alpha=0.8)
        # Error bars on the change
        ax.errorbar(i + width/2, row['post_passthrough'],
                    yerr=1.96 * row['se_dw_post'], fmt='none', color='black', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['Formal\nwage', 'Informal\nwage', 'Self-\nemployed'])
    ax.set_ylabel('β (income → consumption pass-through)')
    ax.set_title('Townsend Insurance Test: Pre vs Post NPD')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'townsend_did_by_type.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved: townsend_did_by_type.png")

    return type_table


# ============================================================
# 4. Consumption variance DiD
# ============================================================
def variance_did(df):
    """Does NPD reduce consumption volatility for informal workers?
    Regress (Δln c)² on informal × post_NPD with year FE."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 4: Consumption Variance DiD")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['cg_squared', 'is_informal', 'post_npd_int', 'ter'])

    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True,
                                   dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    # Spec: (Δln c)² = β₁ informal + β₂ post_NPD + β₃ informal×post_NPD + year FE
    X_vars = ['is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
    X = sm.add_constant(sample[X_vars])
    y = sample['cg_squared']
    mod = sm.OLS(y, X).fit(cov_type='cluster',
                            cov_kwds={'groups': sample['ter']})

    print(f"\n  Dependent variable: (Δ ln c)²")
    key_vars = ['is_informal', 'post_npd_int', 'informal_x_post']
    for v in key_vars:
        stars = '***' if mod.pvalues[v] < 0.01 else '**' if mod.pvalues[v] < 0.05 else '*' if mod.pvalues[v] < 0.1 else ''
        print(f"  {v:25s}: {mod.params[v]:8.4f}{stars:3s} (SE: {mod.bse[v]:.4f})")
    print(f"  N={mod.nobs:,.0f}, R²={mod.rsquared:.4f}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"    Informal variance premium (pre-NPD):  {mod.params['is_informal']:.4f}")
    print(f"    Post-NPD effect on variance (formal): {mod.params['post_npd_int']:.4f}")
    print(f"    Additional NPD effect for informal:   {mod.params['informal_x_post']:.4f}")
    total_informal_post = (mod.params['is_informal'] + mod.params['informal_x_post'])
    print(f"    Informal premium post-NPD:            {total_informal_post:.4f}")

    # By employment type
    print("\n  --- By employment type ---")
    for etype in ['informal_wage', 'self_employed']:
        type_var = f'is_{etype}'
        inter_var = f'{type_var}_x_post'
        sample[inter_var] = sample[type_var] * sample['post_npd_int']
        X_vars_t = [type_var, 'post_npd_int', inter_var] + yr_cols
        X_t = sm.add_constant(sample[X_vars_t])
        mod_t = sm.OLS(y, X_t).fit(cov_type='cluster',
                                     cov_kwds={'groups': sample['ter']})
        stars = '***' if mod_t.pvalues[inter_var] < 0.01 else '**' if mod_t.pvalues[inter_var] < 0.05 else '*' if mod_t.pvalues[inter_var] < 0.1 else ''
        print(f"    {etype}: interaction = {mod_t.params[inter_var]:.4f}{stars} "
              f"(SE: {mod_t.bse[inter_var]:.4f})")

    return mod


# ============================================================
# 5. Event study: Townsend pass-through by event time
# ============================================================
def townsend_event_study(df):
    """Estimate Townsend pass-through for each event-time year."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 5: Townsend Event Study")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'event_time', 'ter'])

    # Bin event time: -10 to -6 as reference group
    sample['et_binned'] = sample['event_time'].clip(-5, 4)

    # Estimate pass-through for informal workers by event-time year
    rows = []
    for et in sorted(sample['et_binned'].unique()):
        sub = sample[sample['et_binned'] == et]

        # Separate regressions for formal vs informal
        for group, mask in [('formal', sub['is_informal'] == 0),
                             ('informal', sub['is_informal'] == 1)]:
            gsub = sub[mask]
            if len(gsub) < 50:
                continue
            X = sm.add_constant(gsub[['wage_growth']])
            y = gsub['consumption_growth']
            mod = sm.OLS(y, X).fit(cov_type='cluster',
                                    cov_kwds={'groups': gsub['ter']})
            rows.append({
                'event_time': et,
                'group': group,
                'beta': mod.params['wage_growth'],
                'se': mod.bse['wage_growth'],
                'p': mod.pvalues['wage_growth'],
                'N': len(gsub),
            })

    es_df = pd.DataFrame(rows)
    es_df.to_csv(os.path.join(TABLE_DIR, 'townsend_event_study.csv'), index=False)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color, label in [('formal', '#2166ac', 'Formal wage'),
                                  ('informal', '#b2182b', 'Informal (wage + self-emp)')]:
        g = es_df[es_df['group'] == group]
        ax.plot(g['event_time'], g['beta'], marker='o', color=color,
                linewidth=2, label=label)
        ax.fill_between(g['event_time'],
                        g['beta'] - 1.96 * g['se'],
                        g['beta'] + 1.96 * g['se'],
                        alpha=0.15, color=color)

    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7, label='NPD adoption')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Event time (years relative to NPD adoption)')
    ax.set_ylabel('β (income → consumption pass-through)')
    ax.set_title('Townsend Insurance Test: Event Study Around NPD Rollout')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'townsend_event_study.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: townsend_event_study.png")

    # Print
    print("\n  Pass-through by event time:")
    pivot = es_df.pivot(index='event_time', columns='group', values='beta')
    if 'formal' in pivot.columns and 'informal' in pivot.columns:
        pivot['gap'] = pivot['informal'] - pivot['formal']
    print(pivot.round(4).to_string())

    return es_df


# ============================================================
# 6. Variance event study
# ============================================================
def variance_event_study(df):
    """Track Var(Δ ln c) by event time for formal vs informal."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 6: Consumption Variance Event Study")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'is_informal', 'event_time'])
    sample['et_binned'] = sample['event_time'].clip(-5, 4)

    rows = []
    for et in sorted(sample['et_binned'].unique()):
        sub = sample[sample['et_binned'] == et]
        for group, mask in [('formal', sub['is_informal'] == 0),
                             ('informal', sub['is_informal'] == 1)]:
            gsub = sub[mask]
            if len(gsub) < 50:
                continue
            cg = gsub['consumption_growth']
            rows.append({
                'event_time': et,
                'group': group,
                'variance': cg.var(),
                'std': cg.std(),
                'N': len(gsub),
                'mean': cg.mean(),
            })

    var_es = pd.DataFrame(rows)
    var_es.to_csv(os.path.join(TABLE_DIR, 'variance_event_study.csv'), index=False)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for group, color, label in [('formal', '#2166ac', 'Formal wage'),
                                  ('informal', '#b2182b', 'Informal (wage + self-emp)')]:
        g = var_es[var_es['group'] == group]
        ax.plot(g['event_time'], g['variance'], marker='o', color=color,
                linewidth=2, label=label)

    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.7, label='NPD adoption')
    ax.set_xlabel('Event time (years relative to NPD adoption)')
    ax.set_ylabel('Var(Δ ln consumption)')
    ax.set_title('Consumption Variance: Event Study Around NPD Rollout')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'variance_event_study.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: variance_event_study.png")

    # Print
    pivot = var_es.pivot(index='event_time', columns='group', values='variance')
    if 'formal' in pivot.columns and 'informal' in pivot.columns:
        pivot['gap'] = pivot['informal'] - pivot['formal']
    print("\n  Variance by event time:")
    print(pivot.round(4).to_string())

    return var_es


# ============================================================
# 7. Robustness: By NPD cohort
# ============================================================
def townsend_by_cohort(df):
    """Separate Townsend DiD for early (2019) vs late (2020) NPD adopters."""
    print("\n" + "=" * 60)
    print("SPECIFICATION 7: Townsend DiD by NPD Cohort")
    print("=" * 60)

    sample = df[df['emp_type'].isin(['formal_wage', 'informal_wage', 'self_employed'])].copy()
    sample = sample.dropna(subset=['consumption_growth', 'wage_growth',
                                    'is_informal', 'post_npd_int', 'ter', 'npd_cohort'])
    year_dummies = pd.get_dummies(sample['year'], prefix='yr', drop_first=True,
                                   dtype=float)
    sample = pd.concat([sample, year_dummies], axis=1)
    yr_cols = list(year_dummies.columns)

    rows = []
    for cohort in ['early', 'late']:
        sub = sample[sample['npd_cohort'] == cohort]
        X_vars = ['wage_growth', 'dw_x_informal', 'dw_x_post',
                  'dw_x_informal_x_post',
                  'is_informal', 'post_npd_int', 'informal_x_post'] + yr_cols
        X = sm.add_constant(sub[X_vars])
        y = sub['consumption_growth']
        mod = sm.OLS(y, X).fit(cov_type='cluster',
                                cov_kwds={'groups': sub['ter']})

        beta_key = mod.params['dw_x_informal_x_post']
        se_key = mod.bse['dw_x_informal_x_post']
        p_key = mod.pvalues['dw_x_informal_x_post']
        stars = '***' if p_key < 0.01 else '**' if p_key < 0.05 else '*' if p_key < 0.1 else ''

        print(f"\n  Cohort: {cohort} (NPD {sub['npd_year'].iloc[0]:.0f}, "
              f"N={len(sub):,}, {sub['ter'].nunique()} oblasts)")
        print(f"    Δln(w)×informal×post_NPD: {beta_key:.4f}{stars} (SE: {se_key:.4f})")

        rows.append({
            'cohort': cohort,
            'npd_year': int(sub['npd_year'].iloc[0]),
            'N': len(sub),
            'n_oblasts': sub['ter'].nunique(),
            'beta_triple': beta_key,
            'se_triple': se_key,
            'p_triple': p_key,
        })

    cohort_table = pd.DataFrame(rows)
    cohort_table.to_csv(os.path.join(TABLE_DIR, 'townsend_did_by_cohort.csv'), index=False)
    return cohort_table


# ============================================================
# Summary results table
# ============================================================
def summary_table(townsend_results):
    """Create a clean summary table of main DiD results."""
    print("\n" + "=" * 60)
    print("SUMMARY: Main DiD Results")
    print("=" * 60)

    rows = []
    for spec_name, mod in townsend_results.items():
        key_var = 'dw_x_informal_x_post'
        if key_var in mod.params:
            rows.append({
                'Specification': spec_name,
                'β (Δln w × informal × post_NPD)': mod.params[key_var],
                'SE': mod.bse[key_var],
                'p-value': mod.pvalues[key_var],
                'N': int(mod.nobs),
                'R²': mod.rsquared,
            })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False, float_format='{:.4f}'.format))
    summary.to_csv(os.path.join(TABLE_DIR, 'did_summary.csv'), index=False)
    return summary


# ============================================================
# Main
# ============================================================
def main():
    df = load_and_prepare()

    # 1. Informality rate DiD
    informality_did(df)

    # 2. Townsend DiD (main specification)
    townsend_results = townsend_did(df)

    # 3. Townsend by employment type
    townsend_by_type(df)

    # 4. Consumption variance DiD
    variance_did(df)

    # 5. Townsend event study
    townsend_event_study(df)

    # 6. Variance event study
    variance_event_study(df)

    # 7. By NPD cohort
    townsend_by_cohort(df)

    # Summary
    summary_table(townsend_results)

    print("\n" + "=" * 60)
    print(f"All tables saved to: {TABLE_DIR}")
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
