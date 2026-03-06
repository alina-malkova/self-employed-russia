"""
02_descriptive_stats.py
Descriptive statistics for the labor informality & welfare cost analysis.

Produces:
  - Summary statistics table (overall + by employment type)
  - Consumption variance by employment type (core welfare measure)
  - Townsend consumption insurance test
  - Time-series plots: informality rates, consumption, consumption growth
  - Consumption growth distributions by group

Requires: data/cleaned/rlms_informality_panel.pkl (from 01_clean_rlms.py)
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "cleaned")
TABLE_DIR = os.path.join(PROJECT_DIR, "output", "tables")
FIG_DIR = os.path.join(PROJECT_DIR, "output", "figures")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PANEL_PATH = os.path.join(DATA_DIR, "rlms_informality_panel.pkl")

EMP_TYPE_LABELS = {
    'formal_wage': 'Formal wage',
    'informal_wage': 'Informal wage',
    'self_employed': 'Self-employed',
}
EMP_TYPE_ORDER = ['formal_wage', 'informal_wage', 'self_employed']


def load_panel():
    print(f"Loading panel from {PANEL_PATH}")
    df = pd.read_pickle(PANEL_PATH)
    df['year'] = df['year'].astype(int)
    print(f"  {len(df):,} obs, {df['idind'].nunique():,} individuals, "
          f"{df['year'].min()}-{df['year'].max()}")
    return df


# ============================================================
# 1. Summary statistics
# ============================================================
def summary_stats_table(df):
    """Panel A: full sample. Panel B: by employment type (employed only)."""
    print("\n=== Summary Statistics ===")

    vars_info = [
        ('age', 'Age'),
        ('female', 'Female'),
        ('employed', 'Employed'),
        ('wage', 'Monthly wage (RUB)'),
        ('hours_worked', 'Hours worked/week'),
        ('informal', 'Informal (composite)'),
        ('no_contract', 'No official contract'),
        ('has_envelope_wage', 'Has envelope wages'),
        ('small_firm', 'Small firm (≤5 employees)'),
        ('total_consumption', 'Total consumption (RUB/mo)'),
        ('food_exp', 'Food expenditure (RUB/mo)'),
        ('clothing_exp', 'Clothing expenditure (RUB/mo)'),
        ('utility_exp', 'Utility expenditure (RUB/mo)'),
        ('consumption_growth', 'Consumption growth (Δ ln c)'),
        ('wage_growth', 'Wage growth (Δ ln w)'),
        ('saved_money', 'Saved money (share)'),
        ('gave_transfers', 'Gave transfers (share)'),
    ]

    # Panel A: full sample
    rows = []
    for var, label in vars_info:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        rows.append({
            'Variable': label,
            'N': len(s),
            'Mean': s.mean(),
            'Std': s.std(),
            'Median': s.median(),
            'Min': s.min(),
            'Max': s.max(),
        })
    panel_a = pd.DataFrame(rows)

    # Panel B: by employment type (employed only)
    emp_df = df[df['emp_type'].isin(EMP_TYPE_ORDER)].copy()
    emp_vars = [
        ('wage', 'Monthly wage (RUB)'),
        ('hours_worked', 'Hours worked/week'),
        ('total_consumption', 'Total consumption (RUB/mo)'),
        ('ln_consumption', 'Log consumption'),
        ('consumption_growth', 'Consumption growth (Δ ln c)'),
        ('wage_growth', 'Wage growth (Δ ln w)'),
        ('age', 'Age'),
        ('female', 'Female'),
    ]
    rows_b = []
    for var, label in emp_vars:
        if var not in emp_df.columns:
            continue
        row = {'Variable': label}
        for etype in EMP_TYPE_ORDER:
            s = emp_df.loc[emp_df['emp_type'] == etype, var].dropna()
            elabel = EMP_TYPE_LABELS[etype]
            row[f'{elabel} Mean'] = s.mean()
            row[f'{elabel} Std'] = s.std()
            row[f'{elabel} N'] = len(s)
        rows_b.append(row)

    # Add group sizes
    group_n = emp_df.groupby('emp_type').size()
    rows_b.insert(0, {
        'Variable': 'Observations',
        **{f'{EMP_TYPE_LABELS[e]} Mean': group_n.get(e, 0) for e in EMP_TYPE_ORDER},
        **{f'{EMP_TYPE_LABELS[e]} Std': '' for e in EMP_TYPE_ORDER},
        **{f'{EMP_TYPE_LABELS[e]} N': '' for e in EMP_TYPE_ORDER},
    })
    panel_b = pd.DataFrame(rows_b)

    # Save
    panel_a.to_csv(os.path.join(TABLE_DIR, 'summary_stats_full.csv'), index=False)
    panel_b.to_csv(os.path.join(TABLE_DIR, 'summary_stats_by_emptype.csv'), index=False)

    print("\nPanel A: Full Sample")
    print(panel_a.to_string(index=False, float_format='{:.2f}'.format))
    print("\nPanel B: By Employment Type (employed only)")
    print(panel_b.to_string(index=False, float_format='{:.2f}'.format))

    return panel_a, panel_b


# ============================================================
# 2. Consumption variance by employment type (core welfare measure)
# ============================================================
def consumption_variance_analysis(df):
    """Compare consumption growth variance across employment types."""
    print("\n=== Consumption Variance by Employment Type ===")

    emp_df = df[df['emp_type'].isin(EMP_TYPE_ORDER) &
                df['consumption_growth'].notna()].copy()

    # Overall stats
    rows = []
    for etype in EMP_TYPE_ORDER:
        sub = emp_df[emp_df['emp_type'] == etype]
        cg = sub['consumption_growth']
        rows.append({
            'Employment type': EMP_TYPE_LABELS[etype],
            'N': len(cg),
            'Mean Δln(c)': cg.mean(),
            'Std Δln(c)': cg.std(),
            'Var Δln(c)': cg.var(),
            'Median Δln(c)': cg.median(),
            'IQR Δln(c)': cg.quantile(0.75) - cg.quantile(0.25),
        })
    var_table = pd.DataFrame(rows)
    print(var_table.to_string(index=False, float_format='{:.4f}'.format))

    # F-test: informal wage vs formal wage
    formal_cg = emp_df.loc[emp_df['emp_type'] == 'formal_wage', 'consumption_growth']
    informal_cg = emp_df.loc[emp_df['emp_type'] == 'informal_wage', 'consumption_growth']
    selfem_cg = emp_df.loc[emp_df['emp_type'] == 'self_employed', 'consumption_growth']

    f_stat_inf = informal_cg.var() / formal_cg.var()
    f_stat_se = selfem_cg.var() / formal_cg.var()
    print(f"\nVariance ratio (informal/formal): {f_stat_inf:.3f}")
    print(f"Variance ratio (self-emp/formal):  {f_stat_se:.3f}")

    # Levene's test for equality of variances
    lev_stat, lev_p = stats.levene(formal_cg.dropna(), informal_cg.dropna(),
                                    selfem_cg.dropna())
    print(f"Levene's test: stat={lev_stat:.2f}, p={lev_p:.4f}")

    # By year
    rows_yr = []
    for year in sorted(emp_df['year'].unique()):
        yr_data = emp_df[emp_df['year'] == year]
        row = {'Year': year}
        for etype in EMP_TYPE_ORDER:
            cg = yr_data.loc[yr_data['emp_type'] == etype, 'consumption_growth']
            elabel = EMP_TYPE_LABELS[etype]
            row[f'{elabel} Var'] = cg.var() if len(cg) > 1 else np.nan
            row[f'{elabel} N'] = len(cg)
        rows_yr.append(row)
    var_by_year = pd.DataFrame(rows_yr)

    var_table.to_csv(os.path.join(TABLE_DIR, 'consumption_variance_by_emptype.csv'),
                     index=False)
    var_by_year.to_csv(os.path.join(TABLE_DIR, 'consumption_variance_by_year.csv'),
                       index=False)

    # --- Plot: variance by year ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'formal_wage': '#2166ac', 'informal_wage': '#b2182b',
              'self_employed': '#4daf4a'}
    for etype in EMP_TYPE_ORDER:
        elabel = EMP_TYPE_LABELS[etype]
        ax.plot(var_by_year['Year'], var_by_year[f'{elabel} Var'],
                marker='o', label=elabel, color=colors[etype], linewidth=2)

    ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='NPD pilot (2019)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Var(Δ ln consumption)')
    ax.set_title('Consumption Growth Variance by Employment Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'consumption_variance_by_year.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved figure: consumption_variance_by_year.png")

    return var_table, var_by_year


# ============================================================
# 3. Townsend consumption insurance test
# ============================================================
def townsend_test(df):
    """
    Regress Δ ln(c) on Δ ln(w) by employment type.
    Full insurance → coefficient = 0 (consumption insulated from income shocks).
    No insurance → coefficient = 1 (consumption moves 1:1 with income).
    """
    print("\n=== Townsend Consumption Insurance Test ===")
    print("  H0: β=0 (full insurance); H1: β>0 (incomplete insurance)")

    emp_df = df[df['emp_type'].isin(EMP_TYPE_ORDER) &
                df['consumption_growth'].notna() &
                df['wage_growth'].notna()].copy()

    rows = []

    # Overall
    from numpy.linalg import lstsq
    for etype in EMP_TYPE_ORDER:
        sub = emp_df[emp_df['emp_type'] == etype].dropna(
            subset=['consumption_growth', 'wage_growth'])
        if len(sub) < 30:
            continue

        y = sub['consumption_growth'].values
        X = np.column_stack([np.ones(len(sub)), sub['wage_growth'].values])
        beta, residuals, _, _ = lstsq(X, y, rcond=None)
        y_hat = X @ beta
        resid = y - y_hat
        n, k = X.shape
        se = np.sqrt(np.diag((resid @ resid / (n - k)) * np.linalg.inv(X.T @ X)))

        t_stat = beta[1] / se[1]
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
        r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))

        rows.append({
            'Employment type': EMP_TYPE_LABELS[etype],
            'N': len(sub),
            'β (Δln w)': beta[1],
            'SE': se[1],
            't-stat': t_stat,
            'p-value': p_val,
            'R²': r2,
            'Intercept': beta[0],
        })

    townsend_table = pd.DataFrame(rows)
    print(townsend_table.to_string(index=False, float_format='{:.4f}'.format))

    townsend_table.to_csv(os.path.join(TABLE_DIR, 'townsend_test.csv'), index=False)

    # Townsend test by pre/post NPD
    print("\n--- Townsend Test: Pre vs Post NPD ---")
    rows_period = []
    for period_label, mask in [('Pre-NPD (2010-2018)',
                                 emp_df['year'] <= 2018),
                                ('Post-NPD (2019-2023)',
                                 emp_df['year'] >= 2019)]:
        period_df = emp_df[mask]
        for etype in EMP_TYPE_ORDER:
            sub = period_df[period_df['emp_type'] == etype].dropna(
                subset=['consumption_growth', 'wage_growth'])
            if len(sub) < 30:
                continue
            y = sub['consumption_growth'].values
            X = np.column_stack([np.ones(len(sub)), sub['wage_growth'].values])
            beta, _, _, _ = lstsq(X, y, rcond=None)
            resid = y - X @ beta
            n, k = X.shape
            se = np.sqrt(np.diag((resid @ resid / (n - k)) * np.linalg.inv(X.T @ X)))
            t_stat = beta[1] / se[1]
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

            rows_period.append({
                'Period': period_label,
                'Employment type': EMP_TYPE_LABELS[etype],
                'N': len(sub),
                'β (Δln w)': beta[1],
                'SE': se[1],
                'p-value': p_val,
            })

    townsend_period = pd.DataFrame(rows_period)
    print(townsend_period.to_string(index=False, float_format='{:.4f}'.format))
    townsend_period.to_csv(os.path.join(TABLE_DIR, 'townsend_test_by_period.csv'),
                           index=False)

    # --- Plot: Townsend coefficients ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(townsend_table))
    colors = ['#2166ac', '#b2182b', '#4daf4a']
    bars = ax.bar(x_pos, townsend_table['β (Δln w)'],
                  yerr=1.96 * townsend_table['SE'],
                  color=colors[:len(townsend_table)], capsize=5, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(townsend_table['Employment type'])
    ax.set_ylabel('β (pass-through of income to consumption)')
    ax.set_title('Townsend Insurance Test by Employment Type')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='No insurance (β=1)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Annotate bars
    for i, row in townsend_table.iterrows():
        stars = '***' if row['p-value'] < 0.01 else '**' if row['p-value'] < 0.05 else '*' if row['p-value'] < 0.1 else ''
        ax.text(i, row['β (Δln w)'] + 1.96 * row['SE'] + 0.01,
                f"{row['β (Δln w)']:.3f}{stars}", ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'townsend_test.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved figure: townsend_test.png")

    return townsend_table


# ============================================================
# 4. Time-series plots
# ============================================================
def time_series_plots(df):
    """Informality rates, mean consumption, and consumption growth over time."""
    print("\n=== Time-Series Plots ===")

    emp_df = df[df['employed'] == 1].copy()
    colors = {'formal_wage': '#2166ac', 'informal_wage': '#b2182b',
              'self_employed': '#4daf4a'}

    # --- Plot 1: Employment type shares over time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    type_shares = emp_df.groupby('year')['emp_type'].value_counts(
        normalize=True).unstack(fill_value=0)
    for etype in EMP_TYPE_ORDER:
        if etype in type_shares.columns:
            ax.plot(type_shares.index, type_shares[etype] * 100,
                    marker='o', label=EMP_TYPE_LABELS[etype],
                    color=colors[etype], linewidth=2)
    ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='NPD pilot')
    ax.set_xlabel('Year')
    ax.set_ylabel('Share of employed (%)')
    ax.set_title('Employment Type Shares Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'employment_type_shares.png'), dpi=150)
    plt.close(fig)

    # --- Plot 2: Mean consumption by employment type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for etype in EMP_TYPE_ORDER:
        sub = emp_df[emp_df['emp_type'] == etype]
        means = sub.groupby('year')['total_consumption'].mean()
        ax.plot(means.index, means.values / 1000, marker='o',
                label=EMP_TYPE_LABELS[etype], color=colors[etype], linewidth=2)
    ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='NPD pilot')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean total consumption (thousand RUB/mo)')
    ax.set_title('Mean Consumption by Employment Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'consumption_by_emptype.png'), dpi=150)
    plt.close(fig)

    # --- Plot 3: Mean consumption growth by employment type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for etype in EMP_TYPE_ORDER:
        sub = emp_df[emp_df['emp_type'] == etype]
        means = sub.groupby('year')['consumption_growth'].mean()
        ax.plot(means.index, means.values, marker='o',
                label=EMP_TYPE_LABELS[etype], color=colors[etype], linewidth=2)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='NPD pilot')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Δ ln(consumption)')
    ax.set_title('Consumption Growth by Employment Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'consumption_growth_by_emptype.png'), dpi=150)
    plt.close(fig)

    # --- Plot 4: Wage by employment type ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for etype in EMP_TYPE_ORDER:
        sub = emp_df[emp_df['emp_type'] == etype]
        means = sub.groupby('year')['wage'].mean()
        ax.plot(means.index, means.values / 1000, marker='o',
                label=EMP_TYPE_LABELS[etype], color=colors[etype], linewidth=2)
    ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.7, label='NPD pilot')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean wage (thousand RUB/mo)')
    ax.set_title('Mean Wages by Employment Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'wages_by_emptype.png'), dpi=150)
    plt.close(fig)

    print("  Saved: employment_type_shares.png, consumption_by_emptype.png, "
          "consumption_growth_by_emptype.png, wages_by_emptype.png")


# ============================================================
# 5. Consumption growth distributions
# ============================================================
def consumption_growth_distributions(df):
    """Kernel density plots of Δ ln c by employment type."""
    print("\n=== Consumption Growth Distributions ===")

    emp_df = df[df['emp_type'].isin(EMP_TYPE_ORDER) &
                df['consumption_growth'].notna()].copy()

    # Winsorize at 1st/99th for plotting
    p01 = emp_df['consumption_growth'].quantile(0.01)
    p99 = emp_df['consumption_growth'].quantile(0.99)
    emp_df['cg_wins'] = emp_df['consumption_growth'].clip(p01, p99)

    colors = {'formal_wage': '#2166ac', 'informal_wage': '#b2182b',
              'self_employed': '#4daf4a'}

    fig, ax = plt.subplots(figsize=(10, 6))
    for etype in EMP_TYPE_ORDER:
        data = emp_df.loc[emp_df['emp_type'] == etype, 'cg_wins'].dropna()
        kde = stats.gaussian_kde(data)
        x_grid = np.linspace(p01, p99, 300)
        ax.plot(x_grid, kde(x_grid), label=EMP_TYPE_LABELS[etype],
                color=colors[etype], linewidth=2)

    ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Δ ln(consumption)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Consumption Growth by Employment Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'consumption_growth_kde.png'), dpi=150)
    plt.close(fig)

    # Pre vs post NPD
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax_i, (period, mask) in enumerate([
        ('Pre-NPD (2011-2018)', emp_df['year'].between(2011, 2018)),
        ('Post-NPD (2019-2023)', emp_df['year'].between(2019, 2023)),
    ]):
        ax = axes[ax_i]
        for etype in EMP_TYPE_ORDER:
            data = emp_df.loc[mask & (emp_df['emp_type'] == etype),
                              'cg_wins'].dropna()
            if len(data) < 30:
                continue
            kde = stats.gaussian_kde(data)
            x_grid = np.linspace(p01, p99, 300)
            ax.plot(x_grid, kde(x_grid), label=EMP_TYPE_LABELS[etype],
                    color=colors[etype], linewidth=2)
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Δ ln(consumption)')
        ax.set_title(period)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Density')
    fig.suptitle('Consumption Growth Distribution: Pre vs Post NPD', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'consumption_growth_kde_pre_post.png'), dpi=150)
    plt.close(fig)

    print("  Saved: consumption_growth_kde.png, consumption_growth_kde_pre_post.png")


# ============================================================
# 6. Region and NPD coverage
# ============================================================
def region_npd_summary(df):
    """Summarize geographic and NPD treatment coverage."""
    print("\n=== Region & NPD Coverage ===")

    # Observations by oblast
    if 'ter_lab' in df.columns:
        oblast_counts = df.groupby('ter_lab').agg(
            N=('idind', 'size'),
            N_individuals=('idind', 'nunique'),
            Informality_rate=('informal', 'mean'),
            Mean_consumption=('total_consumption', 'mean'),
        ).sort_values('N', ascending=False)
        print("\nObservations by Oblast:")
        print(oblast_counts.round(3).to_string())
        oblast_counts.to_csv(os.path.join(TABLE_DIR, 'obs_by_oblast.csv'))

    # NPD wave coverage
    if 'npd_wave' in df.columns:
        wave_counts = df.groupby('npd_wave').agg(
            N=('idind', 'size'),
            N_individuals=('idind', 'nunique'),
            NPD_year=('npd_year', 'first'),
        ).sort_values('NPD_year')
        print("\nObservations by NPD Wave:")
        print(wave_counts.to_string())
        wave_counts.to_csv(os.path.join(TABLE_DIR, 'obs_by_npd_wave.csv'))

    # Pre/post NPD balance
    if 'post_npd' in df.columns:
        emp_df = df[df['emp_type'].isin(EMP_TYPE_ORDER)].copy()
        balance_vars = ['age', 'female', 'wage', 'total_consumption',
                        'informal', 'hours_worked']
        avail = [v for v in balance_vars if v in emp_df.columns]

        rows = []
        for var in avail:
            pre = emp_df.loc[emp_df['post_npd'] == 0, var].dropna()
            post = emp_df.loc[emp_df['post_npd'] == 1, var].dropna()
            t_stat, p_val = stats.ttest_ind(pre, post)
            rows.append({
                'Variable': var,
                'Pre-NPD mean': pre.mean(),
                'Post-NPD mean': post.mean(),
                'Difference': post.mean() - pre.mean(),
                't-stat': t_stat,
                'p-value': p_val,
            })
        balance = pd.DataFrame(rows)
        print("\nPre/Post NPD Balance (employed workers):")
        print(balance.to_string(index=False, float_format='{:.3f}'.format))
        balance.to_csv(os.path.join(TABLE_DIR, 'npd_balance.csv'), index=False)


# ============================================================
# Main
# ============================================================
def main():
    df = load_panel()

    summary_stats_table(df)
    consumption_variance_analysis(df)
    townsend_test(df)
    time_series_plots(df)
    consumption_growth_distributions(df)
    region_npd_summary(df)

    print("\n" + "=" * 60)
    print("All tables saved to:", TABLE_DIR)
    print("All figures saved to:", FIG_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
