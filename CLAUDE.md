# Labor Informality and Welfare Costs in Russia

## Project Overview
Empirical paper measuring the **welfare cost of labor informality** in Russia through **consumption variance**, not wages or TFP. Key insight: informal workers may earn comparable wages but bear higher consumption risk due to lack of social insurance (pensions, sick leave, unemployment benefits).

## Core Contribution
Standard approaches (Ulyssea 2018 structural model; Boeri-Garibaldi / Charlot et al. search models) identify welfare through wages or TFP. This paper measures welfare cost as **consumption volatility** — the channel these wage-based models underestimate. An informal worker earning the same wage still faces greater consumption risk.

---

## Critique of Existing Literature

### Ulyssea (2018) — Structural firm-side model
**Conceptual limitations for this project:**
- Welfare = TFP + aggregate output, not household-level consumption smoothing
- Model is firm-centered; intensive margin (formal firm hires informal workers) matters for Brazil but less for Russia where self-employment dominates
- Sorting into informality driven by firm productivity; does not model voluntary exit by skilled self-employed (exit vs. exclusion distinction per Maloney)

**Practical limitations:**
- Requires matched employer-employee data (RAIS in Brazil) — we have RLMS (household-level)
- Structural estimation calibrated to Brazil (enforcement technology, registration costs) — no direct transfer to Russia
- Counterfactual welfare via structural model is informationally very demanding

### Search Models (Boeri-Garibaldi, Charlot et al.)
**Conceptual limitations:**
- Welfare = wage differential between sectors; does not capture consumption smoothing channel (missing social insurance → consumption volatility)
- Assume informality arises from labor market frictions, not voluntary choice — incorrect for Russia where many informal self-employed choose exit (Maloney framework)
- Low unemployment + high informality describes Russia well, but M-P models explain this poorly without strong assumptions on sector-specific matching efficiency

**Practical limitations:**
- Require high-frequency flow data (formal→informal→unemployment transitions) — RLMS provides annual observations, making matching function estimation difficult
- Welfare gain from formalization = wage gap + unemployment risk, but excludes consumption variance from missing pension contributions, sick leave, etc.

---

## Theoretical Framework
**Reduced-form welfare measure** (not structural model):
- Informality → no social insurance → higher consumption variance → welfare cost
- This channel is missed by wage-based approaches

**Key references for theoretical motivation:**
- Townsend (1994) — consumption insurance test
- Morduch (1995) — income smoothing
- Ulyssea (2018) — standard structural approach (we complement)
- Meghir, Narita & Robin (2015) — standard approach we extend

---

## Data Available

### Treatment variable: NPD staggered rollout
- **`data/npd_monthly_by_region_panel.csv`** — 89 regions × 64 months (Mar 2020 – Feb 2026), NPD registrant counts with phys/IE split. Downloaded from rmsp.nalog.ru official XLSX files.
- **`data/npd_rollout_by_region.csv`** — exact NPD start dates for all 85 regions, 8 distinct treatment cohorts (Jan 2019 – Oct 2020). Suitable for Callaway-Sant'Anna or Sun-Abraham estimator.
- **`data/npd_monthly_national.csv`** — 83 months of national revenue, taxes, checks from FNS API (Feb 2019 – Dec 2025).

### Outcome variable: RLMS-HSE (cleaned)
- **`data/cleaned/rlms_informality_panel.pkl`** — cleaned panel (268K obs, 43K individuals, 2010–2023). Merges IND labor/informality with HH consumption. Built by `code/01_clean_rlms.py`.
- **`data/RLMS_HH_1994_2023.dta`** — household-level file (165K obs, 2011 vars). Symlinked from OneDrive.
- **`data/RLMS_IND_1994_2023.dta`** — individual-level file (441K obs, 3114 vars). Symlinked from OneDrive.
- **Merge keys:** `id_h` (household ID), `id_w` (wave → year), `region` across HH and IND files.
- **RLMS variable corrections:** `e5` is yes/no (not amount); `e6` is empty in 2010–2023 (use `e6_1`/`e6_2` for adult/child clothing cost); `e10` is yes/no (use `e11` for utilities cost); `e18` is yes/no (not amount); `f2` is entirely empty in 2010–2023.

### Key numbers
- NPD registrants grew from 564K (Mar 2020) → 15.9M (Feb 2026)
- Moscow alone: 256K → 2.3M
- Staggered rollout: 4 pilots (Jan 2019) → 23 regions (Jan 2020) → all 85 (by Oct 2020)

See `data/DATA_SOURCES.md` for full documentation of sources, API endpoints, and download methods.

---

## Empirical Strategy

### Identification — NPD Staggered Rollout DiD
The NPD regime was introduced in waves:
1. **Jan 2019:** 4 pilots (Moscow, MO, Kaluga, Tatarstan)
2. **Jan 2020:** 19 more regions
3. **Jul 2020:** ~51 regions (bulk expansion)
4. **Jul–Oct 2020:** 11 stragglers with exact dates (Adygea Jul 3 → Ingushetia Oct 19)

This is a classic **staggered adoption design**. Pre-period (pre-2019) and 8 distinct adoption cohorts provide clean variation for DiD.

**Estimator:** Callaway-Sant'Anna (2021) or Sun-Abraham (2021) to handle heterogeneous treatment effects across cohorts.

**Treatment intensity:** Regional NPD registration rate (from panel data) as continuous treatment measure.

### Outcome variables (from RLMS)
1. **Consumption growth variance** by employment status (formal wage / informal wage / self-employed)
2. **Townsend insurance test:** regress Δ ln(consumption) on Δ ln(income) by group — informal workers should show higher pass-through (worse insurance)
3. **Dynamics:** how does the welfare gap evolve over the rollout period?

### Alternative identification strategies
1. Regional variation in tax audit intensity (Rosstat / FNS data)
2. Sectoral variation in "forced informality" (construction, agriculture)
3. Tax administration reform 2004–2006
4. Matching + DID on sector transitions within RLMS panel

---

## Work Plan

### ~~Step 1: Theoretical Framework~~ DONE
- Reduced-form welfare = consumption variance / Townsend pass-through

### ~~Step 2: Identification Strategy~~ DONE
- NPD staggered rollout as primary identification
- TWFE DiD with clustered SEs (32 oblasts)

### ~~Step 3: Clean RLMS-HSE Data~~ DONE
- `code/01_clean_rlms.py`: merges HH consumption (`e4`, `e6_1/e6_2`, `e11`) with IND labor status via `id_h` + `id_w`
- Adds PSU→oblast mapping, NPD rollout treatment, consumption growth (Δ ln c)
- Output: `data/cleaned/rlms_informality_panel.pkl` (268K obs, 43K individuals, 2010–2023)

### ~~Step 4: Descriptive Facts~~ DONE
- `code/02_descriptive_stats.py`: summary stats, consumption variance, Townsend test, distributions
- See Empirical Results below

### ~~Step 4b: Main DiD Estimation~~ DONE
- `code/03_main_did.py`: Townsend DiD, variance DiD, event studies, cohort analysis
- See Empirical Results below

### Step 5: Write Introduction + Related Literature
- Position against Ulyssea 2018 and search models
- Articulate why consumption-based approach adds value

### Step 6: Robustness & Extensions
- Callaway-Sant'Anna estimator in R (proper staggered DiD)
- Treatment intensity: NPD registration rate (continuous) instead of binary post_NPD
- Wild cluster bootstrap for inference with 32 clusters
- Heterogeneity: by region, age, education, sector

### Step 7: Journal and Conference Targeting
- **Target journals:** Journal of Comparative Economics (Russian transition context), Review of Development Economics
- **Conferences:** SOLE 2026, NBER Summer Institute (Labor Studies) — submit extended abstract once descriptive results are ready

---

## Empirical Results (March 2026)

### Data Summary
- **Source**: RLMS 2010–2023, merged IND + HH
- **Sample**: 268,085 obs, 43,280 individuals, 13,765 households, 32 oblasts
- **Consumption**: food (`e4`) + clothing (`e6_1`+`e6_2`)/3 + utilities (`e11`), monthly RUB
- **Mean consumption**: 14K (2010) → 35K (2023) RUB/month
- **Employment composition** (among employed): ~86% formal wage, ~7% informal wage, ~7% self-employed
- **NPD coverage**: 100% of obs matched to NPD region; 73K post-NPD observations
- **NPD cohorts in RLMS**: early (2019, 4 oblasts: Moscow, MO, Kaluga, Tatarstan), late (2020, 28 oblasts)

### Descriptive: Consumption Variance by Employment Type

| Employment type | N | Mean Δln(c) | Var Δln(c) | Variance ratio vs formal |
|---|---|---|---|---|
| **Formal wage** | 75,239 | 0.080 | 0.281 | 1.00 |
| **Informal wage** | 5,785 | 0.084 | 0.316 | **1.124** |
| **Self-employed** | 5,941 | 0.073 | 0.270 | 0.962 |

Levene's test for equality of variances: p=0.001 (significant).
Informal wage workers have 12.4% higher consumption growth variance than formal.

### Descriptive: Townsend Insurance Test (Pooled)

Regress Δln(c) on Δln(w) by employment type. β=0 is full insurance; β=1 is no insurance.

| Employment type | β (pass-through) | SE | p-value | R² |
|---|---|---|---|---|
| **Formal wage** | 0.064*** | 0.004 | 0.000 | 0.003 |
| **Informal wage** | **0.116***** | 0.016 | 0.000 | 0.013 |
| **Self-employed** | 0.085*** | 0.016 | 0.000 | 0.006 |

**Key finding**: Informal wage workers have 1.8× the income-consumption pass-through of formal workers. Their consumption is significantly less insulated from income shocks.

### Townsend Test: Pre vs Post NPD

| Period | Formal β | Informal wage β | Self-employed β |
|---|---|---|---|
| **Pre-NPD (2010–2018)** | 0.062*** | **0.137***** | 0.090*** |
| **Post-NPD (2019–2023)** | 0.070*** | 0.062 (p=0.02) | 0.074 (p=0.006) |

**Key finding**: Informal wage pass-through drops from 0.137 to 0.062 post-NPD — essentially **converging to formal levels**. Suggestive of improved consumption insurance after NPD availability.

### Main DiD: Townsend Triple-Difference

Δln(c) = β₁ Δln(w) + β₂ Δln(w)×informal + β₃ Δln(w)×post_NPD + **β₄ Δln(w)×informal×post_NPD** + controls + year FE

| Variable | Coefficient | SE | |
|---|---|---|---|
| Δln(w) | 0.060*** | 0.005 | Formal pre-NPD pass-through |
| Δln(w) × informal | 0.048*** | 0.013 | Informal premium (pre-NPD) |
| Δln(w) × post_NPD | 0.009 | 0.011 | NPD effect on formal |
| **Δln(w) × informal × post_NPD** | **−0.042** | **0.036** | **NPD effect on informal insurance** |

N=73,435. Clustered SE at 32 oblasts. R²=0.007.

**Interpretation**: β₄ = −0.042 is the **correct sign and economically meaningful** — NPD cuts the informal pass-through by ~4pp, nearly closing the gap with formal workers. But **not statistically significant** (p=0.25) due to power constraints: only 32 clusters and ~4,100 informal wage workers in the Townsend sample.

### DiD by NPD Cohort

| Cohort | β₄ (triple-diff) | SE | N oblasts |
|---|---|---|---|
| Early (2019 pilots) | −0.055 | 0.078 | 4 |
| Late (2020 bulk) | −0.049 | 0.043 | 28 |

Consistent direction across cohorts. Late cohort closer to significance with more clusters.

### DiD by Employment Type (separate regressions)

| Type | Pre-NPD β | Post-NPD β | Change | SE on change |
|---|---|---|---|---|
| Formal wage | 0.059 | 0.068 | +0.009 | 0.011 |
| **Informal wage** | **0.128** | **0.068** | **−0.060** | 0.043 |
| Self-employed | 0.083 | 0.091 | +0.008 | 0.038 |

**Most compelling result**: Informal wage pass-through drops from 0.128 to 0.068, converging exactly to the formal level (0.068). The effect is concentrated entirely in informal wage workers, not self-employed.

### Consumption Variance DiD

Regress (Δln c)² on informal × post_NPD:

| Variable | Coefficient | SE |
|---|---|---|
| is_informal | 0.006 | 0.018 |
| post_NPD | −0.009 | 0.032 |
| informal × post_NPD | 0.021 | 0.023 |

Not significant. Variance DiD is noisier than the Townsend pass-through approach.

### Informality Rate DiD

NPD has **zero effect** on RLMS-measured informality: β=0.000 (p=0.97). NPD creates a new legal "self-employed" category that does not map onto RLMS survey-based informality definitions (no_contract, envelope wages). This is expected — NPD is a tax regime, not a labor contract.

### Event Study: Townsend Pass-Through

| Event time | Formal β | Informal β | Gap |
|---|---|---|---|
| t−5 | 0.050 | 0.110 | +0.061 |
| t−4 | 0.092 | 0.115 | +0.023 |
| t−3 | 0.086 | 0.207 | +0.121 |
| t−2 | 0.055 | 0.061 | +0.006 |
| t−1 | 0.061 | 0.021 | −0.040 |
| t=0 | 0.049 | 0.065 | +0.016 |
| t+1 | 0.065 | 0.036 | −0.029 |
| t+2 | 0.057 | 0.057 | +0.000 |
| t+3 | 0.110 | 0.149 | +0.040 |
| t+4 | 0.083 | 0.057 | −0.026 |

Pre-NPD: informal consistently above formal (except t−1, t−2). Post-NPD: gap closes, informal fluctuates around formal level. CIs are wide for informal due to small N per year.

### Power and Limitations

1. **32 clusters** (oblasts) — tight constraint for clustered inference. Wild cluster bootstrap recommended for robustness.
2. **NPD cohort structure**: RLMS maps to only 2 effective NPD cohorts (2019: 4 oblasts; 2020: 28 oblasts). Limited cross-cohort variation vs. the 8 actual cohorts.
3. **Informal sample size**: ~4,100 informal wage workers in the Townsend sample. The economically meaningful −0.042 triple-diff requires ~2× sample for significance at 5%.
4. **Consumption measurement**: HH-level (shared within household), not individual. All members of same HH have identical consumption growth.
5. **Wages**: only observed for employed; self-employed "wages" may understate true income.

### Summary Assessment

The **descriptive evidence is strong**: informal workers face significantly worse consumption insurance (β=0.116 vs 0.064, p<0.001), and this gap narrows dramatically post-NPD. The **DiD formalizes** this with the correct sign and meaningful magnitude (−0.042), but lacks statistical power due to 32 clusters and small informal sample. The strongest presentation may be the **by-type comparison** (informal pass-through 0.128→0.068, converging to formal) combined with the event study showing gap closure.

---

## Positioning Statement
Both structural (Ulyssea) and search (Boeri-Garibaldi) approaches identify welfare through wages or TFP. Our contribution: **welfare cost = consumption volatility**, which standard wage-based models underestimate because an informal worker can earn the same wage but bear greater consumption risk. This is closer to the insurance channel in dualistic models but with direct empirical measurement.

---

## File Structure

```
/code/
  01_clean_rlms.py           ← Load RLMS IND+HH, merge, add regions/NPD, save panel
  02_descriptive_stats.py    ← Summary stats, Townsend test, variance analysis, distributions
  03_main_did.py             ← Townsend DiD, variance DiD, event studies, cohort analysis
/data/
  /cleaned/                  ← rlms_informality_panel.csv/.pkl (from 01_clean_rlms.py)
  npd_rollout_by_region.csv  ← NPD start dates for all 85 regions
  npd_monthly_by_region_panel.csv
  RLMS_HH_1994_2023.dta     ← symlink to OneDrive
  RLMS_IND_1994_2023.dta     ← symlink to OneDrive
/output/
  /tables/                   ← CSV results
  /figures/                  ← PNG plots
/drafts/                     ← paper versions
```

## Running the Analysis

```bash
cd ~/Dropbox/_Research/Labor\ Informality
python3 code/01_clean_rlms.py          # ~3 min, builds panel
python3 code/02_descriptive_stats.py   # ~30 sec, tables + figures
python3 code/03_main_did.py            # ~30 sec, DiD estimation
```

## Style and Conventions
- Working language: English (paper), Russian (notes/discussions)
- Data work: Python (pandas, statsmodels, linearmodels)
- Paper format: LaTeX
- RLMS loading: always use `convert_categoricals=False` to avoid crashes on duplicate Stata value labels
- Missing codes: replace ≥ 99999990 with NaN (RLMS convention)
- Wave-to-year: `id_w` 19=2010, ..., 32=2023
- PSU→oblast mapping from Credit Market project (`RLMSsites_pubuse.dta`)
