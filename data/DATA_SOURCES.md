# Data Sources — NPD Self-Employment Statistics

## Key Datasets

### 1. Monthly Region-Level Panel (primary)
| File | Contents | Coverage | Granularity |
|------|----------|----------|-------------|
| **`npd_monthly_by_region_panel.csv`** | NPD registrants by region: total, physical persons, individual entrepreneurs | Mar 2020 – Feb 2026 (64 months) | region × month |
| `rmsp_monthly_xlsx/npd_YYYY-MM.xlsx` | Raw XLSX source files (64 files) | Same | region × month |

**Variables:** `year`, `month`, `region_code`, `region_name`, `npd_total`, `npd_physical`, `npd_ie`
**Dimensions:** 5,619 rows, 89 regions, 64 months
**Source:** rmsp.nalog.ru/statistics2.html — POST to `statistics-npd.xlsx` with `statType=all&statNpdDate=DD.MM.YYYY`

### 2. National Monthly Revenue/Tax/Checks (from FNS API)
| File | Contents | Coverage | Granularity |
|------|----------|----------|-------------|
| **`npd_monthly_national.csv`** | Revenue, taxes, checks — split by entrepreneur vs legal entity | Feb 2019 – Dec 2025 (83 months) | national × month |

**Variables:** `year`, `month`, `revenue_total`, `revenue_entrepreneur`, `revenue_legal`, `tax_total`, `tax_entrepreneur`, `tax_legal`, `checks_total`, `checks_entrepreneur`, `checks_legal`
**Source:** Undocumented API at `geochecki-vpd.nalog.gov.ru/api/api/v1/self-employers/summary/YYYY-MM` (auth disabled, public access)

### 3. Staggered Rollout Schedule (for DiD)
| File | Contents | Coverage |
|------|----------|----------|
| **`npd_rollout_by_region.csv`** | All 85 regions with exact NPD start dates and wave assignment | Jan 2019 – Oct 2020 |

**8 distinct treatment cohorts:**
- Wave 1 (Jan 1, 2019): Moscow, Moscow Oblast, Kaluga Oblast, Tatarstan — 4 pilot regions
- Wave 2 (Jan 1, 2020): SPb, Bashkortostan, + 17 oblasts/krais/AOs — 19 regions
- Wave 3 (Jul 1, 2020): 51 regions (the bulk)
- Wave 3a (Jul 3): Adygea
- Wave 3b (Jul 9): Ulyanovsk Oblast, Tyva
- Wave 3c (Jul 24): North Ossetia-Alania
- Wave 3d (Aug 1): Kalmykia, Vologda, Magadan
- Wave 3e (Sep 1): Chechnya, Karachay-Cherkessia, Zabaykalsky Krai
- Wave 3f (Sep 5): Tambov Oblast
- Wave 3g (Sep 6): Mari El
- Wave 4 (Oct 19, 2020): Ingushetia — last to join

**Sources:** [Glavkniga](https://glavkniga.ru/situations/k509498), [FNS Chechnya](https://www.nalog.gov.ru/rn20/news/activities_fts/9979201/), [RG](https://rg.ru/amp/2020/07/08/nalogovyj-rezhim-samozaniatosti-stal-dostupen-s-1-iiulia-vo-vseh-regionah.html)

### 4. RLMS-HSE Household Panel (outcome variable)
| File | Contents | Coverage | Size |
|------|----------|----------|------|
| **`RLMS_HH_1994_2023.dta`** | Household-level data: consumption, income, assets | 1994–2023 (rounds 3–32) | 2.5 GB, 165K obs, 2011 vars |
| **`RLMS_IND_1994_2023.dta`** | Individual-level data: labor, wages, demographics | 1994–2023 (rounds 3–32) | 10 GB, 441K obs, 3114 vars |
| `rlms_panel_broken_inst.csv` | Pre-cleaned panel (labor vars only, no consumption) | 1994–2023 | 37 MB, 268K obs |

All files are symlinks to originals in OneDrive / Broken Institutions project.

**Key consumption variables (HH file):**
- `e4` — food expenditure (last 30 days)
- `e5`, `e6` — clothing expenditure
- `e10`, `e11` — utility payments
- `e16`, `e17` — savings
- `e18` — transfers received
- `f2` — subjective minimum living cost

**Key labor variables (IND file):**
- `j1` — currently working (yes/no)
- `j4_1` — employment type (formal wage / informal / self-employed)
- `j10` — monthly wages
- `j8` — hours worked

**Merge keys:** `id_h` (household), `id_w` (wave = year), `region`

**Source:** RLMS-HSE (Russia Longitudinal Monitoring Survey), HSE/UNC Chapel Hill. Raw .dta files downloaded from official RLMS website.

**TODO:** Write cleaning script to merge HH consumption with IND labor status. Existing cleaned panel from Broken Institutions project has labor variables but not consumption.

### 5. Supplementary Files
| File | Contents | Source |
|------|----------|--------|
| `npd_self_employed_by_region_2022.csv` | Annual snapshot with phys/IE split | sznpd.ru |
| `npd_self_employed_by_region_2023.csv` | Annual snapshot | sznpd.ru |
| `npd_self_employed_by_region_2024.csv` | Annual snapshot with phys/IE split | sznpd.ru |
| `npd_self_employed_by_region_2025.csv` | Annual snapshot | sznpd.ru |
| `npd_self_employed_by_region_2021.csv` | Partial (national + top 5) | sznpd.ru |
| `npd_rollout_dates.csv` | Simplified 4-wave summary | Various |
| `npd_aggregate_statistics.csv` | Misc national stats from konsol.pro | konsol.pro |
| `npd_sectors_2025.csv` | Top sectors by self-employed count | konsol.pro |

Note: The annual CSVs are now **superseded** by the monthly panel — kept for cross-validation only.

---

## Original Sources and Access Methods

### rmsp.nalog.ru (Unified SME Registry) — MAIN SOURCE
- Monthly XLSX downloads via POST: `curl -X POST "https://rmsp.nalog.ru/statistics-npd.xlsx" -d "statType=all&statNpdDate=31.01.2026"`
- `statNpdDate` format: `DD.MM.YYYY` (end of month)
- Available date range: quarterly from Mar 2020, monthly from Mar 2021, through latest
- Other stat types available: `sz_m`, `sz_q`, `cnt_workers_m`, `cnt_workers_q`, `ip_m`, `ip_q`, `cnt_msp_workers_m`, `cnt_msp_subjects_m`, `cnt_new`
- No authentication required

### geochecki-vpd.nalog.gov.ru (FNS Presentation Analytics) — SUPPLEMENTARY
- API base: `https://geochecki-vpd.nalog.gov.ru/api/api/v1/self-employers/summary/YYYY-MM`
- Returns JSON with national-level revenue, taxes, checks (split by entrepreneur vs legal entity)
- `REACT_AUTH_ENABLED: "false"` — no auth needed
- **Region filtering (`?region=XX`) does NOT work** — parameter is ignored, always returns national totals
- JS source: `github.com/cyber-orkz/geochecks-ui`

### vpd.nalog.gov.ru (Data Delivery Platform) — NOT ACCESSED
- Requires UKEP (qualified electronic signature) for API access
- May contain more granular data (revenue by region, etc.)
- JS-rendered SPA, no public API endpoints found

### sznpd.ru — SUPPLEMENTARY
- Unofficial aggregator citing FNS data
- Annual regional tables: `https://sznpd.ru/statistika-samozanyatosti-na-nachalo-YYYY/`
- Now superseded by rmsp monthly panel

### konsol.pro — SUPPLEMENTARY
- Blog-style aggregation of FNS stats: `https://konsol.pro/blog/samozanyatye-v-tsifrah`
- Monthly national aggregates, sector breakdowns

---

## Still Needed

- [x] ~~RLMS-HSE data~~ — available (symlinked from OneDrive). Needs new cleaning script for consumption + labor merge.
- [ ] Revenue/tax **by region** (not available from geochecki API; may require vpd.nalog.gov.ru with UKEP or analytic.nalog.gov.ru)
- [ ] Pre-2020 region data (rmsp only starts Mar 2020; 2019 pilot-region data would strengthen pre-trends)
