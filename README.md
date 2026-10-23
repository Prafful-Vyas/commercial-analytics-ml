# Commercial Analytics ML

Five commercial data-science problems on one dataset, in Python and SQL: **customer segmentation,
demand forecasting, targeting, campaign effectiveness and next best action**. The setting is
pharma commercial analytics: healthcare professionals (HCPs) prescribing a brand, reached by rep
visits, emails and webinars.

The data is synthetic **with planted ground truth**: every promotional effect, the campaign's true
impact and each HCP's true response to each channel are known. That turns the usual "the model
looks reasonable" into a hard check: did the method recover what actually happened?

| Problem | Method | Checked against |
|---|---|---|
| Segmentation | SQL features (DuckDB), k-means with k chosen by silhouette | planted HCP archetypes (adjusted Rand index) |
| Forecasting | rolling-origin backtest by region: seasonal naive, Holt-Winters (statsmodels), gradient boosting | WAPE / MAPE on held-out weeks |
| Targeting | logistic regression and gradient boosting for email engagement propensity | business rule (rank by past open rate): AUC, lift |
| Campaign effectiveness | naive comparison vs difference-in-differences vs inverse-propensity-weighted DiD, bootstrap CI | planted true ATT |
| Next best action | T-learner uplift models (ridge, CUPED-adjusted outcome) on a randomized channel test; action = max(incremental Rx x value - cost) | planted per-HCP uplift; single-channel and oracle policies; IPS estimate from observed data |

## The dataset (`data.py`)

3,000 HCPs x 156 weeks (468k rows). Each HCP has a hidden archetype (high-volume loyal, digital
adopter, growth potential, low engagement) that drives patient volume, brand share, trend, digital
affinity and rep responsiveness. Weekly Rx is Poisson around a seasonal baseline plus promotion
effects that carry over with geometric adstock (decay 0.6) and differ by HCP.

Two features make it a fair test of causal methods:

- **Confounding.** Reps visit high-volume HCPs more often, and the week-120 email campaign targets
  digitally engaged HCPs, so naive "targeted vs not" comparisons are biased.
- **A randomized test.** In weeks 140-151 regular promotion pauses and each HCP is randomly
  assigned none / rep / email / webinar (4 touches). That is the data an uplift model learns from.

Features are computed in SQL (`features.py`): trailing 13/26/52-week volumes, growth, Rx per
patient, channel response rates, recency, and a trend slope from a moving average with
`REGR_SLOPE`. A test checks the SQL against pandas, and hidden truth columns never reach the features.

## Results

`python -m commercial_ml.run` (seed 7) writes [reports/REPORT.md](reports/REPORT.md) with figures.
`python scripts/seed_sweep.py 7 11 23 42` re-runs everything on four independent datasets:

| | seed 7 | seed 11 | seed 23 | seed 42 |
|---|---|---|---|---|
| Forecast WAPE: Holt-Winters / seasonal naive / GBM | 1.94 / 2.43 / 2.52% | 2.15 / 2.42 / 2.94% | 2.52 / 2.65 / 2.86% | 2.17 / 2.83 / 3.01% |
| Targeting AUC: logistic / past-open-rate rule | 0.686 / 0.621 | 0.695 / 0.620 | 0.669 / 0.615 | 0.711 / 0.661 |
| Campaign: true ATT / naive / IPW-DiD [95% CI] | 3.02 / 0.58 / 3.35 [2.40, 4.30] | 3.21 / 5.95 / 3.38 [2.35, 4.42] | 3.12 / 3.41 / 3.62 [2.62, 4.67] | 3.18 / 2.94 / 2.81 [1.61, 3.86] |
| NBA profit / HCP: uplift policy / best single channel (share of oracle) | 2.96 / 2.63 (79%) | 3.37 / 2.75 (85%) | 3.46 / 2.81 (87%) | 3.34 / 2.79 (83%) |
| Segmentation ARI (k chosen) | 0.35 (5) | 0.28 (2) | 0.30 (2) | 0.30 (2) |

What this shows, and what it doesn't:

- **Campaign effectiveness**: the naive estimate swings from 0.58 to 5.95 against a true effect of
  about 3.1 depending on how the confounding falls; the IPW-DiD 95% interval contained the true
  effect on all four datasets. Plain DiD was also close here because the confounding is mostly
  fixed per-HCP differences.
- **Next best action**: the uplift policy beat every single-channel policy on all four datasets,
  capturing 79-87% of the oracle's profit. Gradient-boosted T-learners were tried first and were
  too noisy (they lost to "webinar for everyone" on seed 7); the CUPED adjustment plus ridge
  outcome models fixed that. That choice was made by comparing variants on seeds 7 and 11, then
  confirmed on 23 and 42.
- **The IPS estimate is noisy.** Estimated only from the observed randomized outcomes (~375
  HCPs per arm), it would have ranked "webinar for everyone" above the model on two of four
  datasets. With real data, where the truth is unknown, a larger test or a longer window would be
  needed before trusting a policy comparison.
- **Forecasting**: Holt-Winters beat seasonal naive on every dataset (11-23% lower WAPE). The
  gradient-boosting model, a correction on top of seasonal naive with lags and planned promotion,
  did **not** beat the baseline: region totals sit near the Poisson noise floor (~2% WAPE) and
  aggregate promotion barely varies, so there is little left to learn.
- **Targeting**: pooling features beats ranking by an HCP's own past open rate, which is noisy
  when only a few emails were sent (top-20% lift 1.63 vs 1.18 on seed 7). The ceiling is modest
  (AUC ~0.7) because opens are random given affinity.
- **Segmentation is the weakest result.** The archetypes overlap by design, silhouette often
  prefers k=2, and agreement with the planted archetypes is moderate (ARI ~0.3). The segment
  profiles are still interpretable (high-volume, digital-engaged, low-engagement groups).

## Run it

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m commercial_ml.run                       # writes reports/
python scripts/seed_sweep.py 7 11 23 42
pytest -q                                         # 8 tests, ~15 s
```

## Tests

Each analysis is tested against the planted truth on a separate 2,000-HCP dataset: SQL features
match pandas and never see hidden columns; segmentation recovers structure (ARI > 0.2);
Holt-Winters beats seasonal naive; the targeting model beats the rule; DiD and IPW-DiD beat the
naive estimate and the CI covers the truth; the uplift policy beats every single-channel policy.
The generator is deterministic and pauses regular promotion during the randomized test.

## Layout

```
src/commercial_ml/
  data.py       synthetic HCP x week data with planted effects and ground truth
  features.py   DuckDB SQL feature pipeline (window functions, REGR_SLOPE)
  models.py     segmentation, forecasting, targeting, campaign effect, next best action
  run.py        runs everything, writes reports/REPORT.md, results.json and figures
scripts/seed_sweep.py
tests/test_models.py
```

## Limits

Synthetic data is cleaner than real commercial data: no data-quality problems, no competitor
activity, no formulary changes, and the true effect structure is simpler than reality. The methods
are standard and transfer directly; the numbers above describe this simulation, not a real brand.
