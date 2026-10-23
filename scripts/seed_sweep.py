"""Re-runs every analysis on several data seeds to check the conclusions are not one lucky draw.

Usage: python scripts/seed_sweep.py 7 11 23 42"""

import sys

from commercial_ml import models
from commercial_ml.data import generate

for seed in [int(s) for s in sys.argv[1:]] or [7, 11, 23]:
    ds = generate(seed=seed)
    seg, fc, tg = models.segment(ds), models.forecast_backtest(ds), models.targeting(ds)
    cp, nba = models.campaign_effect(ds), models.next_best_action(ds)
    p = nba["policies"]
    print(
        f"seed {seed}: ARI {seg['ari_vs_true_archetypes']:.2f} (k={seg['k']}) | "
        f"WAPE HW {fc['holt_winters']['wape']:.2%} vs SN {fc['seasonal_naive']['wape']:.2%} vs GBM {fc['gbm']['wape']:.2%} | "
        f"targeting AUC LR {tg['logistic_regression']['auc']:.3f} vs rule {tg['rule_past_open_rate']['auc']:.3f} | "
        f"campaign true {cp['true_att']:.2f} naive {cp['naive_post_difference']:.2f} IPW-DiD {cp['ipw_did']:.2f} {cp['ipw_did_95ci']} | "
        f"NBA {p['uplift_model_nba']['true_profit_per_hcp']:.2f} (IPS {p['uplift_model_nba']['ips_profit_per_hcp']:.2f}) "
        f"vs {nba['best_single_channel']} {p[nba['best_single_channel']]['true_profit_per_hcp']:.2f} "
        f"(IPS {p[nba['best_single_channel']]['ips_profit_per_hcp']:.2f}), {nba['share_of_oracle_profit']:.0%} of oracle"
    )
