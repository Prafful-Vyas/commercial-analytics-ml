"""Synthetic commercial dataset with planted ground truth.

Simulates healthcare professionals (HCPs) prescribing one brand over 156 weeks, with rep visits,
emails and webinars whose effects carry over (adstock) and differ by HCP. Two things make it a
realistic test bed for causal methods:

- Promotion is confounded: reps visit high-volume HCPs more often, and a mid-year email campaign
  targets digitally engaged HCPs, so naive comparisons are biased.
- Weeks 140-151 are a randomized channel test (none / rep / email / webinar), with regular
  promotion paused, which is the clean data a next-best-action model is trained and judged on.

Because every effect is planted, the true campaign ATT and each HCP's true incremental response
per channel are known, so the models can be scored against reality rather than only against
each other."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS = 156
CHANNELS = ("rep", "email", "webinar")
ARCHETYPES = ("high-volume loyal", "digital adopter", "growth potential", "low engagement")
SPECIALTIES = ("Cardiology", "Endocrinology", "General Practice", "Internal Medicine")
REGIONS = ("North", "South", "East", "West")

ADSTOCK_DECAY = 0.6
CAMPAIGN_WEEKS = range(120, 124)  # one campaign email per week
CAMPAIGN_EVAL_WEEKS = range(120, 132)
TEST_TOUCH_WEEKS = range(140, 144)  # randomized arm: one touch per week of its channel
TEST_OUTCOME_WEEKS = range(140, 152)
ARMS = ("none", *CHANNELS)


@dataclass
class Dataset:
    hcps: pd.DataFrame  # one row per HCP: observable attributes + hidden truth columns prefixed "true_"
    weekly: pd.DataFrame  # one row per HCP-week: rx, touches, email opens, webinar attendance
    truth: dict  # campaign ATT and per-HCP true incremental Rx per channel in the test


def _adstock_weight_sum(n_touch_weeks: int, n_outcome_weeks: int) -> float:
    """Total carried-over effect units from one touch per week for n_touch_weeks, observed over
    n_outcome_weeks starting at the first touch."""
    stock, total = 0.0, 0.0
    for w in range(n_outcome_weeks):
        stock = ADSTOCK_DECAY * stock + (1.0 if w < n_touch_weeks else 0.0)
        total += stock
    return total


def generate(n_hcp: int = 3000, seed: int = 7) -> Dataset:
    rng = np.random.default_rng(seed)
    archetype = rng.choice(4, size=n_hcp, p=[0.2, 0.3, 0.25, 0.25])
    volume = np.exp(rng.normal(np.array([4.0, 3.3, 3.4, 2.6])[archetype], 0.35))  # patients/week on the condition
    digital = np.where(
        archetype == 1, rng.beta(6, 2, n_hcp), np.where(archetype == 3, rng.beta(1.5, 5, n_hcp), rng.beta(2, 4, n_hcp))
    )
    rep_resp = np.where(archetype == 0, rng.uniform(0.8, 1.2, n_hcp), rng.uniform(0.2, 0.6, n_hcp))
    share = np.array([0.18, 0.12, 0.08, 0.05])[archetype] * rng.uniform(0.85, 1.15, n_hcp)
    weekly_trend = np.array([0.0, 0.0005, 0.004, -0.001])[archetype]

    # Per-unit-adstock effect in Rx/week. Rep visits work on high-volume, rep-responsive HCPs;
    # digital channels work on digitally engaged ones.
    beta = {
        "rep": 0.020 * volume * rep_resp,
        "email": 0.012 * volume * digital**2,
        "webinar": 0.030 * volume * digital,
    }
    beta_campaign = 0.016 * volume * digital

    vol_pct = pd.Series(volume).rank(pct=True).to_numpy()
    p_touch = {
        "rep": 0.03 + 0.22 * vol_pct,  # confounding: reps chase volume
        "email": 0.05 + 0.20 * digital,
        "webinar": 0.01 + 0.04 * digital,
    }
    campaign_target = rng.random(n_hcp) < 1 / (1 + np.exp(-6 * (digital - 0.5)))
    arm = rng.integers(0, len(ARMS), n_hcp)

    weeks = np.arange(WEEKS)
    season = 1 + 0.12 * np.sin(2 * np.pi * weeks / 52) + 0.08 * np.exp(-(((weeks % 52) - 48) ** 2) / 8)

    stocks = {c: np.zeros(n_hcp) for c in (*CHANNELS, "campaign")}
    rows = []
    campaign_increment = np.zeros(n_hcp)
    for w in weeks:
        in_test = w in TEST_OUTCOME_WEEKS
        touches = {}
        for c in CHANNELS:
            regular = (rng.random(n_hcp) < p_touch[c]) & (not in_test)
            test = (arm == ARMS.index(c)) & (w in TEST_TOUCH_WEEKS)
            touches[c] = (regular | test).astype(int)
        touches["campaign"] = (campaign_target & (w in CAMPAIGN_WEEKS)).astype(int)
        for c in stocks:
            stocks[c] = ADSTOCK_DECAY * stocks[c] + touches[c]

        base = volume * share * (1 + weekly_trend) ** w * season[w]
        promo = sum(beta[c] * stocks[c] for c in CHANNELS) + beta_campaign * stocks["campaign"]
        if w in CAMPAIGN_EVAL_WEEKS:
            campaign_increment += beta_campaign * stocks["campaign"]
        rx = rng.poisson(base + promo)
        email_open = touches["email"] * (rng.random(n_hcp) < 0.15 + 0.7 * digital)
        webinar_attend = touches["webinar"] * (rng.random(n_hcp) < 0.3 + 0.6 * digital)
        rows.append(
            pd.DataFrame(
                {
                    "hcp_id": np.arange(n_hcp),
                    "week": w,
                    "rx": rx,
                    "rep_visits": touches["rep"],
                    "emails": touches["email"] + touches["campaign"],
                    "campaign_email": touches["campaign"],
                    "email_opens": email_open,
                    "webinars": touches["webinar"],
                    "webinar_attended": webinar_attend,
                }
            )
        )

    weight = _adstock_weight_sum(len(TEST_TOUCH_WEEKS), len(TEST_OUTCOME_WEEKS))
    true_uplift = pd.DataFrame({c: beta[c] * weight for c in CHANNELS})
    true_uplift.insert(0, "none", 0.0)

    hcps = pd.DataFrame(
        {
            "hcp_id": np.arange(n_hcp),
            "specialty": rng.choice(SPECIALTIES, n_hcp),
            "region": rng.choice(REGIONS, n_hcp),
            "years_in_practice": rng.integers(2, 35, n_hcp),
            "patient_volume": np.round(volume, 1),
            "campaign_target": campaign_target.astype(int),
            "test_arm": np.array(ARMS)[arm],
            "true_archetype": np.array(ARCHETYPES)[archetype],
        }
    )
    truth = {
        "campaign_att_rx_per_hcp": float(campaign_increment[campaign_target].mean()),
        "uplift_rx": true_uplift,  # true incremental Rx over the test window, per HCP and arm
    }
    return Dataset(hcps=hcps, weekly=pd.concat(rows, ignore_index=True), truth=truth)
