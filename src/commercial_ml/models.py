"""The five analyses: segmentation, forecasting, targeting, campaign effectiveness, next best action.
Each returns plain dicts/DataFrames so the report and the tests can check them."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from commercial_ml.data import ARMS, CAMPAIGN_EVAL_WEEKS, CAMPAIGN_WEEKS, TEST_OUTCOME_WEEKS, Dataset
from commercial_ml.features import hcp_features, model_matrix

# ---------------------------------------------------------------- segmentation

SEGMENT_FEATURES = ["patient_volume", "rx_per_patient", "rx_trend_slope", "email_open_rate", "webinar_attend_rate"]


def segment(ds: Dataset, as_of: int = 140, k_range: range = range(2, 9), seed: int = 0) -> dict:
    """K-means on behavioural features (log volume, share of patients, trend, digital engagement),
    with k chosen by silhouette score. Scored against the planted archetypes with ARI."""
    f = hcp_features(ds.hcps, ds.weekly, as_of)
    x = f[SEGMENT_FEATURES].copy()
    x["patient_volume"] = np.log(x["patient_volume"])
    xs = StandardScaler().fit_transform(x)
    scores = {}
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(xs)
        scores[k] = silhouette_score(xs, labels, sample_size=min(2000, len(xs)), random_state=seed)
    best_k = max(scores, key=scores.get)
    km = KMeans(best_k, n_init=10, random_state=seed).fit(xs)
    f["segment"] = km.labels_
    profile = f.groupby("segment")[SEGMENT_FEATURES + ["rx_26w"]].mean().round(3)
    profile["hcps"] = f["segment"].value_counts().sort_index()
    return {
        "k": best_k,
        "silhouette_by_k": {k: round(v, 3) for k, v in scores.items()},
        "ari_vs_true_archetypes": round(adjusted_rand_score(ds.hcps["true_archetype"], f["segment"]), 3),
        "profile": profile,
        "assignments": f[["hcp_id", "segment"]],
    }


# ---------------------------------------------------------------- forecasting


def _wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / np.abs(actual).sum())


def forecast_backtest(ds: Dataset, horizon: int = 4, first_origin: int = 104, last_origin: int = 136) -> dict:
    """Rolling-origin backtest of weekly brand Rx by region, horizons 1-4, before the test window.
    Seasonal naive and Holt-Winters are baselines; the GBM is a direct multi-horizon model that
    corrects the seasonal-naive forecast using lags, season and the planned promotion volume."""
    wk = ds.weekly.merge(ds.hcps[["hcp_id", "region"]], on="hcp_id")
    series = wk.groupby(["region", "week"]).agg(rx=("rx", "sum"), touches=("rep_visits", "sum")).reset_index()
    series["touches"] += wk.groupby(["region", "week"])[["emails", "webinars"]].sum().sum(axis=1).to_numpy()

    def gbm_rows(s: pd.DataFrame, origins: list[int]) -> pd.DataFrame:
        y, t = s["rx"].to_numpy(), s["touches"].to_numpy()
        out = []
        for o in origins:
            for h in range(1, horizon + 1):
                target = o + h - 1
                out.append(
                    {
                        "origin": o,
                        "h": h,
                        "target": target,
                        "y": y[target],
                        "lag1": y[o - 1],
                        "lag2": y[o - 2],
                        "lag3": y[o - 3],
                        "lag4": y[o - 4],
                        "ma8": y[o - 8 : o].mean(),
                        "last_year": y[target - 52],
                        "last_year_ratio": y[o - 4 : o].sum() / y[o - 56 : o - 52].sum(),
                        "woy_sin": np.sin(2 * np.pi * target / 52),
                        "woy_cos": np.cos(2 * np.pi * target / 52),
                        "planned_touches": t[target],
                    }
                )
        return pd.DataFrame(out)

    results = {"seasonal_naive": [], "holt_winters": [], "gbm": []}
    actuals = []
    for _region, s in series.groupby("region"):
        s = s.sort_values("week").reset_index(drop=True)
        y = s["rx"].to_numpy(dtype=float)
        for o in range(first_origin, last_origin + 1, horizon):
            actual = y[o : o + horizon]
            actuals.append(actual)
            results["seasonal_naive"].append(
                y[o - 52 : o - 52 + horizon] * (y[o - 4 : o].sum() / y[o - 56 : o - 52].sum())
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hw = ExponentialSmoothing(y[:o], trend="add", seasonal="mul", seasonal_periods=52).fit()
            results["holt_winters"].append(hw.forecast(horizon))
            train = gbm_rows(s, list(range(56, o - horizon + 1)))
            test = gbm_rows(s, [o])
            feats = [c for c in train.columns if c not in ("origin", "target", "y")]
            # The GBM learns a multiplicative correction to the seasonal-naive forecast (last year's
            # value scaled by year-on-year growth), which generalizes across regions and growth rates.
            base_tr = train["last_year"] * train["last_year_ratio"]
            base_te = (test["last_year"] * test["last_year_ratio"]).to_numpy()
            model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_depth=3, random_state=0)
            model.fit(train[feats], train["y"] / base_tr)
            results["gbm"].append(model.predict(test[feats]) * base_te)
    a = np.concatenate(actuals)
    return {
        name: {
            "wape": round(_wape(a, np.concatenate(p)), 4),
            "mape": round(float(np.mean(np.abs(a - np.concatenate(p)) / a)), 4),
        }
        for name, p in results.items()
    }


# ---------------------------------------------------------------- targeting


def targeting(ds: Dataset, as_of: int = 100, horizon_weeks: int = 13, engaged_rate: float = 0.75) -> dict:
    """Which HCPs will engage with email next quarter (open at least 75% of the emails they are
    sent, among HCPs sent 2+)? This decides who a digital campaign should target. The business
    rule to beat ranks HCPs by their own past open rate, which is noisy when few emails were sent;
    the models pool evidence across all behavioural and profile features."""
    f = hcp_features(ds.hcps, ds.weekly, as_of)
    fut = ds.weekly[(ds.weekly.week >= as_of) & (ds.weekly.week < as_of + horizon_weeks)].groupby("hcp_id")
    sent = (fut["emails"].sum() - fut["campaign_email"].sum()).reindex(f["hcp_id"])
    opens = fut["email_opens"].sum().reindex(f["hcp_id"])
    eligible = (sent >= 2).to_numpy()
    y = ((opens / sent.where(sent > 0)).to_numpy() >= engaged_rate).astype(int)[eligible]
    x = model_matrix(f)[eligible]
    rule = f["email_open_rate"].to_numpy()[eligible]
    x_tr, x_te, y_tr, y_te, _, rule_te = train_test_split(x, y, rule, test_size=0.3, random_state=0, stratify=y)

    def lift_at(scores: np.ndarray, frac: float = 0.2) -> float:
        n = max(1, int(len(scores) * frac))
        return float(y_te[np.argsort(-scores, kind="stable")[:n]].mean() / y_te.mean())

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
    }
    out = {
        "eligible_hcps": int(eligible.sum()),
        "engaged_share": round(float(y.mean()), 3),
        "rule_past_open_rate": {
            "auc": round(roc_auc_score(y_te, rule_te), 4),
            "lift_top20": round(lift_at(rule_te), 3),
        },
    }
    for name, m in models.items():
        p = m.fit(x_tr, y_tr).predict_proba(x_te)[:, 1]
        out[name] = {"auc": round(roc_auc_score(y_te, p), 4), "lift_top20": round(lift_at(p), 3)}
    return out


# ---------------------------------------------------------------- campaign effectiveness


def campaign_effect(ds: Dataset) -> dict:
    """Effect of the week-120 email campaign on Rx per targeted HCP over weeks 120-131.
    Targeting was confounded with digital engagement, so a naive post-period comparison is biased.
    Difference-in-differences removes fixed HCP differences; inverse-propensity weighting on
    pre-period features (ATT weights) additionally balances the groups. Each is compared with the
    planted true ATT, with a bootstrap CI for the IPW-DiD estimate."""
    pre_weeks = range(CAMPAIGN_WEEKS[0] - 12, CAMPAIGN_WEEKS[0])
    wk = ds.weekly
    pre = wk[wk.week.isin(pre_weeks)].groupby("hcp_id")["rx"].sum()
    post = wk[wk.week.isin(CAMPAIGN_EVAL_WEEKS)].groupby("hcp_id")["rx"].sum()
    # Regular promotion during the window differs by group too; it is part of what DiD must absorb.
    f = hcp_features(ds.hcps, wk, CAMPAIGN_WEEKS[0])
    treated = ds.hcps.set_index("hcp_id")["campaign_target"].reindex(f["hcp_id"]).to_numpy()
    delta = (post - pre).reindex(f["hcp_id"]).to_numpy()
    post_v = post.reindex(f["hcp_id"]).to_numpy()

    x = StandardScaler().fit_transform(model_matrix(f))
    ps = LogisticRegression(max_iter=2000).fit(x, treated).predict_proba(x)[:, 1].clip(0.02, 0.98)
    w_control = ps / (1 - ps)

    def ipw_did(idx: np.ndarray) -> float:
        t, d, w = treated[idx], delta[idx], w_control[idx]
        return float(d[t == 1].mean() - np.average(d[t == 0], weights=w[t == 0]))

    rng = np.random.default_rng(0)
    boots = [ipw_did(rng.integers(0, len(treated), len(treated))) for _ in range(300)]
    all_idx = np.arange(len(treated))
    return {
        "true_att": round(ds.truth["campaign_att_rx_per_hcp"], 3),
        "naive_post_difference": round(float(post_v[treated == 1].mean() - post_v[treated == 0].mean()), 3),
        "difference_in_differences": round(float(delta[treated == 1].mean() - delta[treated == 0].mean()), 3),
        "ipw_did": round(ipw_did(all_idx), 3),
        "ipw_did_95ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "propensity_auc": round(roc_auc_score(treated, ps), 3),
    }


# ---------------------------------------------------------------- next best action

VALUE_PER_RX = 1.0
CHANNEL_COST = {"none": 0.0, "rep": 2.5, "email": 0.2, "webinar": 1.5}  # in Rx-equivalents per test (4 touches)


def next_best_action(ds: Dataset, seed: int = 0) -> dict:
    """T-learner uplift model on the randomized channel test: one ridge outcome model per arm on a
    CUPED-adjusted outcome, trained on half the HCPs from pre-test features. (Gradient-boosted
    T-learners were compared and were noisier; see README.) The recommended action
    maximises predicted (incremental Rx x value - channel cost). Policies are scored on the
    held-out half with the planted true uplift (oracle profit) and, as a real team would have to,
    with an inverse-propensity estimate from the randomized outcomes only."""
    f = hcp_features(ds.hcps, ds.weekly, TEST_OUTCOME_WEEKS[0])
    x = model_matrix(f)
    arm = ds.hcps.set_index("hcp_id")["test_arm"].reindex(f["hcp_id"]).to_numpy()
    y = ds.weekly[ds.weekly.week.isin(TEST_OUTCOME_WEEKS)].groupby("hcp_id")["rx"].sum().reindex(f["hcp_id"]).to_numpy()
    true_uplift = ds.truth["uplift_rx"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(f)), test_size=0.5, random_state=seed, stratify=arm)

    # CUPED: subtract the part of the outcome explained by pre-period Rx (slope fitted on the
    # training half). Baseline volume differs ~100x across HCPs and would otherwise swamp the
    # few-Rx uplift signal; the adjustment is identical across arms, so differences are unbiased.
    pre = f["rx_13w"].to_numpy()
    theta = np.polyfit(pre[idx_tr], y[idx_tr], 1)[0]
    y_adj = y - theta * pre
    xs = StandardScaler().fit(x.iloc[idx_tr]).transform(x)
    pred = np.zeros((len(idx_te), len(ARMS)))
    for j, a in enumerate(ARMS):
        rows = idx_tr[arm[idx_tr] == a]
        pred[:, j] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(xs[rows], y_adj[rows]).predict(xs[idx_te])
    pred_uplift = pred - pred[:, [0]]
    cost = np.array([CHANNEL_COST[a] for a in ARMS])
    model_choice = np.argmax(pred_uplift * VALUE_PER_RX - cost, axis=1)

    true_profit = true_uplift[idx_te] * VALUE_PER_RX - cost
    oracle_choice = np.argmax(true_profit, axis=1)
    arm_te = np.array([ARMS.index(a) for a in arm[idx_te]])
    y_te = y_adj[idx_te]

    def ips_value(choice: np.ndarray) -> float:
        """Incremental profit of a policy over "none", estimated from the randomized outcomes alone.
        With uniform assignment, HCPs whose random arm matches the policy are a random quarter of
        the population, so their mean reward estimates the policy's value. CUPED-adjusted outcomes
        keep the variance manageable."""
        match = choice == arm_te
        reward = y_te * VALUE_PER_RX - cost[arm_te]
        base = reward[arm_te == 0].mean()  # "none" arm: no incremental profit
        return float((match * reward).sum() / match.sum() - base) if match.any() else float("nan")

    policies = {f"all_{a}": np.full(len(idx_te), j) for j, a in enumerate(ARMS)}
    policies["uplift_model_nba"] = model_choice
    policies["oracle"] = oracle_choice
    rows = len(idx_te)
    summary = {
        name: {
            "true_profit_per_hcp": round(float(true_profit[np.arange(rows), choice].mean()), 3),
            "ips_profit_per_hcp": round(ips_value(choice), 3) if name != "oracle" else None,
        }
        for name, choice in policies.items()
    }
    best_single = max((k for k in summary if k.startswith("all_")), key=lambda k: summary[k]["true_profit_per_hcp"])
    mix = pd.Series(np.array(ARMS)[model_choice]).value_counts(normalize=True).round(3).to_dict()
    return {
        "policies": summary,
        "best_single_channel": best_single,
        "model_action_mix": mix,
        "agreement_with_oracle": round(float((model_choice == oracle_choice).mean()), 3),
        "share_of_oracle_profit": round(
            summary["uplift_model_nba"]["true_profit_per_hcp"] / summary["oracle"]["true_profit_per_hcp"], 3
        ),
    }
