"""Each analysis is checked against the ground truth planted in the synthetic data, on a smaller
dataset so the suite stays fast."""

import duckdb
import numpy as np
import pytest

from commercial_ml import models
from commercial_ml.data import ARMS, TEST_OUTCOME_WEEKS, generate
from commercial_ml.features import hcp_features


@pytest.fixture(scope="module")
def ds():
    return generate(n_hcp=2000, seed=11)


def test_generator_is_deterministic_and_has_ground_truth():
    a, b = generate(n_hcp=200, seed=3), generate(n_hcp=200, seed=3)
    assert a.weekly.equals(b.weekly)
    assert a.truth["campaign_att_rx_per_hcp"] > 0
    assert list(a.truth["uplift_rx"].columns) == list(ARMS)
    assert (a.truth["uplift_rx"]["none"] == 0).all()


def test_regular_promotion_pauses_during_the_randomized_test(ds):
    test = ds.weekly[ds.weekly.week.isin(TEST_OUTCOME_WEEKS)].merge(ds.hcps[["hcp_id", "test_arm"]])
    assert test.loc[test.test_arm == "none", ["rep_visits", "emails", "webinars"]].sum().sum() == 0
    assert test.loc[test.test_arm == "rep", "rep_visits"].groupby(test.hcp_id).sum().eq(4).all()


def test_sql_features_match_pandas(ds):
    f = hcp_features(ds.hcps, ds.weekly, as_of=100).set_index("hcp_id")
    w = ds.weekly[ds.weekly.week < 100]
    rx26 = w[w.week >= 74].groupby("hcp_id")["rx"].sum()
    assert (f["rx_26w"] == rx26).all()
    reps = w[w.week >= 74].groupby("hcp_id")["rep_visits"].sum()
    assert (f["rep_visits_26w"] == reps).all()
    assert not f.filter(like="true_").columns.any(), "hidden truth must never reach the features"
    assert duckdb.__version__


def test_segmentation_recovers_archetype_structure(ds):
    r = models.segment(ds, k_range=range(2, 7))
    assert 3 <= r["k"] <= 6
    assert r["ari_vs_true_archetypes"] > 0.2


def test_holt_winters_beats_seasonal_naive(ds):
    r = models.forecast_backtest(ds, first_origin=112, last_origin=128)
    assert r["holt_winters"]["wape"] < r["seasonal_naive"]["wape"]
    assert all(v["wape"] < 0.06 for v in r.values())


def test_targeting_model_beats_past_open_rate_rule(ds):
    r = models.targeting(ds)
    assert r["logistic_regression"]["auc"] > r["rule_past_open_rate"]["auc"]
    assert r["logistic_regression"]["lift_top20"] > 1.2


def test_campaign_estimators_correct_the_confounding(ds):
    r = models.campaign_effect(ds)
    truth = r["true_att"]
    naive_error = abs(r["naive_post_difference"] - truth)
    assert abs(r["difference_in_differences"] - truth) < naive_error
    assert abs(r["ipw_did"] - truth) < naive_error
    assert r["ipw_did_95ci"][0] <= truth <= r["ipw_did_95ci"][1]


def test_uplift_policy_beats_every_single_channel_policy(ds):
    r = models.next_best_action(ds)
    best_single = r["policies"][r["best_single_channel"]]["true_profit_per_hcp"]
    assert r["policies"]["uplift_model_nba"]["true_profit_per_hcp"] > best_single
    assert 0 < r["share_of_oracle_profit"] <= 1
    assert np.isclose(sum(r["model_action_mix"].values()), 1, atol=0.01)
