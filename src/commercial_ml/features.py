"""HCP-level features computed in SQL (DuckDB) as of a cut-off week, the way they would be pulled
from a warehouse. Window functions give recency, trailing volumes, growth and channel response."""

from __future__ import annotations

import duckdb
import pandas as pd

FEATURE_SQL = """
WITH w AS (
    SELECT * FROM weekly WHERE week < $as_of
),
recent AS (
    SELECT
        hcp_id,
        SUM(rx) FILTER (WHERE week >= $as_of - 26)                        AS rx_26w,
        SUM(rx) FILTER (WHERE week >= $as_of - 13)                        AS rx_13w,
        SUM(rx) FILTER (WHERE week >= $as_of - 26 AND week < $as_of - 13) AS rx_prev_13w,
        SUM(rx) FILTER (WHERE week >= $as_of - 52)                        AS rx_52w,
        SUM(rep_visits) FILTER (WHERE week >= $as_of - 26)                AS rep_visits_26w,
        SUM(emails) FILTER (WHERE week >= $as_of - 26)                    AS emails_26w,
        SUM(email_opens) FILTER (WHERE week >= $as_of - 26)               AS email_opens_26w,
        SUM(webinars) FILTER (WHERE week >= $as_of - 52)                  AS webinars_52w,
        SUM(webinar_attended) FILTER (WHERE week >= $as_of - 52)          AS webinars_attended_52w,
        STDDEV_SAMP(rx) FILTER (WHERE week >= $as_of - 26)                AS rx_volatility_26w,
        $as_of - MAX(week) FILTER (WHERE rep_visits > 0)                  AS weeks_since_rep_visit
    FROM w
    GROUP BY hcp_id
),
trend AS (
    -- Slope of a 4-week moving average over the last 26 weeks, via REGR_SLOPE.
    SELECT hcp_id, REGR_SLOPE(ma4, week) AS rx_trend_slope
    FROM (
        SELECT hcp_id, week,
               AVG(rx) OVER (PARTITION BY hcp_id ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS ma4
        FROM w
        WHERE week >= $as_of - 26
    )
    GROUP BY hcp_id
)
SELECT
    h.hcp_id, h.specialty, h.region, h.years_in_practice, h.patient_volume,
    r.rx_26w, r.rx_13w, r.rx_52w, r.rep_visits_26w, r.emails_26w, r.webinars_52w,
    r.rx_13w / NULLIF(r.rx_prev_13w, 0) - 1                 AS rx_growth_13w,
    r.rx_26w / NULLIF(h.patient_volume * 26, 0)            AS rx_per_patient,
    r.email_opens_26w / NULLIF(r.emails_26w, 0)            AS email_open_rate,
    r.webinars_attended_52w / NULLIF(r.webinars_52w, 0)    AS webinar_attend_rate,
    r.rx_volatility_26w,
    COALESCE(r.weeks_since_rep_visit, 999)                 AS weeks_since_rep_visit,
    t.rx_trend_slope
FROM hcps h
JOIN recent r USING (hcp_id)
JOIN trend t USING (hcp_id)
ORDER BY h.hcp_id
"""


def hcp_features(hcps: pd.DataFrame, weekly: pd.DataFrame, as_of: int) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("hcps", hcps[[c for c in hcps.columns if not c.startswith("true_")]])
    con.register("weekly", weekly)
    df = con.execute(FEATURE_SQL, {"as_of": as_of}).df()
    con.close()
    # Rates are undefined with no sends; the neutral fill is the population median.
    for col in ("email_open_rate", "webinar_attend_rate", "rx_growth_13w"):
        df[col] = df[col].fillna(df[col].median())
    return df


NUMERIC_FEATURES = [
    "years_in_practice",
    "patient_volume",
    "rx_26w",
    "rx_13w",
    "rx_52w",
    "rep_visits_26w",
    "emails_26w",
    "webinars_52w",
    "rx_growth_13w",
    "rx_per_patient",
    "email_open_rate",
    "webinar_attend_rate",
    "rx_volatility_26w",
    "weeks_since_rep_visit",
    "rx_trend_slope",
]
CATEGORICAL_FEATURES = ["specialty", "region"]


def model_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Numeric features plus one-hot categoricals, ready for scikit-learn."""
    return pd.get_dummies(
        features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], columns=CATEGORICAL_FEATURES, dtype=float
    ).fillna(0.0)
