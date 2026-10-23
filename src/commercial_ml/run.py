"""Runs every analysis and writes reports/results.json, reports/REPORT.md and figures.

Usage: python -m commercial_ml.run [--seed 7] [--hcps 3000] [--out reports]"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from commercial_ml import models  # noqa: E402
from commercial_ml.data import ARMS, generate  # noqa: E402


def _figures(ds, seg: dict, nba: dict, out: Path) -> None:
    weekly = ds.weekly.groupby("week")["rx"].sum()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(weekly.index, weekly.values, lw=1.4)
    ax.axvspan(120, 131, alpha=0.15, label="email campaign window")
    ax.axvspan(140, 151, alpha=0.15, color="orange", label="randomized channel test")
    ax.set(xlabel="week", ylabel="brand Rx", title="Weekly brand prescriptions")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "weekly_rx.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    k = list(seg["silhouette_by_k"])
    ax.plot(k, list(seg["silhouette_by_k"].values()), marker="o")
    ax.set(xlabel="k", ylabel="silhouette", title="Choosing the number of segments")
    fig.tight_layout()
    fig.savefig(out / "silhouette.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(nba["policies"])
    ax.barh(names, [nba["policies"][n]["true_profit_per_hcp"] for n in names])
    ax.set(xlabel="true incremental profit per HCP (Rx-equivalents)", title="Next-best-action policies, held-out HCPs")
    fig.tight_layout()
    fig.savefig(out / "nba_policies.png", dpi=120)
    plt.close(fig)


def _report(r: dict) -> str:
    seg, fc, tg, cp, nba = r["segmentation"], r["forecasting"], r["targeting"], r["campaign"], r["next_best_action"]
    lines = [
        "# Results",
        "",
        f"Seed {r['seed']}, {r['hcps']} HCPs x 156 weeks; run took {r['seconds']} s.",
        "",
        "![weekly](weekly_rx.png)",
        "",
        "## Segmentation (k-means, k by silhouette)",
        "",
        f"k = {seg['k']}; adjusted Rand index against the planted archetypes = {seg['ari_vs_true_archetypes']}.",
        "",
        seg["profile_markdown"],
        "",
        "## Forecasting (weekly Rx by region, rolling origin, horizons 1-4)",
        "",
        "| model | WAPE | MAPE |",
        "|---|---|---|",
        *[f"| {m} | {v['wape']:.2%} | {v['mape']:.2%} |" for m, v in fc.items()],
        "",
        "## Targeting (who engages with email next quarter)",
        "",
        f"{tg['eligible_hcps']} eligible HCPs, {tg['engaged_share']:.1%} engaged.",
        "",
        "| scorer | AUC | lift in top 20% |",
        "|---|---|---|",
        *[f"| {m} | {v['auc']:.3f} | {v['lift_top20']:.2f} |" for m, v in tg.items() if isinstance(v, dict)],
        "",
        "## Campaign effectiveness (Rx per targeted HCP, weeks 120-131)",
        "",
        "| estimator | estimate |",
        "|---|---|",
        f"| planted true ATT | {cp['true_att']} |",
        f"| naive post-period difference | {cp['naive_post_difference']} |",
        f"| difference-in-differences | {cp['difference_in_differences']} |",
        f"| IPW difference-in-differences | {cp['ipw_did']} (95% bootstrap CI {cp['ipw_did_95ci'][0]} to {cp['ipw_did_95ci'][1]}) |",
        "",
        "## Next best action (randomized test, held-out half)",
        "",
        "| policy | true profit / HCP | IPS estimate / HCP |",
        "|---|---|---|",
        *[
            f"| {n} | {v['true_profit_per_hcp']} | {v['ips_profit_per_hcp'] if v['ips_profit_per_hcp'] is not None else '-'} |"
            for n, v in nba["policies"].items()
        ],
        "",
        f"Uplift policy captures {nba['share_of_oracle_profit']:.0%} of oracle profit; best single channel is "
        f"`{nba['best_single_channel']}`; recommended mix {nba['model_action_mix']}.",
        "",
        "![nba](nba_policies.png)",
    ]
    return "\n".join(lines) + "\n"


def main(seed: int = 7, n_hcp: int = 3000, out: str = "reports") -> dict:
    start = time.perf_counter()
    ds = generate(n_hcp=n_hcp, seed=seed)
    seg = models.segment(ds)
    results = {
        "seed": seed,
        "hcps": n_hcp,
        "segmentation": {k: v for k, v in seg.items() if k not in ("profile", "assignments")},
        "forecasting": models.forecast_backtest(ds),
        "targeting": models.targeting(ds),
        "campaign": models.campaign_effect(ds),
        "next_best_action": models.next_best_action(ds),
        "arms": list(ARMS),
    }
    results["seconds"] = round(time.perf_counter() - start, 1)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    _figures(ds, seg, results["next_best_action"], out_dir)
    results["segmentation"]["profile_markdown"] = (
        seg["profile"].to_markdown() if _has_tabulate() else seg["profile"].to_string()
    )
    (out_dir / "REPORT.md").write_text(_report(results), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hcps", type=int, default=3000)
    p.add_argument("--out", default="reports")
    a = p.parse_args()
    r = main(a.seed, a.hcps, a.out)
    print(
        json.dumps(
            {k: v for k, v in r.items() if k != "segmentation"} | {"segmentation_k": r["segmentation"]["k"]}, indent=1
        )
    )
