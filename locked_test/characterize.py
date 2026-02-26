#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax",
#   "jaxlib",
#   "numpy",
#   "matplotlib",
#   "requests",
#   "pandas",
#   "pystarboard @ git+https://github.com/CELtd/pystarboard.git",
#   "mechafil-jax @ file:///tmp/mechafil-jax-fix",
# ]
# ///
"""
Characterization test for locked FIL forecast sensitivity.

Varies two independent factors:
  1. Simulation start date (2022-01-01, ~6 months ago, ~1 month ago)
  2. Initialization mode (50/50 split vs data-driven pledge/reward estimate)

All runs are plotted against on-chain actual locked FIL from Spacescope.

This uses the fix/pledge-reward-split branch (via worktree at /tmp/mechafil-jax-fix)
which supports both modes:
  - data-driven: locked_reward_zero computed from protocol recurrence
  - 50/50: fallback when locked_reward_zero is removed from data dict

Run with:  uv run characterize.py
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import time

import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mechafil_jax.data as mdata
import mechafil_jax.sim as sim
from pystarboard.data_spacescope import SpacescopeDataConnection

# -- CONFIG -------------------------------------------------------------------
AUTH_TOKEN      = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
CURRENT_DATE    = date.today() - timedelta(days=3)
FORECAST_LENGTH = 365 * 3   # 3 years forecast
SECTOR_DURATION = 360
LOCK_TARGET     = 0.3
LOOKBACK_DAYS   = 30
# -----------------------------------------------------------------------------

START_DATES = {
    "2022-01-01":  date(2022, 1, 1),
    "~6mo ago":    CURRENT_DATE - timedelta(days=180),
    "~1mo ago":    CURRENT_DATE - timedelta(days=30),
}

print(f"Current date: {CURRENT_DATE}")
print(f"Forecast length: {FORECAST_LENGTH} days")
print(f"Start dates: {', '.join(f'{k}={v}' for k, v in START_DATES.items())}")

# -- Run simulations ---------------------------------------------------------
results = {}
for label, start_date in START_DATES.items():
    end_date = CURRENT_DATE + timedelta(days=FORECAST_LENGTH)

    print(f"\n{'='*60}")
    print(f"  {label}  (start={start_date})")
    print(f"  history {start_date} -> {CURRENT_DATE}  |  forecast -> {end_date}")
    print(f"{'='*60}")

    t0 = time.time()
    print("  Fetching simulation data ...")
    sim_data = mdata.get_simulation_data(AUTH_TOKEN, start_date, CURRENT_DATE, end_date)
    print(f"  done ({time.time()-t0:.1f}s)")

    # Derive scenario params from tail of history
    n = min(LOOKBACK_DAYS, len(sim_data["historical_onboarded_rb_power_pib"]))
    rb_pib  = sim_data["historical_onboarded_rb_power_pib"][-n:]
    qa_pib  = sim_data["historical_onboarded_qa_power_pib"][-n:]
    rr_hist = sim_data["historical_renewal_rate"][-n:]

    rbp_eib = float(np.median(rb_pib) / 1024.0)
    rr_val  = float(np.median(rr_hist))
    ratio   = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
    fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

    print(f"  Scenario (30d median): RBP={rbp_eib:.4f} EiB/d, RR={rr_val:.4f}, FPR={fpr_val:.4f}")

    rbp_vec = jnp.ones(FORECAST_LENGTH) * rbp_eib
    rr_vec  = jnp.ones(FORECAST_LENGTH) * rr_val
    fpr_vec = jnp.ones(FORECAST_LENGTH) * fpr_val

    # Report the pledge/reward split estimate
    locked_total      = sim_data["locked_fil_zero"]
    locked_reward_est = sim_data.get("locked_reward_zero", None)
    if locked_reward_est is not None:
        pct = locked_reward_est / locked_total * 100
        print(f"  locked_fil_zero       = {locked_total/1e6:.2f}M FIL")
        print(f"  est locked_reward     = {locked_reward_est/1e6:.2f}M FIL  ({pct:.1f}%)")
        print(f"  implied locked_pledge = {(locked_total - locked_reward_est)/1e6:.2f}M FIL  ({100-pct:.1f}%)")
        print(f"  (vs 50/50 assumption  = {locked_total/2/1e6:.2f}M each)")
    else:
        print(f"  locked_fil_zero = {locked_total/1e6:.2f}M FIL  (no data-driven estimate)")

    # Run 1: data-driven initialization
    t0 = time.time()
    print("  Running [data-driven] ...")
    r_dd = sim.run_sim(
        rbp_vec, rr_vec, fpr_vec,
        LOCK_TARGET, start_date, CURRENT_DATE, FORECAST_LENGTH, SECTOR_DURATION,
        sim_data,
    )
    print(f"    done ({time.time()-t0:.1f}s)")

    # Run 2: 50/50 initialization (strip locked_reward_zero so it falls back)
    sim_data_5050 = {k: v for k, v in sim_data.items() if k != "locked_reward_zero"}
    t0 = time.time()
    print("  Running [50/50] ...")
    r_5050 = sim.run_sim(
        rbp_vec, rr_vec, fpr_vec,
        LOCK_TARGET, start_date, CURRENT_DATE, FORECAST_LENGTH, SECTOR_DURATION,
        sim_data_5050,
    )
    print(f"    done ({time.time()-t0:.1f}s)")

    results[label] = {
        "start_date": start_date,
        "data_driven": r_dd,
        "50/50": r_5050,
        "locked_reward_est": locked_reward_est,
        "locked_fil_zero": locked_total,
    }

# -- Fetch on-chain actuals ---------------------------------------------------
print("\nFetching on-chain supply data ...")
earliest_start = min(v["start_date"] for v in results.values())
supply_df = SpacescopeDataConnection.query_spacescope_supply_stats(earliest_start, CURRENT_DATE)
supply_df = supply_df.sort_values("date")
onchain_dates  = pd.to_datetime(supply_df["date"]).dt.date.tolist()
onchain_locked = supply_df["locked_fil"].astype(float).values
print(f"  got {len(onchain_dates)} days")

onchain_df = pd.DataFrame({"date": onchain_dates, "onchain": onchain_locked}).set_index("date")

# -- Summary table ------------------------------------------------------------
print(f"\n{'='*80}")
print("SUMMARY: forecast value at current_date vs on-chain")
print(f"{'='*80}")

# On-chain value at current_date
oc_val = onchain_df.loc[:CURRENT_DATE].iloc[-1]["onchain"]
print(f"  On-chain locked at {CURRENT_DATE}: {oc_val/1e6:.2f}M FIL\n")

print(f"  {'Config':<35s} {'Sim (M)':<12s} {'Err (M)':<12s} {'Err %':<10s} {'1yr trend'}")
print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

for label, data in results.items():
    split = (CURRENT_DATE - data["start_date"]).days
    for mode in ["50/50", "data_driven"]:
        y = np.array(data[mode]["network_locked"])
        sim_val = float(y[split])
        err = sim_val - oc_val
        err_pct = err / oc_val * 100

        # 1-year trend: value at split+365 vs split
        if split + 365 < len(y):
            trend = float(y[split + 365]) - float(y[split])
            trend_str = f"{trend/1e6:+.1f}M ({'up' if trend > 0 else 'DOWN'})"
        else:
            trend_str = "n/a"

        tag = f"{label} [{mode}]"
        print(f"  {tag:<35s} {sim_val/1e6:<12.2f} {err/1e6:<+12.2f} {err_pct:<+10.2f} {trend_str}")

# -- Plot ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(
    f"Locked FIL Characterization  |  forecast from {CURRENT_DATE}\n"
    f"3 start dates  x  2 init modes  vs  on-chain actual",
    fontsize=12, fontweight="bold",
)

ax_main = axes[0]
ax_err  = axes[1]

# On-chain actual
ax_main.plot(onchain_dates, onchain_locked / 1e6,
             "k-", linewidth=2.5, label="On-chain actual", zorder=10)

COLORS = {
    "2022-01-01": "#1f77b4",   # blue
    "~6mo ago":   "#ff7f0e",   # orange
    "~1mo ago":   "#2ca02c",   # green
}
MODES = [("50/50", "--", 0.55), ("data_driven", "-", 1.0)]

for label, data in results.items():
    start  = data["start_date"]
    end    = CURRENT_DATE + timedelta(days=FORECAST_LENGTH)
    n_days = (end - start).days
    t      = [start + timedelta(days=i) for i in range(n_days)]
    split  = (CURRENT_DATE - start).days
    color  = COLORS[label]

    for mode, ls, alpha in MODES:
        y   = np.array(data[mode]["network_locked"]) / 1e6
        tag = f"{label} [{mode}]"

        # Historical portion (thin, faded)
        if split > 1:
            ax_main.plot(t[:split+1], y[:split+1],
                         color=color, linestyle=ls, alpha=alpha*0.35, linewidth=1)
        # Forecast portion
        ax_main.plot(t[split:], y[split:],
                     color=color, linestyle=ls, alpha=alpha, linewidth=2, label=tag)

ax_main.axvline(CURRENT_DATE, color="dimgray", linestyle=":", alpha=0.7)
ax_main.text(CURRENT_DATE, 0.97, f"  {CURRENT_DATE}",
             transform=ax_main.get_xaxis_transform(), fontsize=8, color="dimgray", va="top")
ax_main.set_ylabel("Network Locked FIL (M)")
ax_main.set_title("Locked FIL Trajectory")
ax_main.legend(fontsize=7, loc="upper right", ncol=2)
ax_main.grid(True, alpha=0.25)
ax_main.set_ylim(bottom=0)

# -- Error subplot: sim - on-chain during historical window -------------------
ax_err.set_title("Historical Error vs On-chain  (simulation - actual)")
ax_err.set_ylabel("Error (M FIL)")
ax_err.axhline(0, color="black", linewidth=0.5)

for label, data in results.items():
    start  = data["start_date"]
    n_hist = (CURRENT_DATE - start).days + 1
    color  = COLORS[label]

    for mode, ls, alpha in MODES:
        y_sim     = np.array(data[mode]["network_locked"])[:n_hist]
        sim_dates = [start + timedelta(days=i) for i in range(n_hist)]
        sim_df    = pd.DataFrame({"date": sim_dates, "sim": y_sim}).set_index("date")
        merged    = sim_df.join(onchain_df, how="inner")

        if len(merged) > 0:
            errors = (merged["sim"] - merged["onchain"]) / 1e6
            ax_err.plot(merged.index, errors.values,
                        color=color, linestyle=ls, alpha=alpha, linewidth=1.5,
                        label=f"{label} [{mode}]")

ax_err.legend(fontsize=7, loc="best", ncol=2)
ax_err.grid(True, alpha=0.25)
ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()

out_path = "characterization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()
