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
Rolling backtest: start a forecast every quarter, measure error against
on-chain actuals at 0/3/6/12-month horizons, for 4 init methods.

Part 1  – Estimation comparison (fast, no per-date API calls)
           Shows how each method estimates locked_reward over time.

Part 2  – Forecast backtest (quarterly sims)
           Shows end-to-end forecast error vs on-chain.
"""

import sys, time
sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pystarboard.data as pdata
from pystarboard.data_spacescope import SpacescopeDataConnection
import mechafil_jax.data as mdata
import mechafil_jax.sim as sim

AUTH_TOKEN    = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
MAINNET_LAUNCH = date(2020, 10, 15)
TODAY         = date.today() - timedelta(days=3)

# ── Estimation helpers ──────────────────────────────────────────────

def est_balance180(daily_rewards, lookback=200):
    """mechafil's approximation: release = balance/180 (exponential decay)."""
    n = min(len(daily_rewards), lookback)
    locked = 0.0
    for dr in daily_rewards[-n:]:
        locked += 0.75 * dr - locked / 180.0
    return max(locked, 0.0)

def est_balance180_full(daily_rewards):
    """Same recurrence but seeded from mainnet launch (no truncation)."""
    return est_balance180(daily_rewards, lookback=len(daily_rewards))

def est_linear_vesting(daily_rewards):
    """True protocol: each day's 75% reward vests linearly over 180 days."""
    n = min(len(daily_rewards), 180)
    recent = daily_rewards[-n:]          # oldest first
    locked = 0.0
    for i, dr in enumerate(recent):
        days_ago = n - 1 - i             # 0 = today, n-1 = oldest
        remaining = (180 - days_ago) / 180.0
        locked += 0.75 * dr * remaining
    return max(locked, 0.0)


# ═══════════════════════════════════════════════════════════════════
#  PART 1 – estimation comparison (one API call, monthly resolution)
# ═══════════════════════════════════════════════════════════════════
print("PART 1: Estimation comparison")
print("=" * 60)

print("Fetching full supply-stats history …")
t0 = time.time()
pdata.setup_spacescope(AUTH_TOKEN)
full_df = SpacescopeDataConnection.query_spacescope_supply_stats(
    MAINNET_LAUNCH, TODAY,
)
full_df = full_df.sort_values("date").reset_index(drop=True)
full_dates  = np.array([date.fromisoformat(str(d)) for d in full_df["date"]])
full_mined  = full_df["mined_fil"].astype(float).values
full_locked = full_df["locked_fil"].astype(float).values
daily_rewards = np.diff(full_mined)     # length = len(full_dates) - 1
reward_dates  = full_dates[1:]          # each entry = day the reward was earned
print(f"  {len(reward_dates)} daily rewards  ({reward_dates[0]} → {reward_dates[-1]})  [{time.time()-t0:.1f}s]")

# evaluate at the 1st of every month from 2021-07 onward
rows = []
d = date(2021, 7, 1)
while d <= TODAY:
    idx = int(np.searchsorted(reward_dates, d))
    if idx < 30:
        pass
    else:
        rw = daily_rewards[:idx]
        li = min(int(np.searchsorted(full_dates, d)), len(full_locked) - 1)
        total_locked = full_locked[li]
        rows.append(dict(
            date        = d,
            total_locked = total_locked,
            est_5050     = total_locked / 2.0,
            est_b180_200 = est_balance180(rw, 200),
            est_b180_full= est_balance180_full(rw),
            est_linear   = est_linear_vesting(rw),
        ))
    # next month
    d = date(d.year + d.month // 12, d.month % 12 + 1, 1)

p1 = pd.DataFrame(rows)

print(f"\n  {'Date':<12} {'Locked(M)':<11} {'50/50':<11} {'B180-200':<11} {'B180-full':<11} {'Linear':<11}")
for _, r in p1.iloc[::6].iterrows():               # print every 6 months
    print(f"  {r.date!s:<12} {r.total_locked/1e6:<11.2f} "
          f"{r.est_5050/1e6:<11.2f} {r.est_b180_200/1e6:<11.2f} "
          f"{r.est_b180_full/1e6:<11.2f} {r.est_linear/1e6:<11.2f}")

# ── Part 1 plots ────────────────────────────────────────────────────
fig1, (ax_est, ax_pledge) = plt.subplots(2, 1, figsize=(14, 9),
                                          gridspec_kw={"height_ratios": [1, 1]})
fig1.suptitle("Part 1: locked_reward Estimation Methods", fontsize=13, fontweight="bold")

dates = p1["date"]
ax_est.plot(dates, p1.total_locked/1e6, "k-", lw=2, label="Total locked (on-chain)")
ax_est.plot(dates, p1.est_5050/1e6, "--", color="red", lw=1.5, label="50/50")
ax_est.plot(dates, p1.est_b180_200/1e6, "-", color="blue", lw=1.5, label="Balance/180  200d lookback")
ax_est.plot(dates, p1.est_b180_full/1e6, "-", color="purple", lw=1.5, label="Balance/180  full history")
ax_est.plot(dates, p1.est_linear/1e6, "-", color="green", lw=1.5, label="True linear vesting 180d")
ax_est.set_ylabel("M FIL"); ax_est.set_title("Estimated reward-locked FIL")
ax_est.legend(fontsize=8); ax_est.grid(True, alpha=0.25); ax_est.set_ylim(bottom=0)

# implied pledge = total - reward
ax_pledge.plot(dates, p1.total_locked/1e6, "k-", lw=2, label="Total locked")
for col, lbl, c, ls in [
    ("est_5050",      "50/50",           "red",    "--"),
    ("est_b180_200",  "B180-200d",       "blue",   "-"),
    ("est_b180_full", "B180-full",       "purple", "-"),
    ("est_linear",    "Linear",          "green",  "-"),
]:
    ax_pledge.plot(dates, (p1.total_locked - p1[col])/1e6, ls, color=c, lw=1.5,
                   label=f"Implied pledge ({lbl})")
ax_pledge.set_ylabel("M FIL"); ax_pledge.set_title("Implied pledge-locked FIL  (total − reward estimate)")
ax_pledge.legend(fontsize=8); ax_pledge.grid(True, alpha=0.25)
plt.tight_layout()
fig1.savefig("part1_estimation.png", dpi=150, bbox_inches="tight")
print("\nPart 1 saved → part1_estimation.png")


# ═══════════════════════════════════════════════════════════════════
#  PART 2 – forecast backtest (quarterly start dates)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PART 2: Forecast Backtest (quarterly)")
print("=" * 60)

FORECAST_DAYS   = 365
SECTOR_DURATION = 360
LOCK_TARGET     = 0.3
LOOKBACK        = 30
HISTORY_DAYS    = 90        # days of sim history before forecast start

# quarterly dates with ≥ 90 days of forward actuals
bt_dates = []
for y in [2023, 2024, 2025]:
    for m in [1, 4, 7, 10]:
        d = date(y, m, 1)
        if d < TODAY - timedelta(days=90):
            bt_dates.append(d)

METHODS = ["50/50", "B180-200d", "B180-full", "Linear-180d"]
M_COLORS = {"50/50": "red", "B180-200d": "blue", "B180-full": "purple", "Linear-180d": "green"}

print(f"Backtest dates: {', '.join(str(d) for d in bt_dates)}")
bt_rows = []

for bt in bt_dates:
    start = bt - timedelta(days=HISTORY_DAYS)
    end   = bt + timedelta(days=FORECAST_DAYS)

    print(f"\n  {bt} …", end="")
    t0 = time.time()
    try:
        sd = mdata.get_simulation_data(AUTH_TOKEN, start, bt, end)
    except Exception as e:
        print(f"  FAILED ({e})")
        continue
    print(f"  data {time.time()-t0:.0f}s", end="")

    # scenario params from last 30 days
    n = min(LOOKBACK, len(sd["historical_onboarded_rb_power_pib"]))
    rb  = sd["historical_onboarded_rb_power_pib"][-n:]
    qa  = sd["historical_onboarded_qa_power_pib"][-n:]
    rrh = sd["historical_renewal_rate"][-n:]
    rbp_eib = float(np.median(rb) / 1024.0)
    rr_val  = float(np.median(rrh))
    ratio   = np.where(rb > 0, qa / rb, 1.0)
    fpr_val = float(np.median(np.clip((ratio - 1) / 9, 0, 1)))

    rbp_v = jnp.ones(FORECAST_DAYS) * rbp_eib
    rr_v  = jnp.ones(FORECAST_DAYS) * rr_val
    fpr_v = jnp.ones(FORECAST_DAYS) * fpr_val

    locked_total = sd["locked_fil_zero"]
    idx = int(np.searchsorted(reward_dates, bt))
    rw  = daily_rewards[:idx]

    estimates = {
        "50/50":       locked_total / 2.0,
        "B180-200d":   est_balance180(rw, 200),
        "B180-full":   est_balance180_full(rw),
        "Linear-180d": est_linear_vesting(rw),
    }
    split = (bt - start).days          # index where forecast starts

    for method, lr in estimates.items():
        sd2 = dict(sd)
        if method == "50/50":
            sd2.pop("locked_reward_zero", None)
        else:
            sd2["locked_reward_zero"] = lr

        r = sim.run_sim(rbp_v, rr_v, fpr_v, LOCK_TARGET,
                        start, bt, FORECAST_DAYS, SECTOR_DURATION, sd2)
        y = np.array(r["network_locked"])

        for h_mo, h_days in [(0, 0), (3, 90), (6, 180), (12, 365)]:
            target = bt + timedelta(days=h_days)
            if target > TODAY:
                continue
            oc_idx = int(np.searchsorted(full_dates, target))
            if oc_idx >= len(full_locked):
                continue
            sim_idx = split + h_days
            if sim_idx >= len(y):
                continue
            oc_v  = full_locked[oc_idx]
            sim_v = float(y[sim_idx])
            bt_rows.append(dict(
                forecast_start=bt, method=method, horizon_mo=h_mo,
                sim_val=sim_v, onchain_val=oc_v,
                err_pct=(sim_v - oc_v) / oc_v * 100,
            ))

    print(f"  total {time.time()-t0:.0f}s")

bt_df = pd.DataFrame(bt_rows)

# ── Part 2 summary table ───────────────────────────────────────────
for h in [0, 3, 6, 12]:
    hdf = bt_df[bt_df.horizon_mo == h]
    if hdf.empty:
        continue
    print(f"\n  ── {h}-month horizon ──")
    print(f"  {'Start':<12}", end="")
    for m in METHODS:
        print(f" {m:>13}", end="")
    print()
    for bt in bt_dates:
        sub = hdf[hdf.forecast_start == bt]
        if sub.empty:
            continue
        print(f"  {bt!s:<12}", end="")
        for m in METHODS:
            row = sub[sub.method == m]
            if not row.empty:
                print(f" {row.iloc[0].err_pct:>+11.1f}%", end="")
            else:
                print(f" {'n/a':>13}", end="")
        print()
    # mean absolute error
    print(f"  {'MAE':<12}", end="")
    for m in METHODS:
        vals = hdf[hdf.method == m].err_pct.abs()
        print(f" {vals.mean():>11.1f}%", end="") if len(vals) else print(f" {'':>13}", end="")
    print()

# ── Part 2 plot ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
fig2.suptitle("Part 2: Forecast Error vs On-chain  (% at each horizon)",
              fontsize=13, fontweight="bold")

for i, h in enumerate([0, 3, 6, 12]):
    ax = axes2[i // 2][i % 2]
    hdf = bt_df[bt_df.horizon_mo == h]
    for m in METHODS:
        sub = hdf[hdf.method == m].sort_values("forecast_start")
        if not sub.empty:
            ax.plot(sub.forecast_start, sub.err_pct, "o-",
                    color=M_COLORS[m], label=m, markersize=5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{h}-month horizon")
    ax.set_ylabel("Error (%)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
fig2.savefig("part2_backtest.png", dpi=150, bbox_inches="tight")
print(f"\nPart 2 saved → part2_backtest.png")
print("\nDone.")
