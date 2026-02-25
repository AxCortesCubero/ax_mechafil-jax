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
#   "mechafil-jax @ file:///home/kiran/code/cel/mechafil-jax",
# ]
# ///
"""
Multi-scenario Filecoin 1-year forecast.

Scenarios are defined as dicts and overlaid on the same 4x3 panel.
Easy to add new scenarios by appending to the SCENARIOS list.

Produces:
  1. forecast_panel.png  – 4x3 panel with all scenarios overlaid
  2. bugfix_comparison.png – before/after comparison of the 50/50 fix
"""

import sys, time
sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mechafil_jax.data as mdata
import mechafil_jax.sim as sim
import mechafil_jax.minting as minting
import mechafil_jax.constants as C
from pystarboard.data_spacescope import SpacescopeDataConnection

# ── CONFIG ──────────────────────────────────────────────────────────
AUTH_TOKEN      = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
CURRENT_DATE    = date.today() - timedelta(days=3)
HISTORY_DAYS    = 365           # days of history (365 needed for FoFR 1Y rolling window)
FORECAST_DAYS   = 730           # 2-year forecast
SIM_EXTRA       = 365           # extra days so FoFR convolution extends through forecast
SECTOR_DURATION = 360
LOCK_TARGET     = 0.3
LOOKBACK        = 30            # days to median for input derivation
# ────────────────────────────────────────────────────────────────────

START_DATE = CURRENT_DATE - timedelta(days=HISTORY_DAYS)
END_DATE   = CURRENT_DATE + timedelta(days=FORECAST_DAYS)
SIM_END    = CURRENT_DATE + timedelta(days=FORECAST_DAYS + SIM_EXTRA)
SIM_FCST   = FORECAST_DAYS + SIM_EXTRA

print(f"History:  {START_DATE} -> {CURRENT_DATE}  ({HISTORY_DAYS} days)")
print(f"Forecast: {CURRENT_DATE} -> {END_DATE}  ({FORECAST_DAYS} days)")

# ── Fetch data ──────────────────────────────────────────────────────
print("\nFetching simulation data ...")
t0 = time.time()
sim_data = mdata.get_simulation_data(AUTH_TOKEN, START_DATE, CURRENT_DATE, SIM_END)
print(f"  done ({time.time()-t0:.1f}s)")

# ── Derive 30-day inputs ───────────────────────────────────────────
rb_pib  = sim_data["historical_onboarded_rb_power_pib"][-LOOKBACK:]
qa_pib  = sim_data["historical_onboarded_qa_power_pib"][-LOOKBACK:]
rr_hist = sim_data["historical_renewal_rate"][-LOOKBACK:]

rbp_pib_day = float(np.median(rb_pib))
# Use mean for RR: locked FIL is a cumulative stock, so the mean
# (not median) is the correct aggregation for total pledge retention.
rr_val      = float(np.mean(rr_hist))
ratio       = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val     = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

# NOTE: sim.run_sim docstring says rb_onboard_power is "in EiB" but
# forecast_power_stats adds it directly to rb_power_zero which is in PiB.
# The correct unit is PiB. The docstring is wrong.

print(f"\nInputs (30-day):")
print(f"  RB Onboarding : {rbp_pib_day:.2f} PiB/day  (median)")
print(f"  Renewal Rate  : {rr_val:.4f}  ({rr_val*100:.1f}%, mean)")
print(f"  FIL+ Rate     : {fpr_val:.4f}  ({fpr_val*100:.1f}%, median)")

# Report pledge/reward split
locked_total = sim_data["locked_fil_zero"]
locked_reward = sim_data.get("locked_reward_zero", locked_total / 2.0)
print(f"\nInitialization:")
print(f"  locked_fil_zero   = {locked_total/1e6:.2f}M FIL")
print(f"  locked_reward_est = {locked_reward/1e6:.2f}M FIL  ({locked_reward/locked_total*100:.1f}%)")
print(f"  locked_pledge_est = {(locked_total-locked_reward)/1e6:.2f}M FIL  ({(1-locked_reward/locked_total)*100:.1f}%)")

# ── Fit log-linear trend on historical RB onboarding ───────────────
hist_rbp_full = np.array(sim_data["historical_onboarded_rb_power_pib"])
# Use a smoothed version to avoid fitting noise; filter out zeros
valid = hist_rbp_full > 0
x_hist = np.arange(len(hist_rbp_full))
if valid.sum() > 10:
    log_rb = np.log(hist_rbp_full[valid])
    x_valid = x_hist[valid]
    # Fit: log(rb_pib) = a + b * day_index
    b, a = np.polyfit(x_valid, log_rb, 1)
    # Extrapolate into forecast
    x_fcst = np.arange(len(hist_rbp_full), len(hist_rbp_full) + SIM_FCST)
    rbp_trend_pib = np.exp(a + b * x_fcst)
    trend_start_pib = float(rbp_trend_pib[0])
    trend_end_pib = float(rbp_trend_pib[FORECAST_DAYS - 1])
    daily_pct = (np.exp(b) - 1) * 100
    print(f"\nLog-linear trend fit:")
    print(f"  Daily rate    : {daily_pct:+.3f}%/day  ({daily_pct*365:+.1f}%/yr)")
    print(f"  Forecast start: {trend_start_pib:.2f} PiB/day")
    print(f"  Forecast end  : {trend_end_pib:.2f} PiB/day")
else:
    print("\nWARNING: not enough valid RB data to fit trend")
    rbp_trend_pib = np.ones(SIM_FCST) * rbp_pib_day

# ── Common inputs (RR and FIL+ are static for all scenarios) ───────
rr_vec  = jnp.ones(SIM_FCST) * rr_val
fpr_vec = jnp.ones(SIM_FCST) * fpr_val

# ════════════════════════════════════════════════════════════════════
# SCENARIOS
# ════════════════════════════════════════════════════════════════════
# ── Decline scenario: RBP decays exponentially + RR declines linearly ──
# Combines declining new entrants with declining renewals
rbp_decay_rate = 3.0  # 3x reduction per year
rr_end_frac    = 0.50 # RR drops to 50% of current

days_fc = jnp.arange(SIM_FCST)
rbp_decline_pib = rbp_pib_day * (1.0 / rbp_decay_rate) ** (days_fc / 365.0)
rr_ramp = jnp.linspace(rr_val, rr_val * rr_end_frac, FORECAST_DAYS)
rr_decline = jnp.concatenate([rr_ramp, jnp.ones(SIM_EXTRA) * rr_ramp[-1]])

rbp_1y = float(rbp_decline_pib[364])
rbp_2y = float(rbp_decline_pib[min(729, SIM_FCST-1)])
print(f"\nDecline scenario:")
print(f"  RBP: {rbp_pib_day:.2f} -> {rbp_1y:.2f} (1Y) -> {rbp_2y:.2f} (2Y) PiB/d  ({rbp_decay_rate:.0f}x/yr exponential decay)")
print(f"  RR:  {rr_val*100:.0f}% -> {rr_val*rr_end_frac*100:.0f}% (linear over {FORECAST_DAYS}d)")

SCENARIOS = [
    {
        "name": "Status Quo",
        "color": "steelblue",
        "linestyle": "--",
        "rbp_pib": jnp.ones(SIM_FCST) * rbp_pib_day,
        "rr": jnp.ones(SIM_FCST) * rr_val,
    },
    {
        "name": "Decline",
        "color": "crimson",
        "linestyle": "--",
        "rbp_pib": rbp_decline_pib,
        "rr": rr_decline,
    },
]

# ── Run all scenarios ──────────────────────────────────────────────
for sc in SCENARIOS:
    print(f"\nRunning simulation [{sc['name']}] ...")
    t0 = time.time()
    sc["results"] = sim.run_sim(
        sc["rbp_pib"], sc["rr"], fpr_vec,
        LOCK_TARGET, START_DATE, CURRENT_DATE, SIM_FCST, SECTOR_DURATION,
        sim_data,
    )
    print(f"  done ({time.time()-t0:.1f}s)")

# ── Fetch on-chain actuals ─────────────────────────────────────────
print("Fetching on-chain data ...")
supply_df = SpacescopeDataConnection.query_spacescope_supply_stats(START_DATE, CURRENT_DATE)
supply_df = supply_df.sort_values("date")
oc_dates  = [date.fromisoformat(str(d)) for d in supply_df["date"]]
oc_locked = supply_df["locked_fil"].astype(float).values
oc_circ   = supply_df["circulating_fil"].astype(float).values
print(f"  {len(oc_dates)} days")

# ── Time axis ───────────────────────────────────────────────────────
total_days = (END_DATE - START_DATE).days
t = [START_DATE + timedelta(days=i) for i in range(total_days)]
split = HISTORY_DAYS
N = total_days

def d(arr):
    """Slice a result array to the display window."""
    a = np.array(arr)
    assert len(a) >= N, f"result array too short: {len(a)} < N={N}. Increase SIM_EXTRA."
    return a[:N]

# ── Build input vectors for plotting ────────────────────────────────
hist_rbp = np.array(sim_data["historical_onboarded_rb_power_pib"])
hist_rr  = np.array(sim_data["historical_renewal_rate"])
hist_qa  = np.array(sim_data["historical_onboarded_qa_power_pib"])
hist_ratio = np.where(hist_rbp > 0, hist_qa / hist_rbp, 1.0)
hist_fpr = np.clip((hist_ratio - 1.0) / 9.0, 0.0, 1.0)

# ── Shared plot helpers ─────────────────────────────────────────────
HIST_KW  = dict(color="dimgray", linewidth=1.2, linestyle="-", alpha=0.7)
OC_KW    = dict(color="black", linewidth=1.5, linestyle="-", alpha=0.7)
VLINE_KW = dict(color="dimgray", linewidth=0.8, linestyle=":", alpha=0.6)
INPUT_KW = dict(color="darkorange", linewidth=1.5, linestyle="-")

def add_vline(ax):
    ax.axvline(CURRENT_DATE, **VLINE_KW)

# ════════════════════════════════════════════════════════════════════
# FIGURE 1: Scenario Forecast Panel
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 3, figsize=(17, 14), sharex=True)
sc_labels = ", ".join(s["name"] for s in SCENARIOS)
fig.suptitle(
    f"Filecoin 2Y Forecast  |  {CURRENT_DATE}\n"
    f"RBP={rbp_pib_day:.2f} PiB/d,  FIL+={fpr_val*100:.1f}%  (30d median)  |  Scenarios: {sc_labels}",
    fontsize=12, fontweight="bold", y=0.98,
)

# Helper: plot historical (gray) + per-scenario forecast lines
def plot_scenarios(ax, key, scale=1.0):
    """Plot shared history and per-scenario forecast for a result key."""
    # All scenarios share the same history; use first
    y0 = d(SCENARIOS[0]["results"][key]) * scale
    ax.plot(t[:split+1], y0[:split+1], **HIST_KW, label="Historical")
    for sc in SCENARIOS:
        y = d(sc["results"][key]) * scale
        ax.plot(t[split:], y[split:], color=sc["color"], linewidth=2.0,
                linestyle=sc["linestyle"], label=sc["name"])

# ── Row 0: Inputs ──────────────────────────────────────────────────
ax = axes[0, 0]
t_inp = t[:len(hist_rbp)]
ax.plot(t_inp, hist_rbp, **INPUT_KW, label="Historical")
for sc in SCENARIOS:
    rbp_pib_fc = np.array(sc["rbp_pib"][:FORECAST_DAYS])
    ax.plot(t[split:], rbp_pib_fc, color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("PiB/day"); ax.set_title("RB Onboarding Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[0, 1]
t_rr = t[:len(hist_rr)]
ax.plot(t_rr, hist_rr * 100, **INPUT_KW, label="Historical")
for sc in SCENARIOS:
    rr_fc = np.array(sc["rr"][:FORECAST_DAYS]) * 100
    ax.plot(t[split:], rr_fc, color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("%"); ax.set_title("Renewal Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(0, 100)

ax = axes[0, 2]
t_fpr = t[:len(hist_fpr)]
ax.plot(t_fpr, hist_fpr * 100, **INPUT_KW, label="Historical")
ax.plot(t[split:], [fpr_val*100]*len(t[split:]), color="dimgray", linewidth=2.0,
        linestyle="--", label=f"Forecast ({fpr_val*100:.1f}%)")
add_vline(ax); ax.set_ylabel("%"); ax.set_title("FIL+ Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(0, 100)

# ── Row 1: Power + Minting + Pledge/Reward ─────────────────────────
ax = axes[1, 0]
# RBP and QAP on same plot (log scale)
y0_rb = d(SCENARIOS[0]["results"]["rb_total_power_eib"])
y0_qa = d(SCENARIOS[0]["results"]["qa_total_power_eib"])
ax.plot(t[:split+1], y0_rb[:split+1], color="dimgray", linewidth=1.2, alpha=0.5, linestyle="-")
ax.plot(t[:split+1], y0_qa[:split+1], **HIST_KW, label="Historical")
for sc in SCENARIOS:
    y_rb = d(sc["results"]["rb_total_power_eib"])
    y_qa = d(sc["results"]["qa_total_power_eib"])
    ax.plot(t[split:], y_rb[split:], color=sc["color"], linewidth=1.2,
            linestyle=":", alpha=0.6)
    ax.plot(t[split:], y_qa[split:], color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=f"{sc['name']} (QAP)")
add_vline(ax); ax.set_ylabel("EiB"); ax.set_title("Network Power (RBP thin, QAP bold)", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[1, 1]
plot_scenarios(ax, "day_network_reward")
add_vline(ax); ax.set_ylabel("FIL/day"); ax.set_title("Minting Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[1, 2]
# Pledge/Reward locked ratio
for sc in SCENARIOS:
    y_pledge = d(sc["results"]["network_locked_pledge"])
    y_reward = d(sc["results"]["network_locked_reward"])
    y_ratio = y_pledge / np.maximum(y_reward, 1e-10)
    ax.plot(t[1:split+1], y_ratio[1:split+1], **HIST_KW, label="Historical" if sc is SCENARIOS[0] else None)
    ax.plot(t[split:], y_ratio[split:], color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("Ratio"); ax.set_title("Pledge / Reward Locked", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# ── Row 2: Supply ──────────────────────────────────────────────────
ax = axes[2, 0]
plot_scenarios(ax, "network_locked", scale=1e-6)
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Network Locked", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[2, 1]
plot_scenarios(ax, "circ_supply", scale=1e-6)
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Circulating Supply", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(0, 1000)

ax = axes[2, 2]
y0_nl = d(SCENARIOS[0]["results"]["network_locked"])
y0_cs = d(SCENARIOS[0]["results"]["circ_supply"])
ax.plot(t[:split+1], (y0_nl/y0_cs)[:split+1], **HIST_KW, label="Historical")
for sc in SCENARIOS:
    nl = d(sc["results"]["network_locked"])
    cs = d(sc["results"]["circ_supply"])
    ax.plot(t[split:], (nl/cs)[split:], color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("Ratio"); ax.set_title("Locked / Circ Supply", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# ── Row 3: Economics ───────────────────────────────────────────────
ax = axes[3, 0]
y0 = d(SCENARIOS[0]["results"]["day_pledge_per_QAP"])
ax.plot(t[1:split+1], y0[1:split+1], **HIST_KW, label="Historical")
for sc in SCENARIOS:
    y = d(sc["results"]["day_pledge_per_QAP"])
    ax.plot(t[split:], y[split:], color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("FIL"); ax.set_title("Pledge / 32 GiB QA Sector", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[3, 1]
for sc in SCENARIOS:
    y_roi = np.array(sc["results"]["1y_sector_roi"]) * 100
    y = y_roi[:N]
    t_roi = t[:len(y)]
    roi_split = min(split, len(y))
    if sc is SCENARIOS[0]:
        ax.plot(t_roi[:roi_split], y[:roi_split], **HIST_KW, label="Historical")
    ax.plot(t_roi[roi_split-1:], y[roi_split-1:], color=sc["color"], linewidth=2.0,
            linestyle=sc["linestyle"], label=sc["name"])
add_vline(ax); ax.set_ylabel("%"); ax.set_title("1Y Realized FoFR", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[3, 2]
plot_scenarios(ax, "day_rewards_per_sector")
add_vline(ax); ax.set_ylabel("FIL/day"); ax.set_title("Daily Rewards / Sector", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# Format shared x-axis on bottom row only
for ax in axes[3, :]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.30)
fig.savefig("forecast_panel.png", dpi=150, bbox_inches="tight")
print(f"\nForecast panel saved -> forecast_panel.png")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2: Onboarding Required to Offset SP Attrition
# ════════════════════════════════════════════════════════════════════
# At steady state, to keep QAP constant:
#   new_QAP/day = (1 - rr) * QAP / sector_duration
#   new_RBP/day = new_QAP/day / qa_multiplier
#
# qa_multiplier = 1 + 9 * fil_plus_rate  (10x at 100% FIL+)

qa_current_eib = float(np.array(SCENARIOS[0]["results"]["qa_total_power_eib"])[split])
qa_mult = 1.0 + 9.0 * fpr_val  # effective QA multiplier

rr_range = np.linspace(0, 1, 200)
# Required onboarding to maintain QAP (PiB/day)
required_rbp_pib = (1 - rr_range) * qa_current_eib * 1024.0 / (SECTOR_DURATION * qa_mult)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(rr_range * 100, required_rbp_pib, color="steelblue", linewidth=2.5,
         label="Required RBP to maintain QAP")
ax3.axhline(rbp_pib_day, color="darkorange", linewidth=1.5, linestyle="--",
            label=f"Current onboarding ({rbp_pib_day:.2f} PiB/d)")
ax3.axvline(rr_val * 100, color="dimgray", linewidth=1, linestyle=":", alpha=0.6)

# Find the breakeven RR (where current onboarding = required)
rr_breakeven = 1.0 - rbp_pib_day * SECTOR_DURATION * qa_mult / (qa_current_eib * 1024.0)
rr_breakeven = max(0, min(1, rr_breakeven))

# Mark current operating point and breakeven
ax3.plot(rr_val * 100, rbp_pib_day, "o", color="darkorange", markersize=12, zorder=5)
ax3.annotate(
    f"Current: RR={rr_val*100:.0f}%, RBP={rbp_pib_day:.2f} PiB/d\n"
    f"Breakeven: RR={rr_breakeven*100:.1f}%  (margin: {(rr_val - rr_breakeven)*100:+.1f}pp)",
    xy=(rr_val * 100, rbp_pib_day),
    xytext=(45, rbp_pib_day + 1.2),
    fontsize=10, color="darkorange", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
)

# Shade deficit zone (where current onboarding < required)
deficit_mask = required_rbp_pib > rbp_pib_day
ax3.fill_between(rr_range[deficit_mask] * 100, rbp_pib_day, required_rbp_pib[deficit_mask],
                 alpha=0.15, color="crimson", label="Deficit (QAP shrinking)")
surplus_mask = required_rbp_pib <= rbp_pib_day
ax3.fill_between(rr_range[surplus_mask] * 100, required_rbp_pib[surplus_mask], rbp_pib_day,
                 alpha=0.15, color="seagreen", label="Surplus (QAP growing)")

ax3.set_xlabel("Renewal Rate (%)", fontsize=12)
ax3.set_ylabel("RB Onboarding Required (PiB/day)", fontsize=12)
ax3.set_title(
    f"Onboarding Required to Maintain Network QAP ({qa_current_eib:.1f} EiB)\n"
    f"Sector duration={SECTOR_DURATION}d,  FIL+ multiplier={qa_mult:.1f}x",
    fontsize=12, fontweight="bold",
)
ax3.set_xlim(0, 100)
ax3.set_ylim(bottom=0)
ax3.legend(fontsize=9, loc="upper right")
ax3.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("onboarding_required.png", dpi=150, bbox_inches="tight")
print(f"Onboarding required plot saved -> onboarding_required.png")


# ════════════════════════════════════════════════════════════════════
# FIGURE 3: Before / After Bug Fix Comparison (uses Flat scenario)
# ════════════════════════════════════════════════════════════════════
print("Running simulation [old 50/50] ...")
sim_data_old = {k: v for k, v in sim_data.items() if k != "locked_reward_zero"}
rbp_flat = jnp.ones(SIM_FCST) * rbp_pib_day
results_fixed = SCENARIOS[0]["results"]  # Flat scenario
results_old = sim.run_sim(
    rbp_flat, rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, SIM_FCST, SECTOR_DURATION,
    sim_data_old,
)
print(f"  done")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle(
    "Bug Fix: 50/50 Pledge/Reward Split -> Data-Driven Linear Vesting\n"
    f"Forecast from {CURRENT_DATE},  start_date={START_DATE}",
    fontsize=12, fontweight="bold", y=0.98,
)

OLD_KW  = dict(color="red",       linewidth=1.8, linestyle="--", alpha=0.8, label="Old (50/50)")
NEW_KW  = dict(color="steelblue", linewidth=1.8, linestyle="-",             label="Fixed (linear vesting)")

# Network Locked
ax = axes2[0, 0]
y_new = d(results_fixed["network_locked"]) / 1e6
y_old = d(results_old["network_locked"]) / 1e6
y_new_pledge = d(results_fixed["network_locked_pledge"]) / 1e6
y_old_pledge = d(results_old["network_locked_pledge"]) / 1e6
ax.fill_between(t, y_new_pledge, y_new, alpha=0.2, color="darkorange")
ax.fill_between(t, 0, y_new_pledge, alpha=0.2, color="steelblue")
ax.plot(t, y_new, color="steelblue", linewidth=1.5, label="Fixed (total)")
ax.plot(t, y_new_pledge, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6, label="Fixed (pledge)")
ax.plot(t, y_old, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="Old 50/50 (total)")
ax.plot(t, y_old_pledge, color="red", linewidth=0.8, linestyle=":", alpha=0.4, label="Old 50/50 (pledge)")
ax.plot(oc_dates, oc_locked / 1e6, **OC_KW, label="On-chain")
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Network Locked (Pledge + Reward)", fontsize=11)
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.2); ax.margins(y=0.05)

# Circulating Supply
ax = axes2[0, 1]
y_new = d(results_fixed["circ_supply"]) / 1e6
y_old = d(results_old["circ_supply"]) / 1e6
ax.plot(oc_dates, oc_circ / 1e6, **OC_KW, label="On-chain")
ax.plot(t[:split+1], y_old[:split+1], color="red", linewidth=1, alpha=0.4, linestyle="--")
ax.plot(t[split:], y_old[split:], **OLD_KW)
ax.plot(t[:split+1], y_new[:split+1], color="steelblue", linewidth=1, alpha=0.4)
ax.plot(t[split:], y_new[split:], **NEW_KW)
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Circulating Supply", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.margins(y=0.05)

# FoFR
ax = axes2[1, 0]
y_new_full = np.array(results_fixed["1y_sector_roi"]) * 100
y_old_full = np.array(results_old["1y_sector_roi"]) * 100
disp2 = min(len(y_new_full), total_days)
y_new_r = y_new_full[:disp2]; y_old_r = y_old_full[:disp2]
t_roi = t[:disp2]
roi_split = min(split, disp2)
ax.plot(t_roi[:roi_split], y_old_r[:roi_split], color="red", linewidth=1, alpha=0.4, linestyle="--")
ax.plot(t_roi[roi_split-1:], y_old_r[roi_split-1:], **OLD_KW)
ax.plot(t_roi[:roi_split], y_new_r[:roi_split], color="steelblue", linewidth=1, alpha=0.4)
ax.plot(t_roi[roi_split-1:], y_new_r[roi_split-1:], **NEW_KW)
add_vline(ax); ax.set_ylabel("%"); ax.set_title("1Y Realized FoFR", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.margins(y=0.05)

# Initialization split (bar chart)
ax = axes2[1, 1]
labels = ["Old (50/50)", "Fixed (linear)"]
reward_vals = [locked_total / 2 / 1e6, locked_reward / 1e6]
pledge_vals = [locked_total / 2 / 1e6, (locked_total - locked_reward) / 1e6]
x = np.arange(len(labels))
w = 0.35
b1 = ax.bar(x - w/2, pledge_vals, w, label="Pledge-locked", color="steelblue", alpha=0.8)
b2 = ax.bar(x + w/2, reward_vals, w, label="Reward-locked", color="darkorange", alpha=0.8)
ax.set_ylabel("M FIL"); ax.set_title("Initial Pledge/Reward Split", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis="y")
for b in b1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{b.get_height():.1f}M", ha="center", va="bottom", fontsize=8)
for b in b2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
            f"{b.get_height():.1f}M", ha="center", va="bottom", fontsize=8)

for ax in [axes2[0, 0], axes2[0, 1], axes2[1, 0]]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

fig2.subplots_adjust(top=0.88, hspace=0.30, wspace=0.25)
fig2.savefig("bugfix_comparison.png", dpi=150, bbox_inches="tight")
print(f"Bug fix comparison saved -> bugfix_comparison.png")


# ── Summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FORECAST SUMMARY")
print(f"{'='*60}")

end_idx = total_days - 1
header_cols = "  ".join(f"{sc['name']:>15s}" for sc in SCENARIOS)
print(f"\n  {'Metric':<25s} {'At Start':>15s}  {header_cols}")
print(f"  {'-'*25} {'-'*15}  {'  '.join('-'*15 for _ in SCENARIOS)}")

# Network Locked
vals = "  ".join(f"{np.array(sc['results']['network_locked'])[end_idx]/1e6:>15.1f}" for sc in SCENARIOS)
print(f"  {'Network Locked (M)':<25s} {np.array(SCENARIOS[0]['results']['network_locked'])[split]/1e6:>15.1f}  {vals}")

# Circ Supply
vals = "  ".join(f"{np.array(sc['results']['circ_supply'])[end_idx]/1e6:>15.1f}" for sc in SCENARIOS)
print(f"  {'Circ Supply (M)':<25s} {np.array(SCENARIOS[0]['results']['circ_supply'])[split]/1e6:>15.1f}  {vals}")

# L/CS
def lcs(sc, idx):
    nl = float(np.array(sc['results']['network_locked'])[idx])
    cs = float(np.array(sc['results']['circ_supply'])[idx])
    return nl / cs
vals = "  ".join(f"{lcs(sc, end_idx):>15.4f}" for sc in SCENARIOS)
print(f"  {'L/CS Ratio':<25s} {lcs(SCENARIOS[0], split):>15.4f}  {vals}")

# Pledge/Sector
vals = "  ".join(f"{float(np.array(sc['results']['day_pledge_per_QAP'])[end_idx]):>15.4f}" for sc in SCENARIOS)
print(f"  {'Pledge/Sector (FIL)':<25s} {float(np.array(SCENARIOS[0]['results']['day_pledge_per_QAP'])[split]):>15.4f}  {vals}")

# FoFR
roi_vals = []
for sc in SCENARIOS:
    roi = np.array(sc["results"]["1y_sector_roi"])
    roi_end = min(end_idx, len(roi) - 1)
    roi_vals.append(f"{roi[roi_end]*100:>15.2f}")
vals = "  ".join(roi_vals)
roi0 = np.array(SCENARIOS[0]["results"]["1y_sector_roi"])
print(f"  {'1Y FoFR (%)':<25s} {roi0[min(split, len(roi0)-1)]*100:>15.2f}  {vals}")

plt.show()
