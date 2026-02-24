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
1-year Filecoin forecast using last-30-day medians as static inputs.

Produces two figures:
  1. forecast_panel.png  – 4x3 panel (inputs, power, supply, economics)
  2. bugfix_comparison.png – before/after comparison of the 50/50 fix

Historical data shown as solid, forecast as dashed, on-chain as black.
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
FORECAST_DAYS   = 365           # 1-year forecast
SIM_EXTRA       = 365           # extra days so FoFR convolution extends through forecast
SECTOR_DURATION = 360
LOCK_TARGET     = 0.3
LOOKBACK        = 30            # days to median for input derivation
# ────────────────────────────────────────────────────────────────────

START_DATE = CURRENT_DATE - timedelta(days=HISTORY_DAYS)
END_DATE   = CURRENT_DATE + timedelta(days=FORECAST_DAYS)
SIM_END    = CURRENT_DATE + timedelta(days=FORECAST_DAYS + SIM_EXTRA)  # extended for FoFR

print(f"History:  {START_DATE} -> {CURRENT_DATE}  ({HISTORY_DAYS} days)")
print(f"Forecast: {CURRENT_DATE} -> {END_DATE}  ({FORECAST_DAYS} days)")

# ── Fetch data ──────────────────────────────────────────────────────
print("\nFetching simulation data ...")
t0 = time.time()
sim_data = mdata.get_simulation_data(AUTH_TOKEN, START_DATE, CURRENT_DATE, SIM_END)
print(f"  done ({time.time()-t0:.1f}s)")

# ── Derive 30-day median inputs ────────────────────────────────────
rb_pib  = sim_data["historical_onboarded_rb_power_pib"][-LOOKBACK:]
qa_pib  = sim_data["historical_onboarded_qa_power_pib"][-LOOKBACK:]
rr_hist = sim_data["historical_renewal_rate"][-LOOKBACK:]

rbp_pib_day = float(np.median(rb_pib))
rbp_eib_day = rbp_pib_day / 1024.0
rr_val      = float(np.median(rr_hist))
ratio       = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val     = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

print(f"\nInputs (30-day median, held constant for forecast):")
print(f"  RB Onboarding : {rbp_pib_day:.2f} PiB/day  ({rbp_eib_day:.4f} EiB/day)")
print(f"  Renewal Rate  : {rr_val:.4f}  ({rr_val*100:.1f}%)")
print(f"  FIL+ Rate     : {fpr_val:.4f}  ({fpr_val*100:.1f}%)")

# Report pledge/reward split
locked_total = sim_data["locked_fil_zero"]
locked_reward = sim_data.get("locked_reward_zero", locked_total / 2.0)
print(f"\nInitialization:")
print(f"  locked_fil_zero   = {locked_total/1e6:.2f}M FIL")
print(f"  locked_reward_est = {locked_reward/1e6:.2f}M FIL  ({locked_reward/locked_total*100:.1f}%)")
print(f"  locked_pledge_est = {(locked_total-locked_reward)/1e6:.2f}M FIL  ({(1-locked_reward/locked_total)*100:.1f}%)")

# ── Run simulations ─────────────────────────────────────────────────
SIM_FCST = FORECAST_DAYS + SIM_EXTRA
rbp_vec = jnp.ones(SIM_FCST) * rbp_eib_day
rr_vec  = jnp.ones(SIM_FCST) * rr_val
fpr_vec = jnp.ones(SIM_FCST) * fpr_val

print("\nRunning simulation [fixed] ...")
t0 = time.time()
results = sim.run_sim(
    rbp_vec, rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, SIM_FCST, SECTOR_DURATION,
    sim_data,
)
print(f"  done ({time.time()-t0:.1f}s)")

# Run the old 50/50 version for comparison
print("Running simulation [old 50/50] ...")
sim_data_old = {k: v for k, v in sim_data.items() if k != "locked_reward_zero"}
results_old = sim.run_sim(
    rbp_vec, rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, SIM_FCST, SECTOR_DURATION,
    sim_data_old,
)
print(f"  done")

# ── Fetch on-chain actuals ──────────────────────────────────────────
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
N = total_days  # display window length; results arrays may be longer due to SIM_EXTRA

def d(arr):
    """Slice a result array to the display window."""
    return np.array(arr)[:N]

# ── Build input vectors for plotting ────────────────────────────────
hist_rbp = np.array(sim_data["historical_onboarded_rb_power_pib"])
hist_rr  = np.array(sim_data["historical_renewal_rate"])
hist_qa  = np.array(sim_data["historical_onboarded_qa_power_pib"])
hist_ratio = np.where(hist_rbp > 0, hist_qa / hist_rbp, 1.0)
hist_fpr = np.clip((hist_ratio - 1.0) / 9.0, 0.0, 1.0)

# ── Shared plot helpers ─────────────────────────────────────────────
HIST_KW  = dict(color="steelblue", linewidth=1.5, linestyle="-")
FCST_KW  = dict(color="steelblue", linewidth=2.0, linestyle="--")
OC_KW    = dict(color="black", linewidth=1.5, linestyle="-", alpha=0.7)
VLINE_KW = dict(color="dimgray", linewidth=0.8, linestyle=":", alpha=0.6)
INPUT_KW = dict(color="darkorange", linewidth=1.5, linestyle="-")
INPUT_FCST_KW = dict(color="darkorange", linewidth=2.0, linestyle="--")

def add_vline(ax):
    ax.axvline(CURRENT_DATE, **VLINE_KW)

# ════════════════════════════════════════════════════════════════════
# FIGURE 1: Forecast Panel
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 3, figsize=(17, 14), sharex=True)
fig.suptitle(
    f"Filecoin 1Y Forecast  |  {CURRENT_DATE}\n"
    f"Inputs: RBP={rbp_pib_day:.2f} PiB/d,  RR={rr_val*100:.1f}%,  FIL+={fpr_val*100:.1f}%  (30d median, static)",
    fontsize=12, fontweight="bold", y=0.98,
)

# ── Row 0: Inputs ──────────────────────────────────────────────────
ax = axes[0, 0]
t_inp = t[:len(hist_rbp)]
ax.plot(t_inp, hist_rbp, **INPUT_KW, label="Historical")
ax.plot(t[split:], [rbp_pib_day]*len(t[split:]), **INPUT_FCST_KW, label=f"Forecast ({rbp_pib_day:.2f})")
add_vline(ax); ax.set_ylabel("PiB/day"); ax.set_title("RB Onboarding Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[0, 1]
t_rr = t[:len(hist_rr)]
ax.plot(t_rr, hist_rr * 100, **INPUT_KW, label="Historical")
ax.plot(t[split:], [rr_val*100]*len(t[split:]), **INPUT_FCST_KW, label=f"Forecast ({rr_val*100:.1f}%)")
add_vline(ax); ax.set_ylabel("%"); ax.set_title("Renewal Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(0, 100)

ax = axes[0, 2]
t_fpr = t[:len(hist_fpr)]
ax.plot(t_fpr, hist_fpr * 100, **INPUT_KW, label="Historical")
ax.plot(t[split:], [fpr_val*100]*len(t[split:]), **INPUT_FCST_KW, label=f"Forecast ({fpr_val*100:.1f}%)")
add_vline(ax); ax.set_ylabel("%"); ax.set_title("FIL+ Rate", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(0, 100)

# ── Row 1: Power ───────────────────────────────────────────────────
ax = axes[1, 0]
y = d(results["rb_total_power_eib"])
ax.plot(t[:split+1], y[:split+1], **HIST_KW)
ax.plot(t[split:], y[split:], **FCST_KW)
add_vline(ax); ax.set_ylabel("EiB"); ax.set_title("Raw Byte Power", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[1, 1]
y = d(results["qa_total_power_eib"])
ax.plot(t[:split+1], y[:split+1], **HIST_KW, label="QAP")
ax.plot(t[split:], y[split:], **FCST_KW)
baseline = minting.compute_baseline_power_array(
    np.datetime64(START_DATE), np.datetime64(END_DATE), sim_data["init_baseline_eib"],
)
ax.plot(t, np.array(baseline), ":", color="red", linewidth=1, label="Baseline")
add_vline(ax); ax.set_ylabel("EiB"); ax.set_title("Quality Adj. Power", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[1, 2]
y = d(results["day_network_reward"])
ax.plot(t[:split+1], y[:split+1], **HIST_KW)
ax.plot(t[split:], y[split:], **FCST_KW)
add_vline(ax); ax.set_ylabel("FIL/day"); ax.set_title("Minting Rate", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# ── Row 2: Supply ──────────────────────────────────────────────────
ax = axes[2, 0]
y_total = d(results["network_locked"]) / 1e6
y_pledge = d(results["network_locked_pledge"]) / 1e6
y_reward = d(results["network_locked_reward"]) / 1e6
ax.fill_between(t, y_pledge, y_total, alpha=0.3, color="darkorange", label="Reward-locked")
ax.fill_between(t, 0, y_pledge, alpha=0.3, color="steelblue", label="Pledge-locked")
ax.plot(t, y_total, color="steelblue", linewidth=1.2)
ax.plot(oc_dates, oc_locked / 1e6, **OC_KW, label="On-chain")
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Network Locked (Pledge + Reward)", fontsize=10)
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[2, 1]
y = d(results["circ_supply"]) / 1e6
ax.plot(oc_dates, oc_circ / 1e6, **OC_KW, label="On-chain")
ax.plot(t[:split+1], y[:split+1], **HIST_KW, label="Sim (hist)")
ax.plot(t[split:], y[split:], **FCST_KW, label="Forecast")
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Circulating Supply", fontsize=10)
ax.legend(fontsize=7); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[2, 2]
nl = d(results["network_locked"])
cs = d(results["circ_supply"])
ax.plot(t[:split+1], (nl/cs)[:split+1], **HIST_KW)
ax.plot(t[split:], (nl/cs)[split:], **FCST_KW)
add_vline(ax); ax.set_ylabel("Ratio"); ax.set_title("Locked / Circ Supply", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# ── Row 3: Economics ───────────────────────────────────────────────
ax = axes[3, 0]
y = d(results["day_pledge_per_QAP"])
ax.plot(t[1:split+1], y[1:split+1], **HIST_KW)
ax.plot(t[split:], y[split:], **FCST_KW)
add_vline(ax); ax.set_ylabel("FIL"); ax.set_title("Pledge / 32 GiB QA Sector", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[3, 1]
y_roi = np.array(results["1y_sector_roi"]) * 100
y = y_roi[:N]
t_roi = t[:len(y)]
roi_split = min(split, len(y))
ax.plot(t_roi[:roi_split], y[:roi_split], **HIST_KW)
ax.plot(t_roi[roi_split-1:], y[roi_split-1:], **FCST_KW)
add_vline(ax); ax.set_ylabel("%"); ax.set_title("1Y Realized FoFR", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

ax = axes[3, 2]
y = d(results["day_rewards_per_sector"])
ax.plot(t[:split+1], y[:split+1], **HIST_KW)
ax.plot(t[split:], y[split:], **FCST_KW)
add_vline(ax); ax.set_ylabel("FIL/day"); ax.set_title("Daily Rewards / Sector", fontsize=10)
ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# Format shared x-axis on bottom row only
for ax in axes[3, :]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

fig.subplots_adjust(top=0.92, hspace=0.25, wspace=0.30)
fig.savefig("forecast_panel.png", dpi=150, bbox_inches="tight")
print(f"\nForecast panel saved -> forecast_panel.png")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2: Before / After Bug Fix Comparison
# ════════════════════════════════════════════════════════════════════
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
y_new = d(results["network_locked"]) / 1e6
y_old = d(results_old["network_locked"]) / 1e6
y_new_pledge = d(results["network_locked_pledge"]) / 1e6
y_new_reward = d(results["network_locked_reward"]) / 1e6
y_old_pledge = d(results_old["network_locked_pledge"]) / 1e6
y_old_reward = d(results_old["network_locked_reward"]) / 1e6
# Fixed: stacked areas
ax.fill_between(t, y_new_pledge, y_new, alpha=0.2, color="darkorange")
ax.fill_between(t, 0, y_new_pledge, alpha=0.2, color="steelblue")
ax.plot(t, y_new, color="steelblue", linewidth=1.5, label="Fixed (total)")
ax.plot(t, y_new_pledge, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.6, label="Fixed (pledge)")
# Old 50/50
ax.plot(t, y_old, color="red", linewidth=1.5, linestyle="--", alpha=0.8, label="Old 50/50 (total)")
ax.plot(t, y_old_pledge, color="red", linewidth=0.8, linestyle=":", alpha=0.4, label="Old 50/50 (pledge)")
# On-chain
ax.plot(oc_dates, oc_locked / 1e6, **OC_KW, label="On-chain")
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Network Locked (Pledge + Reward)", fontsize=11)
ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# Circulating Supply
ax = axes2[0, 1]
y_new = d(results["circ_supply"]) / 1e6
y_old = d(results_old["circ_supply"]) / 1e6
ax.plot(oc_dates, oc_circ / 1e6, **OC_KW, label="On-chain")
ax.plot(t[:split+1], y_old[:split+1], color="red", linewidth=1, alpha=0.4, linestyle="--")
ax.plot(t[split:], y_old[split:], **OLD_KW)
ax.plot(t[:split+1], y_new[:split+1], color="steelblue", linewidth=1, alpha=0.4)
ax.plot(t[split:], y_new[split:], **NEW_KW)
add_vline(ax); ax.set_ylabel("M FIL"); ax.set_title("Circulating Supply", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

# FoFR
ax = axes2[1, 0]
y_new_full = np.array(results["1y_sector_roi"]) * 100
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
ax.legend(fontsize=8); ax.grid(True, alpha=0.2); ax.set_ylim(bottom=0)

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
# annotate bars
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
nl = np.array(results["network_locked"])
cs = np.array(results["circ_supply"])
roi = np.array(results["1y_sector_roi"])
pledge = np.array(results["day_pledge_per_QAP"])
end_idx = total_days - 1  # display end index (not extended sim end)

nl_old = np.array(results_old["network_locked"])
cs_old = np.array(results_old["circ_supply"])

print(f"\n  {'Metric':<25s} {'Fixed (start)':>15s} {'Fixed (+12mo)':>15s} {'Old 50/50 (+12mo)':>18s}")
print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*18}")
print(f"  {'Network Locked (M)':<25s} {nl[split]/1e6:>15.1f} {nl[end_idx]/1e6:>15.1f} {nl_old[end_idx]/1e6:>18.1f}")
print(f"  {'Circ Supply (M)':<25s} {cs[split]/1e6:>15.1f} {cs[end_idx]/1e6:>15.1f} {cs_old[end_idx]/1e6:>18.1f}")
print(f"  {'L/CS Ratio':<25s} {nl[split]/cs[split]:>15.4f} {nl[end_idx]/cs[end_idx]:>15.4f} {nl_old[end_idx]/cs_old[end_idx]:>18.4f}")
print(f"  {'Pledge/Sector (FIL)':<25s} {pledge[split]:>15.4f} {pledge[end_idx]:>15.4f}")
if split < len(roi) and len(roi) > 1:
    roi_end = min(end_idx, len(roi) - 1)
    print(f"  {'1Y FoFR (%)':<25s} {roi[min(split, len(roi)-1)]*100:>15.2f} {roi[roi_end]*100:>15.2f}")
plt.show()
