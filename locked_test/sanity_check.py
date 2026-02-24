#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax",
#   "jaxlib",
#   "numpy",
#   "requests",
#   "pandas",
#   "pystarboard @ git+https://github.com/CELtd/pystarboard.git",
#   "mechafil-jax @ file:///home/kiran/code/cel/mechafil-jax",
# ]
# ///
"""
Sanity check: verify the fixed forecast produces numbers that
are consistent with on-chain data and analytical expectations.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import numpy as np
import jax.numpy as jnp

import mechafil_jax.data as mdata
import mechafil_jax.sim as sim
from pystarboard.data_spacescope import SpacescopeDataConnection

AUTH = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
CURRENT_DATE = date.today() - timedelta(days=3)
HISTORY_DAYS = 365
FORECAST_DAYS = 365
SECTOR_DURATION = 360
LOCK_TARGET = 0.3

START_DATE = CURRENT_DATE - timedelta(days=HISTORY_DAYS)
SIM_END = CURRENT_DATE + timedelta(days=FORECAST_DAYS)

print("Fetching data ...")
sim_data = mdata.get_simulation_data(AUTH, START_DATE, CURRENT_DATE, SIM_END)

# On-chain actuals
supply_df = SpacescopeDataConnection.query_spacescope_supply_stats(START_DATE, CURRENT_DATE)
supply_df = supply_df.sort_values("date")
oc_locked = supply_df["locked_fil"].astype(float).values
oc_circ = supply_df["circulating_fil"].astype(float).values

# Derive inputs (PiB!)
rb_pib = np.array(sim_data["historical_onboarded_rb_power_pib"][-30:])
qa_pib = np.array(sim_data["historical_onboarded_qa_power_pib"][-30:])
rr_hist = np.array(sim_data["historical_renewal_rate"][-30:])

rbp_pib_day = float(np.median(rb_pib))
rr_val = float(np.median(rr_hist))
ratio = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

rr_vec = jnp.ones(FORECAST_DAYS) * rr_val
fpr_vec = jnp.ones(FORECAST_DAYS) * fpr_val

# Run sim with CORRECT PiB input
print("Running sim (PiB input)...")
r = sim.run_sim(
    jnp.ones(FORECAST_DAYS) * rbp_pib_day,
    rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, FORECAST_DAYS, SECTOR_DURATION,
    sim_data,
)

split = HISTORY_DAYS
N = HISTORY_DAYS + FORECAST_DAYS

# Check 1: Historical power continuity
print("\n=== CHECK 1: Power continuity at history/forecast boundary ===")
rb_eib = np.array(r["rb_total_power_eib"])
qa_eib = np.array(r["qa_total_power_eib"])
print(f"RBP  day {split-1}: {rb_eib[split-1]:.4f} EiB")
print(f"RBP  day {split}:   {rb_eib[split]:.4f} EiB  (delta: {rb_eib[split]-rb_eib[split-1]:+.4f})")
print(f"RBP  day {split+1}: {rb_eib[split+1]:.4f} EiB  (delta: {rb_eib[split+1]-rb_eib[split]:+.4f})")
print(f"QAP  day {split-1}: {qa_eib[split-1]:.4f} EiB")
print(f"QAP  day {split}:   {qa_eib[split]:.4f} EiB  (delta: {qa_eib[split]-qa_eib[split-1]:+.4f})")
print(f"QAP  day {split+1}: {qa_eib[split+1]:.4f} EiB  (delta: {qa_eib[split+1]-qa_eib[split]:+.4f})")

hist_delta = np.diff(qa_eib[:split])
fcst_delta = np.diff(qa_eib[split:split+30])
print(f"Avg historical QAP delta: {np.mean(hist_delta)*1024:+.2f} PiB/day")
print(f"Avg forecast QAP delta (1st 30d): {np.mean(fcst_delta)*1024:+.2f} PiB/day")
PASS = abs(qa_eib[split] - qa_eib[split-1]) < 0.1
print(f"{'PASS' if PASS else 'FAIL'}: QAP jump at boundary < 0.1 EiB")

# Check 2: Daily onboarded power continuity
print("\n=== CHECK 2: Daily onboarded power continuity ===")
rb_ob = np.array(r["rb_day_onboarded_power_pib"])
print(f"Hist last 3: {rb_ob[split-3:split]}")
print(f"Fcst first 3: {rb_ob[split:split+3]}")
PASS2 = abs(rb_ob[split] - rbp_pib_day) < 0.01
print(f"{'PASS' if PASS2 else 'FAIL'}: First forecast onboarding = median ({rbp_pib_day:.4f} PiB)")

# Check 3: Locked FIL vs on-chain at history end
print("\n=== CHECK 3: Locked FIL vs on-chain ===")
sim_locked = np.array(r["network_locked"])
sim_circ = np.array(r["circ_supply"])
print(f"On-chain locked (last): {oc_locked[-1]/1e6:.2f}M FIL")
print(f"Sim locked at split:    {sim_locked[split]/1e6:.2f}M FIL")
pct_err = abs(sim_locked[split] - oc_locked[-1]) / oc_locked[-1] * 100
print(f"Error: {pct_err:.1f}%")
PASS3 = pct_err < 10
print(f"{'PASS' if PASS3 else 'FAIL'}: Locked FIL error < 10% at history end")

print(f"\nOn-chain circ (last): {oc_circ[-1]/1e6:.2f}M FIL")
print(f"Sim circ at split:    {sim_circ[split]/1e6:.2f}M FIL")
pct_err_c = abs(sim_circ[split] - oc_circ[-1]) / oc_circ[-1] * 100
print(f"Error: {pct_err_c:.1f}%")
PASS3b = pct_err_c < 5
print(f"{'PASS' if PASS3b else 'FAIL'}: Circ supply error < 5% at history end")

# Check 4: Forecast direction makes sense
print("\n=== CHECK 4: Forecast direction ===")
locked_start = float(sim_locked[split])
locked_1y = float(sim_locked[-1])
print(f"Locked start: {locked_start/1e6:.1f}M  -> 1Y: {locked_1y/1e6:.1f}M  (change: {(locked_1y-locked_start)/1e6:+.1f}M)")
# With 82% RR and current onboarding, network should be slowly growing
qa_start = float(qa_eib[split])
qa_1y = float(qa_eib[-1])
print(f"QAP start: {qa_start:.3f} EiB -> 1Y: {qa_1y:.3f} EiB  (change: {(qa_1y-qa_start):+.3f} EiB)")

# Check 5: Minting rate sanity
print("\n=== CHECK 5: Minting rate ===")
minting = np.array(r["day_network_reward"])
print(f"Minting at split: {minting[split]:.0f} FIL/day")
print(f"Minting at +1Y:   {minting[-1]:.0f} FIL/day")
PASS5 = 20000 < minting[split] < 200000
print(f"{'PASS' if PASS5 else 'FAIL'}: Minting rate in reasonable range (20k-200k FIL/day)")

# Check 6: Pledge per sector sanity
print("\n=== CHECK 6: Pledge per sector ===")
dppq = np.array(r["day_pledge_per_QAP"])
print(f"Pledge/sector at split: {dppq[split]:.6f} FIL")
print(f"Pledge/sector at +1Y:   {dppq[-1]:.6f} FIL")
# At current network: ~100M locked, ~20 EiB QAP, 32GiB sector
# Rough: 0.1-0.3 FIL per sector seems reasonable
PASS6 = 0.01 < dppq[split] < 1.0
print(f"{'PASS' if PASS6 else 'FAIL'}: Pledge/sector in reasonable range (0.01-1.0 FIL)")

# Check 7: FoFR sanity
print("\n=== CHECK 7: FoFR ===")
roi = np.array(r["1y_sector_roi"])
roi_split = min(split, len(roi) - 1)
print(f"FoFR at split: {roi[roi_split]*100:.2f}%")
PASS7 = 5 < roi[roi_split]*100 < 100
print(f"{'PASS' if PASS7 else 'FAIL'}: FoFR in reasonable range (5-100%)")

print("\n=== OVERALL ===")
all_pass = all([PASS, PASS2, PASS3, PASS3b, PASS5, PASS6, PASS7])
print(f"{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
