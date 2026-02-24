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
Verify whether sim.run_sim expects rb_onboard_power in EiB or PiB
by checking power trajectory continuity at the history/forecast boundary.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import numpy as np
import jax.numpy as jnp

import mechafil_jax.data as mdata
import mechafil_jax.sim as sim

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

# Derive inputs
rb_pib = np.array(sim_data["historical_onboarded_rb_power_pib"][-30:])
qa_pib = np.array(sim_data["historical_onboarded_qa_power_pib"][-30:])
rr_hist = np.array(sim_data["historical_renewal_rate"][-30:])

rbp_pib_day = float(np.median(rb_pib))
rbp_eib_day = rbp_pib_day / 1024.0
rr_val = float(np.median(rr_hist))
ratio = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

rr_vec = jnp.ones(FORECAST_DAYS) * rr_val
fpr_vec = jnp.ones(FORECAST_DAYS) * fpr_val

split = HISTORY_DAYS

print(f"\nRBP: {rbp_pib_day:.2f} PiB/d = {rbp_eib_day:.6f} EiB/d")
print(f"Historical daily RB onboard (last 5): {np.array(sim_data['historical_onboarded_rb_power_pib'][-5:])}")
print(f"  (these are in PiB)")

# Run sim passing EiB (as current forecast.py does)
print("\n=== SIM WITH EiB INPUT ===")
r_eib = sim.run_sim(
    jnp.ones(FORECAST_DAYS) * rbp_eib_day, rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, FORECAST_DAYS, SECTOR_DURATION,
    sim_data,
)
rb_eib = np.array(r_eib["rb_total_power_eib"])
qa_eib = np.array(r_eib["qa_total_power_eib"])
rb_ob_pib_eib = np.array(r_eib["rb_day_onboarded_power_pib"])
print(f"Historical RBP[-3:]: {rb_eib[split-3:split]} EiB")
print(f"Forecast   RBP[:3]:  {rb_eib[split:split+3]} EiB")
print(f"Forecast   RBP[-3:]: {rb_eib[-3:]} EiB")
print(f"Historical QAP[-3:]: {qa_eib[split-3:split]} EiB")
print(f"Forecast   QAP[:3]:  {qa_eib[split:split+3]} EiB")
print(f"Forecast   QAP[-3:]: {qa_eib[-3:]} EiB")
print(f"Daily onboarded RBP (hist last 3, PiB): {rb_ob_pib_eib[split-3:split]}")
print(f"Daily onboarded RBP (fcst first 3, PiB): {rb_ob_pib_eib[split:split+3]}")

# Run sim passing PiB (what we think is correct)
print("\n=== SIM WITH PiB INPUT ===")
r_pib = sim.run_sim(
    jnp.ones(FORECAST_DAYS) * rbp_pib_day, rr_vec, fpr_vec,
    LOCK_TARGET, START_DATE, CURRENT_DATE, FORECAST_DAYS, SECTOR_DURATION,
    sim_data,
)
rb_pib_res = np.array(r_pib["rb_total_power_eib"])
qa_pib_res = np.array(r_pib["qa_total_power_eib"])
rb_ob_pib = np.array(r_pib["rb_day_onboarded_power_pib"])
print(f"Historical RBP[-3:]: {rb_pib_res[split-3:split]} EiB")
print(f"Forecast   RBP[:3]:  {rb_pib_res[split:split+3]} EiB")
print(f"Forecast   RBP[-3:]: {rb_pib_res[-3:]} EiB")
print(f"Historical QAP[-3:]: {qa_pib_res[split-3:split]} EiB")
print(f"Forecast   QAP[:3]:  {qa_pib_res[split:split+3]} EiB")
print(f"Forecast   QAP[-3:]: {qa_pib_res[-3:]} EiB")
print(f"Daily onboarded RBP (hist last 3, PiB): {rb_ob_pib[split-3:split]}")
print(f"Daily onboarded RBP (fcst first 3, PiB): {rb_ob_pib[split:split+3]}")

print("\n=== SUMMARY ===")
print(f"Last historical QAP: {qa_eib[split-1]:.3f} EiB")
print(f"First forecast QAP (EiB input): {qa_eib[split]:.3f} EiB  (delta: {qa_eib[split]-qa_eib[split-1]:+.3f})")
print(f"First forecast QAP (PiB input): {qa_pib_res[split]:.3f} EiB  (delta: {qa_pib_res[split]-qa_pib_res[split-1]:+.3f})")
print(f"End forecast QAP (EiB input): {qa_eib[-1]:.3f} EiB")
print(f"End forecast QAP (PiB input): {qa_pib_res[-1]:.3f} EiB")
print()
print(f"Last hist daily RB onboard: {rb_ob_pib_eib[split-1]:.4f} PiB")
print(f"First fcst daily RB onboard (EiB input): {rb_ob_pib_eib[split]:.6f} PiB  <-- 1024x too small!")
print(f"First fcst daily RB onboard (PiB input): {rb_ob_pib[split]:.4f} PiB  <-- matches historical")

# Locked FIL comparison
nl_eib = float(np.array(r_eib["network_locked"])[-1])
nl_pib = float(np.array(r_pib["network_locked"])[-1])
print(f"\nNetwork Locked @ +1Y:")
print(f"  EiB input: {nl_eib/1e6:.1f}M FIL")
print(f"  PiB input: {nl_pib/1e6:.1f}M FIL")
print(f"  Delta:     {(nl_pib-nl_eib)/1e6:+.1f}M FIL")
