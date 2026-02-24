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
Diagnose units: is rb_onboard_power in EiB or PiB?

The sim docstring says EiB, but let's check what the historical data uses
and what the power forecast actually produces.
"""

import sys; sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import numpy as np
import jax.numpy as jnp

import mechafil_jax.data as mdata
import mechafil_jax.power as power

AUTH = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
CURRENT_DATE = date.today() - timedelta(days=3)
HISTORY_DAYS = 365
FORECAST_DAYS = 365
SECTOR_DURATION = 360

START_DATE = CURRENT_DATE - timedelta(days=HISTORY_DAYS)
SIM_END = CURRENT_DATE + timedelta(days=FORECAST_DAYS)

print("Fetching data ...")
sim_data = mdata.get_simulation_data(AUTH, START_DATE, CURRENT_DATE, SIM_END)

# Check historical values
print("\n=== HISTORICAL DATA (units check) ===")
hist_rb_pib = np.array(sim_data["historical_onboarded_rb_power_pib"])
hist_qa_pib = np.array(sim_data["historical_onboarded_qa_power_pib"])
hist_rb_eib = np.array(sim_data["historical_raw_power_eib"])
hist_qa_eib = np.array(sim_data["historical_qa_power_eib"])

print(f"historical_onboarded_rb_power_pib[-5:]: {hist_rb_pib[-5:]}")
print(f"  -> These are DAILY onboarding rates in PiB")
print(f"historical_raw_power_eib[-5:]: {hist_rb_eib[-5:]}")
print(f"  -> These are TOTAL network power in EiB")
print(f"historical_qa_power_eib[-5:]: {hist_qa_eib[-5:]}")
print(f"  -> These are TOTAL QA power in EiB")

print(f"\nrb_power_zero (PiB): {sim_data['rb_power_zero']}")
print(f"qa_power_zero (PiB): {sim_data['qa_power_zero']}")

# Now check: what does forecast_power_stats expect?
# Its docstring says day_rb_onboarded_power is the input
# Let's pass values and see what total_power comes out to

rb_pib_median = float(np.median(hist_rb_pib[-30:]))
rb_eib_median = rb_pib_median / 1024.0
rr_val = float(np.median(np.array(sim_data["historical_renewal_rate"][-30:])))
ratio = np.where(hist_rb_pib[-30:] > 0, hist_qa_pib[-30:] / hist_rb_pib[-30:], 1.0)
fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

print(f"\n=== POWER FORECAST: PASS EiB ===")
rbp_eib_vec = jnp.ones(FORECAST_DAYS) * rb_eib_median
rr_vec = jnp.ones(FORECAST_DAYS) * rr_val
fpr_vec = jnp.ones(FORECAST_DAYS) * fpr_val

rb_forecast, qa_forecast = power.forecast_power_stats(
    sim_data["rb_power_zero"], sim_data["qa_power_zero"],
    rbp_eib_vec,
    sim_data["rb_known_scheduled_expire_vec"],
    sim_data["qa_known_scheduled_expire_vec"],
    rr_vec, fpr_vec,
    SECTOR_DURATION, FORECAST_DAYS,
)

rb_total = np.array(rb_forecast["total_power"])
qa_total = np.array(qa_forecast["total_power"])
rb_onboard = np.array(rb_forecast["onboarded_power"])
qa_onboard = np.array(qa_forecast["onboarded_power"])

print(f"Input: rb_eib_median = {rb_eib_median:.6f}")
print(f"rb_power_zero (PiB) = {sim_data['rb_power_zero']:.1f}")
print(f"qa_power_zero (PiB) = {sim_data['qa_power_zero']:.1f}")
print(f"rb_forecast total_power[0]: {rb_total[0]:.1f}")
print(f"rb_forecast total_power[-1]: {rb_total[-1]:.1f}")
print(f"qa_forecast total_power[0]: {qa_total[0]:.1f}")
print(f"qa_forecast total_power[-1]: {qa_total[-1]:.1f}")
print(f"rb_forecast onboarded[0]: {rb_onboard[0]:.6f}")
print(f"qa_forecast onboarded[0]: {qa_onboard[0]:.6f}")
print(f"rb_forecast onboarded (cumsum)[-1]: {rb_onboard.sum():.1f}")

print(f"\n=== CHECKING CONCATENATION IN SIM ===")
print(f"In sim.py line 105: rb_total_power_eib = concat(historical_raw_power_eib, rb_power_forecast['total_power'][:-1] / 1024.0)")
print(f"This divides by 1024 -> assumes total_power from forecast is in PiB")
print(f"Forecast total_power[-1] = {rb_total[-1]:.1f} (if PiB = {rb_total[-1]/1024:.3f} EiB)")
print(f"Historical total power last = {hist_rb_eib[-1]:.3f} EiB = {hist_rb_eib[-1]*1024:.1f} PiB")

# So what's the initial rb_power_zero?
print(f"\n=== CONTINUITY CHECK ===")
print(f"rb_power_zero = {sim_data['rb_power_zero']:.1f}")
print(f"historical_raw_power_eib[-1] = {hist_rb_eib[-1]:.3f} EiB = {hist_rb_eib[-1]*1024:.1f} PiB")
print(f"Forecast starts from rb_power_zero + onboarded[0] - expired[0] + renewed[0]")
print(f"So rb_power_zero must be in PiB (same units as forecast total_power)")

# Now: what does the forecast do with the rb_onboard_power input?
# In power.py line 150: total_rb_onboarded_power = day_rb_onboarded_power.cumsum()
# line 183: rb_total_power = rb_power_zero + cumsum(onboarded) - cumsum(expired) + cumsum(renewed)
# So rb_onboard_power must be in the SAME UNITS as rb_power_zero!

print(f"\n=== KEY FINDING ===")
print(f"rb_power_zero is in PiB = {sim_data['rb_power_zero']:.1f}")
print(f"But we're passing rb_onboard_power in EiB = {rb_eib_median:.6f}")
print(f"That's {rb_pib_median:.2f} PiB, but the forecast_power_stats adds it directly to rb_power_zero (PiB)")
print(f"So passing EiB when PiB is expected means we're 1024x too small!")

print(f"\n=== VERIFICATION: PASS PiB INSTEAD ===")
rbp_pib_vec = jnp.ones(FORECAST_DAYS) * rb_pib_median

rb_forecast2, qa_forecast2 = power.forecast_power_stats(
    sim_data["rb_power_zero"], sim_data["qa_power_zero"],
    rbp_pib_vec,
    sim_data["rb_known_scheduled_expire_vec"],
    sim_data["qa_known_scheduled_expire_vec"],
    rr_vec, fpr_vec,
    SECTOR_DURATION, FORECAST_DAYS,
)

rb_total2 = np.array(rb_forecast2["total_power"])
qa_total2 = np.array(qa_forecast2["total_power"])
rb_onboard2 = np.array(rb_forecast2["onboarded_power"])
qa_onboard2 = np.array(qa_forecast2["onboarded_power"])

print(f"Input: rb_pib_median = {rb_pib_median:.6f}")
print(f"rb_forecast total_power[0]: {rb_total2[0]:.1f}")
print(f"rb_forecast total_power[-1]: {rb_total2[-1]:.1f}")
print(f"qa_forecast total_power[0]: {qa_total2[0]:.1f}")
print(f"qa_forecast total_power[-1]: {qa_total2[-1]:.1f}")
print(f"rb_forecast onboarded[0]: {rb_onboard2[0]:.6f}")
print(f"qa_forecast onboarded[0]: {qa_onboard2[0]:.6f}")

# Now check which produces continuous power trajectory
print(f"\n=== CONTINUITY: which aligns with historical? ===")
print(f"Historical total RBP last day: {hist_rb_eib[-1]*1024:.1f} PiB")
print(f"Forecast (EiB input) day 0 total: {rb_total[0]:.1f} PiB")
print(f"Forecast (PiB input) day 0 total: {rb_total2[0]:.1f} PiB")
print(f"Expected: ~{hist_rb_eib[-1]*1024:.0f} PiB")
