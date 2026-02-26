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
Diagnostic: does onboarding actually affect locked FIL?

Compare 3 scenarios that ONLY vary RBP (same RR):
  A. Current RBP (0.99 PiB/d)
  B. 10x RBP (9.9 PiB/d)
  C. Zero RBP (0.001 PiB/d, ~zero)

If locked FIL is identical across all three, the insensitivity is real.
We also dump the intermediate power & pledge values to understand WHY.
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
SIM_EXTRA = 0
SECTOR_DURATION = 360
LOCK_TARGET = 0.3

START_DATE = CURRENT_DATE - timedelta(days=HISTORY_DAYS)
SIM_END = CURRENT_DATE + timedelta(days=FORECAST_DAYS + SIM_EXTRA)
SIM_FCST = FORECAST_DAYS + SIM_EXTRA

print(f"History:  {START_DATE} -> {CURRENT_DATE}")
print(f"Forecast: {CURRENT_DATE} -> {SIM_END}")

# Fetch data
print("\nFetching data ...")
sim_data = mdata.get_simulation_data(AUTH, START_DATE, CURRENT_DATE, SIM_END)

# Derive baseline inputs
rb_pib = sim_data["historical_onboarded_rb_power_pib"][-30:]
qa_pib = sim_data["historical_onboarded_qa_power_pib"][-30:]
rr_hist = sim_data["historical_renewal_rate"][-30:]

rbp_pib_day = float(np.median(rb_pib))
rbp_eib_day = rbp_pib_day / 1024.0
rr_val = float(np.median(rr_hist))
ratio = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

print(f"  RBP: {rbp_pib_day:.2f} PiB/d ({rbp_eib_day:.6f} EiB/d)")
print(f"  RR:  {rr_val*100:.1f}%")
print(f"  FPR: {fpr_val*100:.1f}%")

qa_mult = 1.0 + 9.0 * fpr_val
print(f"  QA mult: {qa_mult:.1f}x")

# Common inputs
rr_vec = jnp.ones(SIM_FCST) * rr_val
fpr_vec = jnp.ones(SIM_FCST) * fpr_val

# Three scenarios: vary ONLY RBP
scenarios = {
    "Current (1x)": rbp_eib_day,
    "10x RBP":      rbp_eib_day * 10,
    "Zero RBP":     rbp_eib_day * 0.001,
}

N = HISTORY_DAYS + FORECAST_DAYS  # total display days
split = HISTORY_DAYS

for name, rbp in scenarios.items():
    rbp_vec = jnp.ones(SIM_FCST) * rbp
    results = sim.run_sim(
        rbp_vec, rr_vec, fpr_vec,
        LOCK_TARGET, START_DATE, CURRENT_DATE, SIM_FCST, SECTOR_DURATION,
        sim_data,
    )

    # Extract key metrics at forecast end (1Y out)
    idx = N - 1
    qap = float(np.array(results["qa_total_power_eib"])[idx])
    rbp_tot = float(np.array(results["rb_total_power_eib"])[idx])
    locked = float(np.array(results["network_locked"])[idx])
    locked_pledge = float(np.array(results["network_locked_pledge"])[idx])
    locked_reward = float(np.array(results["network_locked_reward"])[idx])
    circ = float(np.array(results["circ_supply"])[idx])
    minting = float(np.array(results["day_network_reward"])[idx])

    # Daily onboarded QAP at end
    day_ob_qa = float(np.array(results["qa_day_onboarded_power_pib"])[idx])
    day_rn_qa = float(np.array(results["qa_day_renewed_power_pib"])[idx])

    # Pledge calculation components
    day_pledge = float(np.array(results["day_locked_pledge"])[idx])
    day_renewed_pledge = float(np.array(results["day_renewed_pledge"])[idx])

    # day_pledge_per_QAP
    dppq = float(np.array(results["day_pledge_per_QAP"])[idx])

    print(f"\n{'='*60}")
    print(f"Scenario: {name}  (RBP = {rbp*1024:.3f} PiB/d)")
    print(f"{'='*60}")
    print(f"  QAP          : {qap:.3f} EiB   ({qap*1024:.1f} PiB)")
    print(f"  RBP total    : {rbp_tot:.3f} EiB")
    print(f"  Minting rate : {minting:.4f} FIL/day")
    print(f"  Circ supply  : {circ/1e6:.1f}M FIL")
    print(f"  Day onboard QA: {day_ob_qa:.2f} PiB/d")
    print(f"  Day renewed QA: {day_rn_qa:.2f} PiB/d")
    print(f"  Day pledge new: {day_pledge:.4f} FIL")
    print(f"  Day pledge ren: {day_renewed_pledge:.4f} FIL")
    print(f"  Pledge/sector : {dppq:.6f} FIL")
    print(f"  --")
    print(f"  Locked total : {locked/1e6:.2f}M FIL")
    print(f"  Locked pledge: {locked_pledge/1e6:.2f}M FIL")
    print(f"  Locked reward: {locked_reward/1e6:.2f}M FIL")

    # Analytical check: what fraction of network is daily onboarding?
    if qap > 0:
        frac = (day_ob_qa / 1024.0) / qap
        print(f"  Daily ob/QAP : {frac*100:.4f}%")
        # Storage pledge component for new onboarding
        sp_new = 20 * minting * (day_ob_qa / (qap * 1024.0))
        # Consensus pledge component
        cp_new = LOCK_TARGET * circ * (day_ob_qa / (qap * 1024.0))
        total_new = sp_new + cp_new
        print(f"  Est. pledge from new onboard: {total_new:.4f} FIL/day")
        print(f"    storage component: {sp_new:.4f} FIL/day")
        print(f"    consensus component: {cp_new:.4f} FIL/day")
