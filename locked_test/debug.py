#!/usr/bin/env -S uv run
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
#   "mechafil-jax @ file:///home/luca/programmi/cel/mechafil/programs/mechafil-jax",
# ]
# ///
"""
mechafil-jax debug simulation script.

Fetches historical Filecoin network data, derives scenario parameters from
the last 30 days of actuals, runs a forward simulation, and plots:
  - Network Locked FIL
  - Circulating Supply

Historical data is shown as a solid line; the simulation forecast continues
as a dashed line of the same colour, joined at the current date.

Run with:  uv run debug.py
"""

from datetime import date, timedelta
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import mechafil_jax.data as mdata
import mechafil_jax.sim as sim
from pystarboard.data_spacescope import SpacescopeDataConnection

# ── CONFIG ────────────────────────────────────────────────────────────────────
AUTH_TOKEN      = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
START_DATE      = date(2022, 1, 1)
CURRENT_DATE    = date.today() - timedelta(days=3)
FORECAST_LENGTH = 365 * 3   # days into the future
SECTOR_DURATION = 360  # Rverage sector duration in days
LOCK_TARGET     = 0.3        # target lock ratio
LOOKBACK_DAYS   = 30         # days to average for RBP / RR / FPR
# ─────────────────────────────────────────────────────────────────────────────

end_date = CURRENT_DATE + timedelta(days=FORECAST_LENGTH)

print(f"Dates: history {START_DATE} → {CURRENT_DATE}, forecast → {end_date}")
print("Fetching simulation data from Spacescope …")
sim_data = mdata.get_simulation_data(AUTH_TOKEN, START_DATE, CURRENT_DATE, end_date)
print("  done.")

# ── Derive 30-day average scenario parameters from historical tail ─────────────
rb_pib  = sim_data["historical_onboarded_rb_power_pib"][-LOOKBACK_DAYS:]
qa_pib  = sim_data["historical_onboarded_qa_power_pib"][-LOOKBACK_DAYS:]
rr_hist = sim_data["historical_renewal_rate"][-LOOKBACK_DAYS:]

rbp_eib = float(np.median(rb_pib) / 1024.0)                            # PiB/day → EiB/day
rr_val  = float(np.median(rr_hist))
# FIL+ rate: from QA/RB onboarding ratio (standard=1x, FIL+=10x)
ratio   = np.where(rb_pib > 0, qa_pib / rb_pib, 1.0)
fpr_val = float(np.median(np.clip((ratio - 1.0) / 9.0, 0.0, 1.0)))

print(f"\n30-day average scenario parameters:")
print(f"  RBP : {rbp_eib:.4f} EiB/day")
print(f"  RR  : {rr_val:.4f}")
print(f"  FPR : {fpr_val:.4f}")

# ── Run simulation ────────────────────────────────────────────────────────────
rbp_vec = jnp.ones(FORECAST_LENGTH) * rbp_eib
rr_vec  = jnp.ones(FORECAST_LENGTH) * rr_val
fpr_vec = jnp.ones(FORECAST_LENGTH) * fpr_val

print("\nRunning simulation …")
results = sim.run_sim(
    rbp_vec, rr_vec, fpr_vec,
    LOCK_TARGET,
    START_DATE, CURRENT_DATE, FORECAST_LENGTH, SECTOR_DURATION,
    sim_data,
)
print("  done.")


# ── On-chain historical data (Spacescope) ────────────────────────────────────
print("Fetching on-chain supply data …")
supply_df = SpacescopeDataConnection.query_spacescope_supply_stats(START_DATE, CURRENT_DATE)
supply_df = supply_df.sort_values("date")
onchain_dates  = [date.fromisoformat(str(d)) for d in supply_df["date"]]
onchain_circ   = supply_df["circulating_fil"].astype(float).values
onchain_locked = supply_df["locked_fil"].astype(float).values
print(f"  got {len(onchain_dates)} days")

# ── Time axis & split index ───────────────────────────────────────────────────
total_days = (end_date - START_DATE).days
t     = [START_DATE + timedelta(days=i) for i in range(total_days)]
split = (CURRENT_DATE - START_DATE).days   # index where forecast begins

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle(
    f"mechafil-jax  |  {CURRENT_DATE}  |  "
    f"RBP={rbp_eib:.3f} EiB/d  RR={rr_val:.3f}  FPR={fpr_val:.3f}  "
    f"(30-day avg)",
    fontsize=11,
)

ONCHAIN_KW  = dict(color="black",      linewidth=1.5, label="On-chain (Spacescope)")
HIST_KW     = dict(color="steelblue",  linewidth=2.0, linestyle="-",  label="mechafil-jax historical")
FCST_KW     = dict(color="steelblue",  linewidth=2.0, linestyle="--", label="mechafil-jax forecast")
VLINE_KW    = dict(color="dimgray",    linewidth=1.0, linestyle=":",  alpha=0.7)

panels = [
    (axes[0], "network_locked", onchain_locked, "Network Locked FIL"),
    (axes[1], "circ_supply",    onchain_circ,   "Circulating Supply"),
]

for ax, key, y_onchain, title in panels:
    y_fix = np.array(results[key])     / 1e6

    ax.plot(onchain_dates,  y_onchain / 1e6,     **ONCHAIN_KW)
    ax.plot(t[:split + 1], y_fix[:split + 1],    **HIST_KW)
    ax.plot(t[split:],     y_fix[split:],         **FCST_KW)

    ax.axvline(CURRENT_DATE, **VLINE_KW)
    ax.text(
        CURRENT_DATE, 0.97, f"  {CURRENT_DATE}",
        transform=ax.get_xaxis_transform(),
        fontsize=8, color="dimgray", va="top",
    )

    ax.set_ylabel("M-FIL")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

out_path = "sim_debug.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
plt.show()
