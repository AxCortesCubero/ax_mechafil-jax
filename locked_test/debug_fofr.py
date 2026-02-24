#!/usr/bin/env -S uv run --python 3.12
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jax", "jaxlib", "numpy", "matplotlib", "requests", "pandas",
#   "pystarboard @ git+https://github.com/CELtd/pystarboard.git",
#   "mechafil-jax @ file:///home/kiran/code/cel/mechafil-jax",
# ]
# ///
import sys; sys.stdout.reconfigure(line_buffering=True)
from datetime import date, timedelta
import numpy as np, jax.numpy as jnp
import mechafil_jax.data as mdata, mechafil_jax.sim as sim

AUTH = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
cd = date.today() - timedelta(days=3)
sd = cd - timedelta(days=180)
ed = cd + timedelta(days=365)
d = mdata.get_simulation_data(AUTH, sd, cd, ed)
rb = d["historical_onboarded_rb_power_pib"][-30:]
qa = d["historical_onboarded_qa_power_pib"][-30:]
rr = d["historical_renewal_rate"][-30:]
rbp = float(np.median(rb)/1024)
rrv = float(np.median(rr))
rat = np.where(rb>0, qa/rb, 1.0)
fpr = float(np.median(np.clip((rat-1)/9,0,1)))
r = sim.run_sim(jnp.ones(365)*rbp, jnp.ones(365)*rrv, jnp.ones(365)*fpr,
                0.3, sd, cd, 365, 360, d)

total_days = (ed - sd).days
roi = np.array(r['1y_sector_roi'])
rps = np.array(r['1y_return_per_sector'])
dppq = np.array(r['day_pledge_per_QAP'])

print(f"total_days: {total_days}, split: 180")
print(f"roi shape: {roi.shape}")
print(f"rps shape: {rps.shape}")
print(f"dppq shape: {dppq.shape}")
print(f"roi[:5]: {roi[:5]}")
print(f"roi[175:185]: {roi[175:185]}")
print(f"roi[-5:]: {roi[-5:]}")
print(f"any nan roi: {np.any(np.isnan(roi))}")
print(f"any nan rps: {np.any(np.isnan(rps))}")
print(f"any nan dppq: {np.any(np.isnan(dppq))}")
