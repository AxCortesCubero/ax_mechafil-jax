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
"""Quick smoke test to verify Spacescope API access."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from datetime import date, timedelta
import pystarboard.data as pdata
from pystarboard.data_spacescope import SpacescopeDataConnection

AUTH_TOKEN = "Bearer ghp_EviOPunZooyAagPPmftIsHfWarumaFOUdBUZ"
pdata.setup_spacescope(AUTH_TOKEN)
print("Auth setup done")

sd = date.today() - timedelta(days=10)
ed = date.today() - timedelta(days=3)
print(f"Fetching supply stats {sd} -> {ed} ...")
df = SpacescopeDataConnection.query_spacescope_supply_stats(sd, ed)
print(f"Got {len(df)} rows")
print(df[["date", "locked_fil", "circulating_fil"]].to_string())

print("\nTesting get_simulation_data (1mo window) ...")
import mechafil_jax.data as mdata
start = date.today() - timedelta(days=30)
current = date.today() - timedelta(days=3)
end = current + timedelta(days=30)
sim_data = mdata.get_simulation_data(AUTH_TOKEN, start, current, end)
print(f"locked_fil_zero = {sim_data['locked_fil_zero']}")
print(f"locked_reward_zero = {sim_data.get('locked_reward_zero', 'NOT PRESENT')}")
print(f"circ_supply_zero = {sim_data['circ_supply_zero']}")
print("\nSmoke test passed!")
