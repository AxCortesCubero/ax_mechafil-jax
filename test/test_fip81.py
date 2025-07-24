import os
import unittest
from datetime import date, timedelta
from dotenv import load_dotenv


#from jax import config
#config.update("jax_enable_x64", True)

import mechafil_jax.data as data
import mechafil_jax.supply as jax_supply
import mechafil_jax.constants as C
import mechafil_jax.sim as sim
import scenario_generator.utils as u

import numpy as np
import jax.numpy as jnp
import tqdm.auto as tqdm
import pickle
import json


class TestFip81(unittest.TestCase):
    def get_offline_data(self, start_date, current_date, end_date):
        # Load the .env file
        load_dotenv()
        # Get the path to the authentication file from the environment variable
        auth_file_path = os.getenv('SPACESCOPE_AUTH_PATH')
        
        if auth_file_path:
            # Load the token from the JSON file
            with open(auth_file_path, 'r') as f:
                auth_data = json.load(f)
        
            # Assuming the key in the JSON file is "auth_key" and it contains the Bearer token
            PUBLIC_AUTH_TOKEN = auth_data.get('auth_key')
            
            if PUBLIC_AUTH_TOKEN:
                pass
            else:
                raise Exception("The 'auth_key' is missing in the JSON file.")
        else:
            raise Exception("SPACESCOPE_AUTH_PATH environment variable is not set.")
        
        offline_data = data.get_simulation_data(PUBLIC_AUTH_TOKEN, start_date, current_date, end_date)

        _, hist_rbp = u.get_historical_daily_onboarded_power(current_date - timedelta(days=180), current_date)
        _, hist_rr = u.get_historical_renewal_rate(current_date - timedelta(days=180), current_date)
        _, hist_fpr = u.get_historical_filplus_rate(current_date - timedelta(days=180), current_date)
        smoothed_last_historical_rbp = float(np.median(hist_rbp[-30:]))
        smoothed_last_historical_rr = float(np.median(hist_rr[-30:]))
        smoothed_last_historical_fpr = float(np.median(hist_fpr[-30:]))

        result = (
            offline_data, smoothed_last_historical_rbp, smoothed_last_historical_rr,
            smoothed_last_historical_fpr, hist_rbp, hist_rr, hist_fpr
        )
        return result

    def test_fip81(self):

        # Load data from original fip81 implementation
        test_dir = os.path.dirname(__file__)  # This will get the directory of the current test file
        file_path = os.path.join(test_dir, 'results_original_branch.pkl')
        with open(file_path, 'rb') as f:
            results_original = pickle.load(f)
        
        # Setup correct date to match reference calculations
        current_date = date(2025, 7, 6) # WARNING: do not change this variable
        forecast_length_days = 365 * 5 # WARNING: do not change this variable
        
        # Setup
        mo_start = max(current_date.month - 1 % 12, 1)
        start_date = date(current_date.year, mo_start, 1)
        end_date = current_date + timedelta(days=forecast_length_days)

        #Get Data
        offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, *_ = self.get_offline_data(start_date, current_date, end_date)

        sector_duration_days = 540
        lock_target = 0.3
        rbp = jnp.ones(forecast_length_days) * smoothed_rbp
        rr = jnp.ones(forecast_length_days) * smoothed_rr
        fpr = jnp.ones(forecast_length_days) * smoothed_fpr

        # Run simulation
        simulation_results = sim.run_sim(
            rbp, rr, fpr, lock_target, start_date, current_date,
            forecast_length_days, sector_duration_days, offline_data,
            use_available_supply=False
        )

        for key in results_original[0]:  # Assuming results are stored as dictionaries
            original_result = results_original[0][key]
            simulation_result = simulation_results[key]
            print(f"Comparing key: {key}")
        
            self.assertTrue(
                np.allclose(original_result, simulation_result, rtol=1e-10, atol=1e-10), 
                f"Mismatch found in {key}!\nOriginal result: {original_result[:5]} ...\nSimulation result: {simulation_result[:5]} ..."
            )

if __name__ == '__main__':
    unittest.main()