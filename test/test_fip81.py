import os
import unittest
from datetime import date, timedelta
from dotenv import load_dotenv


from jax import config
config.update("jax_enable_x64", True)

import pystarboard.data as data
import mechafil_jax.supply as jax_supply
import mechafil_jax.constants as C

import numpy as np
import jax.numpy as jnp
import tqdm.auto as tqdm

class TestSupply(unittest.TestCase):
    def test_forecast_supply(self):
        # Load the .env file
        load_dotenv()
        auth_file_path = os.getenv('SPACESCOPE_AUTH_PATH')
        
        # Setup data access
        if auth_file_path:
            data.setup_spacescope(auth_file_path)
        else:
            raise Exception("SPACESCOPE_AUTH_PATH environment variable is not set")

        forecast_length = 360*2
        start_date = date(2021, 3, 16)
        current_date = date.today() - timedelta(days=2)
        end_date = current_date + timedelta(days=forecast_length)

        # Get sector scheduled expirations
        res = data.get_sector_expiration_stats(start_date, current_date, end_date)
        rb_known_scheduled_expire_vec = res[0]
        qa_known_scheduled_expire_vec = res[1]
        known_scheduled_pledge_release_full_vec = res[2]
        # Get daily stats
        fil_stats_df = data.get_historical_network_stats(start_date, current_date, end_date)
        current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
        rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
        qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0

        # consider sweeping these to build confidence in JAX implementation
        rr = 0.3
        fpr = 0.3
        duration = 360
        rbp = 3
        lock_target = 0.3

        vest_df
        mint_df

        # convert mint + vest into dictionaries
        vest_dict = {
            "days": vest_df['days'].values,
            'total_vest': np.asarray(vest_df['total_vest'].values),
        }
        mint_dict = {
            'days': mint_df['days'].values,
            'network_RBP_EIB': np.asarray(mint_df['network_RBP'].values) / C.EIB,
            'network_QAP_EIB': np.asarray(mint_df['network_QAP'].values) / C.EIB,
            'day_onboarded_power_QAP_PIB': np.asarray(mint_df['day_onboarded_power_QAP'].values / C.PIB),
            'day_renewed_power_QAP_PIB': np.asarray(mint_df['day_renewed_power_QAP'].values / C.PIB),
            'cum_simple_reward': np.asarray(mint_df['cum_simple_reward'].values),
            'network_baseline_EIB': np.asarray(mint_df['network_baseline'].values) / C.EIB,
            'capped_power_EIB': np.asarray(mint_df['capped_power'].values) / C.EIB,
            'cum_capped_power_EIB': np.asarray(mint_df['cum_capped_power'].values) / C.EIB,
            'network_time': np.asarray(mint_df['network_time'].values),
            'cum_baseline_reward': np.asarray(mint_df['cum_baseline_reward'].values),
            'cum_network_reward': np.asarray(mint_df['cum_network_reward'].values),
            'day_network_reward': np.asarray(mint_df['day_network_reward'].values),
        }
        historical_target_lock = jnp.ones(len(past_renewal_rate_vec)) * 0.3
        lock_target = jnp.ones(forecast_length) * lock_target
        full_lock_target_vec = jnp.concatenate(
            [historical_target_lock, lock_target]
        )
        # these are the default values
        gamma_vec = jnp.ones(len(full_lock_target_vec)) * 1.0
        cil_jax = jax_supply.forecast_circulating_supply(
            np.datetime64(start_date),
            np.datetime64(current_date),
            np.datetime64(end_date),
            circ_supply_zero,
            locked_fil_zero,
            daily_burnt_fil,
            duration,
            renewal_rate_vec,
            jnp.asarray(burnt_fil_vec),
            vest_dict,
            mint_dict,
            jnp.asarray(known_scheduled_pledge_release_full_vec),
            lock_target=full_lock_target_vec,
            gamma=gamma_vec,
        )
        keys = ['circ_supply', 'network_gas_burn', 'day_locked_pledge', 'day_renewed_pledge',
                'network_locked_pledge', 'network_locked', 'network_locked_reward', 'disbursed_reserve']
        for k in keys:
            self.assertTrue(np.allclose(cil_df[k].values, np.asarray(cil_jax[k])), k)

if __name__ == '__main__':
    unittest.main()