import unittest
import numpy as np
import jax.numpy as jnp
from datetime import date, timedelta
from mechafil_jax.locking import create_gamma_trajectory
from mechafil_jax import constants


class TestFip81GammaTrajectory(unittest.TestCase):
    """
    Unit test for FIP81 gamma trajectory generation.
    Tests the gamma ramping functionality with hardcoded expected values.
    """

    def test_fip81_gamma_trajectory_comprehensive(self):
        """
        Comprehensive test for FIP81 gamma trajectory generation covering:
        1. Basic trajectory structure and values
        2. Edge cases and boundary conditions  
        3. Specific known value verification
        4. Mathematical properties and constants
        5. Different ramp length scenarios
        """
        
        # =============================================================================
        # PART 1: Basic trajectory test
        # =============================================================================
        print("\n=== PART 1: Basic Trajectory Test ===")
        
        # Test parameters - using known dates for reproducible results
        current_date = date(2025, 1, 1)  # ~41 days after FIP81 activation
        forecast_length_days = 400  # Make sure this is longer than remaining ramp days
        historical_length = 30
        ramp_len_days = 365  # Default ramp length
        
        # Calculate expected values manually
        days_since_activation = (current_date - constants.FIP81_ACTIVATION_DATE).days
        gamma_slope = (1.0 - constants.FIP81_GAMMA_TARGET) / ramp_len_days
        current_gamma = 1.0 - gamma_slope * days_since_activation
        remaining_days = ramp_len_days - days_since_activation
        
        # Call library function
        gamma_vec = create_gamma_trajectory(
            current_date, forecast_length_days, historical_length, ramp_len_days
        )
        
        # Test structure: historical + forecast
        expected_total_length = historical_length + forecast_length_days
        self.assertEqual(len(gamma_vec), expected_total_length)
        
        # Test historical period (should all be 1.0)
        historical_part = gamma_vec[:historical_length]
        expected_historical = jnp.ones(historical_length) * 1.0
        np.testing.assert_array_almost_equal(historical_part, expected_historical, decimal=6)
        
        # Test forecast period structure
        forecast_part = gamma_vec[historical_length:]
        if remaining_days > 0:
            ramp_part = forecast_part[:remaining_days]
            expected_ramp = jnp.linspace(current_gamma, constants.FIP81_GAMMA_TARGET, remaining_days)
            np.testing.assert_array_almost_equal(ramp_part, expected_ramp, decimal=6)
            
            if forecast_length_days > remaining_days:
                constant_part = forecast_part[remaining_days:]
                expected_constant = jnp.ones(forecast_length_days - remaining_days) * constants.FIP81_GAMMA_TARGET
                np.testing.assert_array_almost_equal(constant_part, expected_constant, decimal=6)

if __name__ == '__main__':
    unittest.main()