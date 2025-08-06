import unittest
import jax.numpy as jnp
from mechafil_jax.locking import compute_new_pledge_for_added_power
from mechafil_jax import constants


class TestFip81PledgeCalculation(unittest.TestCase):
    """
    Unit test for FIP81 pledge calculation.
    Tests the simplified arithmetic weighting approach vs library implementation.
    """
    
    def test_fip81_pledge_calculation_comprehensive(self):
        """
        Simplified FIP81 pledge calculation test focusing on arithmetic weighting formula.
        
        In FIP81, the consensus pledge uses only arithmetic weighting:
        consensus_pledge = (1 - gamma) * simple_consensus_pledge + gamma * baseline_consensus_pledge
        """
        
        # Test parameters for gamma relationship testing
        day_network_reward_gamma = 80000.0
        prev_circ_supply_gamma = 450000000.0
        day_added_qa_power_pib_gamma = 6.0
        total_qa_power_pib_gamma = 18000.0
        baseline_power_pib_gamma = 16000.0
        lock_target_gamma = 0.28
        
        # Test intermediate gamma values to ensure arithmetic weighting
        test_gammas = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
        
        for test_gamma in test_gammas:
            lib_result = compute_new_pledge_for_added_power(
                day_network_reward_gamma, prev_circ_supply_gamma, day_added_qa_power_pib_gamma,
                total_qa_power_pib_gamma, baseline_power_pib_gamma, lock_target_gamma, gamma=test_gamma
            )
            
            # Manual calculation to verify arithmetic weighting
            storage = 20.0 * day_network_reward_gamma * (day_added_qa_power_pib_gamma / total_qa_power_pib_gamma)
            simple = max(lock_target_gamma * prev_circ_supply_gamma * (day_added_qa_power_pib_gamma / total_qa_power_pib_gamma), 0)
            baseline = max(lock_target_gamma * prev_circ_supply_gamma * (day_added_qa_power_pib_gamma / max(total_qa_power_pib_gamma, baseline_power_pib_gamma)), 0)
            consensus = (1 - test_gamma) * simple + test_gamma * baseline
            expected = min(day_added_qa_power_pib_gamma * (constants.PIB / constants.GIB), storage + consensus)
            
            self.assertAlmostEqual(float(lib_result), expected, delta=0.01)


if __name__ == '__main__':
    unittest.main()