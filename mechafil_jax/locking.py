from typing import Union
import datetime

import numpy as np
from numpy.typing import NDArray

import jax
import jax.lax as lax
import jax.numpy as jnp

from mechafil_jax import constants

"""
A port of the locking module in mechafil to JAX.
"""

# Block reward collateral
@jax.jit
def compute_day_locked_rewards(day_network_reward: float) -> float:
    return 0.75 * day_network_reward

@jax.jit
def compute_day_reward_release(prev_network_locked_reward: float) -> float:
    return prev_network_locked_reward / 180.0


# Initial pledge collateral
@jax.jit
def compute_day_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power_pib: float,
    day_renewed_qa_power_pib: float,
    total_qa_power_eib: float,
    baseline_power_eib: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float,
) -> float:
    # convert powers to PIB
    total_qa_power_pib = total_qa_power_eib * 1024.0
    baseline_power_pib = baseline_power_eib * 1024.0

    onboards_delta = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma
    )
    renews_delta = compute_renewals_delta_pledge(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        renewal_rate,
        scheduled_pledge_release,
        lock_target,
        gamma
    )
    return onboards_delta + renews_delta

@jax.jit
def compute_day_locked_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power_pib: float,
    day_renewed_qa_power_pib: float,
    total_qa_power_eib: float,
    baseline_power_eib: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float,
) -> float:
    # convert powers to PIB
    total_qa_power_pib = total_qa_power_eib * 1024.0
    baseline_power_pib = baseline_power_eib * 1024.0

    # print('jax', day_network_reward, prev_circ_supply, day_onboarded_qa_power_pib, total_qa_power_pib, baseline_power_pib, lock_target)
    # Total locked from new onboards
    onboards_locked = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma
    )
    # print('jax', onboards_locked)
    # Total locked from renewals
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma
    )
    renews_locked = jnp.maximum(original_pledge, new_pledge)
    # Total locked pledge
    locked = onboards_locked + renews_locked
    # print('jax', locked, renews_locked)
    return locked, renews_locked

@jax.jit
def compute_renewals_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float,
) -> float:
    # Delta from sectors expiring
    expire_delta = -(1 - renewal_rate) * scheduled_pledge_release
    # Delta from sector renewing
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
        gamma
    )
    renew_delta = jnp.maximum(0.0, new_pledge - original_pledge)
    # Delta for all scheduled sectors
    delta = expire_delta + renew_delta
    return delta

@jax.jit
def compute_new_pledge_for_added_power(
    day_network_reward: float,
    prev_circ_supply: float,
    day_added_qa_power_pib: float,
    total_qa_power_pib: float,
    baseline_power_pib: float,
    lock_target: float,
    gamma: float,
) -> float:
    # storage collateral
    storage_pledge = 20.0 * day_network_reward * (day_added_qa_power_pib / total_qa_power_pib)
    # consensus collateral
    # normalized_qap_growth = day_added_qa_power_pib / jnp.maximum(total_qa_power_pib, baseline_power_pib)
    # consensus_pledge = jnp.maximum(lock_target * prev_circ_supply * normalized_qap_growth, 0)

    # Compute Simple and Baseline Pledge 
    simple_consensus_pledge = lock_target * prev_circ_supply * (day_added_qa_power_pib/total_qa_power_pib)
    simple_consensus_pledge = jnp.maximum(simple_consensus_pledge, 0)

    normalized_qap_growth = day_added_qa_power_pib / jnp.maximum(total_qa_power_pib, baseline_power_pib) # Fraction for Consensus Pledge Component
    baseline_consensus_pledge = lock_target * prev_circ_supply * normalized_qap_growth
    baseline_consensus_pledge = jnp.maximum(baseline_consensus_pledge, 0)

    # Compute Simple and Baseline Pledge Contributions to Consensus Pledge
    consensus_pledge = weighted_consensus(gamma, simple_consensus_pledge, baseline_consensus_pledge)

    # total added pledge
    added_pledge = storage_pledge + consensus_pledge

    pledge_cap = day_added_qa_power_pib * (constants.PIB / constants.GIB)  # The # of bytes in a GiB (Gibibyte)
    return jnp.minimum(pledge_cap, added_pledge)

@jax.jit
def weighted_consensus(w, simple_consensus_pledge, baseline_consensus_pledge):
    return (1 - w) * simple_consensus_pledge + w * baseline_consensus_pledge

@jax.jit
def have_known_pledge_info(arggs):
    known_scheduled_pledge_release_vec, day_i = arggs
    return known_scheduled_pledge_release_vec[day_i]

@jax.jit
def dont_have_known_pledge_info(arggs):
    return 0.0

@jax.jit
def have_modeled_pledge_info(arggs):
    day_pledge_locked_vec, day_i, duration = arggs
    return day_pledge_locked_vec[day_i - duration]

@jax.jit
def dont_have_modeled_pledge_info(arggs):
    return 0.0

@jax.jit
def get_day_schedule_pledge_release(
    day_i,
    current_day_i,
    day_pledge_locked_vec: Union[jnp.array, NDArray],
    known_scheduled_pledge_release_vec: np.array,
    duration: int,
) -> float:
    
    # scheduled releases coming from known active sectors
    pred = day_i > len(known_scheduled_pledge_release_vec) - 1
    known_day_release = lax.cond(
        pred, 
        dont_have_known_pledge_info,
        have_known_pledge_info, 
        (known_scheduled_pledge_release_vec, day_i)
     )
    
    pred = day_i - duration >= current_day_i
    model_day_release = lax.cond(
        pred,
        have_modeled_pledge_info,
        dont_have_modeled_pledge_info,
        (day_pledge_locked_vec, day_i, duration)
    )
    
    # Total pledge schedule releases
    day_pledge_schedules_release = known_day_release + model_day_release
    return day_pledge_schedules_release


# Creates the trajectory of the gamma smoothing function for FIP-81 activation
# TODO: lock target, shall I include it here?
def create_gamma_trajectory(
    current_date: np.datetime64,
    forecast_length_days: int,
    historical_length: int,
    ramp_len_days: int = 365,
):
    activation_date = np.datetime64(constants.FIP81_ACTIVATION_DATE, "D")
    current_date = np.datetime64(current_date, "D")
    
    # Calculate start_date from current_date and historical_length
    start_date = current_date - np.timedelta64(historical_length, "D")
    
    # Ramp slope
    gamma_slope = (1.0 - constants.FIP81_GAMMA_TARGET) / ramp_len_days

    # Days since activation (can be negative if before activation)
    days_since_activation = int((current_date - activation_date) / np.timedelta64(1, "D"))

    # Historical gamma - account for FIP-81 activation within historical period
    historical_gamma_list = []
    for i in range(historical_length):
        day_date = start_date + np.timedelta64(i, "D")
        days_since_activation_hist = int((day_date - activation_date) / np.timedelta64(1, "D"))
        
        if days_since_activation_hist < 0:
            gamma_val = 1.0
        else:
            gamma_val = max(1.0 - gamma_slope * days_since_activation_hist, constants.FIP81_GAMMA_TARGET)
        
        historical_gamma_list.append(gamma_val)
    
    historical_gamma = jnp.array(historical_gamma_list)

    # If still before activation → gamma stays at 1.0
    if days_since_activation < 0:
        gamma_trajectory = jnp.ones(forecast_length_days)
    else:
        current_gamma = max(1.0 - gamma_slope * days_since_activation, constants.FIP81_GAMMA_TARGET)

        # Remaining ramp days (non-negative)
        remaining_days = max(ramp_len_days - days_since_activation, 0)

        if remaining_days > 0:
            v1 = jnp.linspace(current_gamma, constants.FIP81_GAMMA_TARGET, remaining_days)
            v2 = jnp.ones(forecast_length_days - remaining_days) * constants.FIP81_GAMMA_TARGET
            gamma_trajectory = jnp.concatenate([v1, v2])
        else:
            # Already past ramp end
            gamma_trajectory = jnp.ones(forecast_length_days) * constants.FIP81_GAMMA_TARGET

    gamma_vec = jnp.concatenate([historical_gamma, gamma_trajectory])
    return gamma_vec
