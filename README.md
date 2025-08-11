# MechaFil-JAX

A JAX-based implementation of the Filecoin economic model simulator, providing high-performance simulations of the Filecoin network's economics including power growth, minting, vesting, and supply dynamics.

## Overview

MechaFil-JAX is a Python package that simulates the Filecoin network's economic behavior using JAX for accelerated numerical computations. It models key aspects of the Filecoin protocol including:

- **Power Dynamics**: Raw Byte Power (RBP) and Quality Adjusted Power (QAP) calculations
- **Token Minting**: Network reward calculations based on storage power and baseline
- **Token Supply**: Circulating supply management including locked tokens and vesting schedules

## Key Modules

- **`sim.py`**: Main simulation orchestrator
- **`power.py`**: Storage power calculations (RBP/QAP)
- **`minting.py`**: Token minting and reward calculations
- **`supply.py`**: Token supply and circulation dynamics
- **`vesting.py`**: Token vesting schedule calculations
- **`locking.py`**: Token pledge calculations
- **`data.py`**: Historical data loading and preprocessing

## Examples

See the `notebooks/` directory for detailed examples:

- **`sim_example.ipynb`**: Basic simulation example comparing NumPy and JAX implementations
- **`time_mechafil_jax.ipynb`**: Performance benchmarking
