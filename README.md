# MechaFil-JAX

A JAX-based implementation of the Filecoin economic model simulator, providing high-performance simulations of the Filecoin network's economics including power growth, minting, vesting, and supply dynamics.

## Overview

MechaFil-JAX is a Python package that simulates the Filecoin network's economic behavior using JAX for accelerated numerical computations. It models key aspects of the Filecoin protocol including:

- **Power Dynamics**: Raw Byte Power (RBP) and Quality Adjusted Power (QAP) calculations
- **Token Minting**: Network reward calculations based on storage power and baseline
- **Token Supply**: Circulating supply management including locked tokens and vesting schedules
- **FIP-81 Support**: Implementation of FIP-81 consensus pledge calculations with gamma smoothing

## Features

- **High Performance**: JAX-accelerated computations for fast simulations
- **Comprehensive Modeling**: Complete economic model covering power, minting, vesting, and supply
- **FIP-81 Compliance**: Support for FIP-81 consensus pledge calculations
- **Historical Data Integration**: Works with real Filecoin network data via Spacescope API
- **Extensive Testing**: Unit tests covering critical functionality
- **Jupyter Notebooks**: Example simulations and debugging tools

## Installation

### Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f environment.yaml
conda activate cel

# Install the package
pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from datetime import date, timedelta
import mechafil_jax.sim as sim
from mechafil_jax.data import get_offline_data

# Set simulation parameters
print('Loading historical data ...')
current_date = date(2025, 7, 6)
mo_start = max(current_date.month - 1 % 12, 1)
start_date = date(current_date.year, mo_start, 1)
forecast_length_days = 365 * 5
end_date = current_date + timedelta(days=forecast_length_days)

# Load historical data and get smoothed parameters
offline_data, smoothed_rbp, smoothed_rr, smoothed_fpr, *_ = get_offline_data(
    start_date, current_date, end_date
)

sector_duration_days = 540
lock_target = 0.3

print(f'Current smoothed last historical RBP: {smoothed_rbp}')
print('Running simulations ...')

# Use historical smoothed values for forecast
rbp = jnp.ones(forecast_length_days) * smoothed_rbp
rr = jnp.ones(forecast_length_days) * smoothed_rr
fpr = jnp.ones(forecast_length_days) * smoothed_fpr

# Run simulation
simulation_results = sim.run_sim(
    rbp, rr, fpr, lock_target, start_date, current_date,
    forecast_length_days, sector_duration_days, offline_data,
    use_available_supply=False
)

# Access results
print(f"Final circulating supply: {simulation_results['circ_supply'][-1]:.2f} FIL")
print(f"Final network locked: {simulation_results['network_locked'][-1]:.2f} FIL")
```

## Key Modules

- **`sim.py`**: Main simulation orchestrator
- **`power.py`**: Storage power calculations (RBP/QAP)
- **`minting.py`**: Token minting and reward calculations
- **`supply.py`**: Token supply and circulation dynamics
- **`vesting.py`**: Token vesting schedule calculations
- **`locking.py`**: FIP-81 gamma trajectory and pledge calculations
- **`data.py`**: Historical data loading and preprocessing

## Testing

Run the test suite:

```bash
# Using the provided script
./run_tests.sh

# Or directly with pytest
python -m pytest test/
```

## Examples

See the `notebooks/` directory for detailed examples:

- **`sim_example.ipynb`**: Basic simulation example comparing NumPy and JAX implementations
- **`time_mechafil_jax.ipynb`**: Performance benchmarking
- **Module debugging notebooks**: Individual component testing and validation

## Dependencies

- **JAX/JAXLib**: High-performance numerical computing
- **NumPy**: Numerical arrays and operations
- **SciPy**: Scientific computing utilities
- **Matplotlib**: Plotting and visualization
- **PyStarboard**: Filecoin data access utilities

## Development

The project follows standard Python development practices:

- Code organized in modules under `mechafil_jax/`
- Comprehensive test suite in `test/`
- Jupyter notebooks for examples and debugging
- Conda environment for dependency management

## Attribution

This project is an adaptation by the Crypto Econ Lab (CEL) for Filecoin network analysis and modeling.