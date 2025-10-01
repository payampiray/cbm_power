# cbm_power

Computational Behavioral/Brain Modeling (CBM) Library for Power Analysis and Sample-Size Optimization for Computational Studies.

## Overview

`cbm_power` is a Python package for estimating statistical power and optimizing sample sizes in computational modeling studies.  
It provides tools for simulation-based power estimation and automated sample-size optimization. It implements the method described in this study:
Piray Payam, Addressing low statistical power in computational modeling studies in psychology and neuroscience, 2025, Nature Human Behaviour.

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/cbm_power.git
cd cbm_power
pip install -e .
```

Dependencies include:
- numpy
- ax-platform for Bayesian optimization

## Usage

```python
from cbm_power import Power, SampleSize, Config

# Run power estimation
pwr = Power()
power, _ = pwr.compute_power(
    num_participants=410,
    num_models=10
)

print("Estimated power:", power)

# Optimize sample size
cfg = Config(num_models=10)  # Create a config first
ss = SampleSize(cfg)
best_sample_size = ss.compute_sample_size()  # this saves a JSON and a log file
print("Optimized sample size:", best_sample_size)
```

## Repository structure

```
cbm_power/
├── cbm_power/
│   ├── __init__.py        # Exposes Power, SampleSize, Config
│   ├── power.py           # Power estimation logic
│   ├── distributions.py   # Distribution utilities
│   ├── sample_size.py     # Sample size optimization
│   ├── config.py          # Config dataclass used in SampleSize
│   └── loggers.py         # Loggers used in SampleSize
├── docs/
│   ├── manual.ipynb       # Detailed usage guide & troubleshooting
├── README.md
├── .gitignore
├── license
└── requirements.txt

```

## Output

Running sample-size optimization automatically saves:
- **JSON file** with results
- **Log file** with optimization trace

Both are stored in the current directory.

## Documentation

A detailed manual (Jupyter Notebook) explaining the method, background, and step-by-step usage of `cbm_power` is available in the [`docs/`](docs/) directory.  
It includes:

- Key conceptual points about power in computational modeling studies
- Explanation of the multi-stage optimization method
- Worked examples with the `Power` and `SampleSize` classes
- Troubleshooting guidance (performance, memory, and numerical stability)

## License

MIT License.

## Reference

If you use this package in your research, please cite the following paper:

Piray, Payam, "Addressing low statistical power in computational modeling studies in psychology and neuroscience", *Nature Human Behaviour*, 2025.

