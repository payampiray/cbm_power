# CBM Power — User Manual

**Computational Behavioral/Brain Modeling (CBM) — Power & Sample Size**

This notebook introduces a method for computing power in computational modeling studies that involve **model selection among multiple competing models**, and shows how to use the Python API (`Power`, `SampleSize`, `Config`).

## 1. Motivation & key ideas

- **Goal.** Introduce a principled method to compute statistical power in computational modeling studies with model selection across a set of models.
- **Insight.** When the number of competing models increases, the power of model selection decreases (it becomes harder to reliably identify the best model).
- **Significance.** This factor has been largely neglected in the field; as a result, many studies may be underpowered.
- **Key idea:** Treat model comparison as a stochastic process arising from finite data and uncertainty about population-level generative parameters. Use simulation to estimate the probability that the *correct* model wins by an exceedance-probability threshold.


### Reference
If you use this code or method, please cite:
> Piray, Payam, “Addressing low statistical power in computational modeling studies in psychology and neuroscience”, *Nature Human Behaviour*, 2025.


## 2. Setup

Install the library (-e for editable mode):

```bash
pip install -e .
```

Then import the API:



```python
# Minimal imports from the package
# from cbm_power import Power, SampleSize, Config

# For this demo notebook we avoid heavy runs and just show usage examples.
print("Ready to use: Power, SampleSize, Config")
```

## 3. Computing power for a given sample size & model space

The main function of the `Power` class is **`compute_power`**.  
It returns **two** objects:
1. `estimated_power` (float), and  
2. a `result` object containing detailed fields (posterior parameters, exceedance probabilities, etc.).

**Minimal configuration** requires only:
- `num_participants` (sample size, **N**), and
- `num_models` (size of the model space, **K**).

### Example (pseudo-code)
```python
from cbm_power import Power

# Example (for a faster experiment, decrease num_sim to 1000)
pwr = Power()
estimated_power, result = pwr.compute_power(
    num_participants=410,
    num_models=10,
    num_sim = 10_000
)
print(f"Power ≈ {estimated_power:.2f}")  # two-decimal display

`compute_power(...) -> (estimated_power, result)`

- **`estimated_power` (float):** Mean probability (over simulations) that the selected best model's exceedance probability passes the null threshold for the acceptable false positive rate.

- **`result` (a dataclass object):**
  - `exceedance_threshold` *(float)* — the threshold chosen under the null to control false positives at `false_positive_acceptable`.
  - `false_positive_rate` *(float)* — measured false positive rate at the chosen threshold (should be ≤ specified tolerance).
  - `posterior_parameters` *(array)* — group-level Dirichlet parameters used to compute exceedance probabilities.
  - `exceedance_prob` *(array)* — per-simulation exceedance probabilities across models.
  - `population_samples` *(array)* — sampled population-level multinomial probabilities (rows where model 1 is best).
  - **Configuration snapshot (for reproducibility)** — e.g. `num_sim`, `num_samples_4ep`, `max_iter`, and any true/target parameters used internally.

## 4. Sample-Size optimization

`SampleSize.run()` performs a 4-stage optimization of sample size, N:
1. **Stage 0:** Initialization via a fast procedure that does not compute exceedance probabilities.
2. **Stages 1–2:** Bayesian optimization (BO) over N using repeated evaluations of power (using ax-platform package).
3. **Stage 3:** Exhaustive downward search from the best candidate to find the smallest N that still achieves the target power at two-decimal resolution.

Since optimization can be complex, this process may take some time.

### Example (pseudo-code)
```python
from cbm_power import SampleSize, Config

cfg = Config(
    num_models=5,
    false_positive_acceptable=0.05,
)
ss = SampleSize(cfg)
cbm = ss.compute_sample_size()   # Runs all 4 stages; saves JSON + log automatically
# optinal: filename_stem="my_run"  # controls my_run.json / my_run.log
# cbm = ss.compute_sample_size(save_path=filename_stem)   # Runs all 4 stages; saves JSON + log automatically

print("Optimized N:", cbm.output.sample_size)  # dataclass field access
```

## 5. Configuration guide

`Config` fields (common):
- `num_models` *(int, required)* — size of the model space (K).
- `rng` *(int or numpy.random.Generator)* — random seed or RNG for reproducibility. Optional; defaults to seed 0.
- `prior_parameter` *(float)* — Dirichlet prior strength (default 1.0).
- `false_positive_acceptable` *(float)* — acceptable level for determining the exceedance threshold under the null (default 0.05).
- `desired_power` *(float)* — target power for optimization (default 0.80).
- `optimize_true` *(bool)* — if `True`, attempt to find a Dirichlet parameter matching `target_effect_size` (default `False`).
- `target_effect_size` *(float or None)* — target mean effect size (the population-level difference between probabilities of the best and second-best models) used when optimizing the true distribution. If set to None and optimize_true is True, it defaults to 0.3, which is considered a medium effect size for two models (per Cohen). Note that for studies with more than two competing models, this value may correspond to a large or even very large effect size.
- `true_parameter` *(float or None)* — if given and `optimize_true=False`, use this value for the population Dirichlet; otherwise it can be ignored and optimized (default 1).

Per-stage arrays (length 4):
- `max_iter` — maximum Bayesian optimization iterations per stage (default `[50, 100, 100, None]`).
- `tol` — stop tolerance per stage (default `[0.05, 0.02, 0.015, 0.015]`).
- `replicates` — number of repetitions with different seeds per trial (default `[1, 5, 20, 10]`).
- `num_sim` — number of simulations scenarios (default `[10000, 1000, 1000, 5000]`).
- `num_sim_4ep` — number of samples for computing exceedance probability (default `[10000, 1000, 1000, 5000]`).

## 6. Reproducibility & saved artifacts

- **JSON**: Full results (initialization, BO stages, exhaustive trials, configuration snapshot).
- **Log**: Console-like trace of trials and stage summaries.

If a filename stem is provided in `Config`, outputs will be saved as `<stem>.json` and `<stem>.log`. If **not provided**, the library defaults to timestamped filenames like `cbm_YYYYMMDD_HHMMSS.json`.


## 7. Troubleshooting & performance Tips

- **Performance / Memory.** Two knobs control runtime and memory:
  - `num_sim`: number of population scenarios sampled for estimating power.
  - `num_samples_4ep`: number of samples per scenario for estimating exceedance probabilities.

  Reduce these to speed up experiments. In this case, run multiple seeds and compute mean power with error bars (e.g., standard error or a confidence interval). On a typical laptop (e.g., macOS with 16 GB RAM), `num_sim=10_000` and `num_samples_4ep=10_000` works successfully.

- **Flat objective regions.** When power is rounded to two decimals, many N values can look equivalent. The optimizer handles ties, and the exhaustive final stage ensures the smallest adequate N is returned.

- **Numerical warnings.** Gaussian-process-based BO may add small jitter during optimization. This is generally harmless and indicates numerical stabilization.


## 8. Interpreting saved outputs

`SampleSize.compute_sample_size()` creates two artifacts in the working directory (unless a file_path is given in SampleSize.compute_sample_size(save_path=file_path)):

- `cbm_YYYYMMDD_HHMMSS.json` — a full, JSON-serializable record of:
  - Final output (`sample_size`, `power` mean and `dispersion`, final replicate vector)
  - Stage-by-stage results (including the exhaustive Stage 3 trials)
  - Config used (without non-serializable RNG objects)
  - Artifact paths
- `cbm_YYYYMMDD_HHMMSS.log` — a mirror of console output (trials, early stops, exhaustive search trace).
