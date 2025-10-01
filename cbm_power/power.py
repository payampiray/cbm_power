from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Any
import numpy as np
import warnings

from .distributions import (
    dirichlet_rows,
    multinomial_rows, compute_exceedance,
)

Array = np.ndarray


# ---------------- Results dataclasses ----------------

# ---- 1) config snapshot (inputs) ----
@dataclass(frozen=True)
class PowerConfig:
    num_sim: int
    num_samples_4ep: int
    prior_parameter: float
    max_iter: int
    false_positive_acceptable: float     # replaces alpha
    true_parameter: Optional[float]      # what you actually used, or None if optimized
    target_effect_size: Optional[float]  # target for optimization (if used)


# ---- 2) Model outputs / metrics (small, easy to serialize) ----
@dataclass
class PowerMetrics:
    power: float
    exceedance_threshold: float
    false_positive_rate: float
    true_dirichlet_parameter: float

# ---- 3) Heavy arrays (keep separate so you can drop or summarize) ----
@dataclass
class PowerData:
    posterior_parameters: Array
    exceedance_prob: Array
    population_samples: Array

# ---- 4) The full result container ----
@dataclass
class PowerResult:
    metrics: PowerMetrics
    data: Optional[PowerData]        # set to None when saving “lightweight” JSON
    config: PowerConfig

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """JSON-ready dict. By default omits large arrays."""
        d = {
            "metrics": asdict(self.metrics),
            "config": asdict(self.config),
        }
        if include_arrays and self.data is not None:
            d["data"] = {
                "posterior_parameters": self.data.posterior_parameters.tolist(),
                "exceedance_prob": self.data.exceedance_prob.tolist(),
                "population_samples": self.data.population_samples.tolist(),
            }
        else:
            d["data"] = None
        return d


class Power:
    """
    Estimate power given an exceedance threshold, using simulated population
    samples and group-level posterior parameters.
    """

    def __init__(self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        """
        Initialize a Power estimator.

        Parameters
        ----------
        seed : int, optional
            If provided and rng is None, creates a new Generator with this seed.
        rng : np.random.Generator, optional
            If provided, use this RNG instance directly (preferred for reproducibility).
        """
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

    def compute_power(
        self,
        num_participants: int,
        num_models: int,
        false_positive_acceptable: float = 0.05,
        prior_parameter: float = 1.0,
        true_parameter: float = 1.0,
        target_effect_size: float = None,
        num_sim: int = 10000,
        num_samples_4ep: int = 10000,
        max_iter: int = 100,
    ) -> Tuple[float, PowerResult]:

        # 0) determine true_dirichlet_parameter with optimization or using the user-specified value (default 1.0)

        # Decide whether to optimize the Dirichlet parameter
        # optimize is True if:
        #   - true_parameter is None  -> we must find a parameter that hits a target effect size
        #   - OR target_effect_size is provided -> user explicitly wants to optimize, even if true_parameter was given
        optimize = (true_parameter is None) or (target_effect_size is not None)

        if optimize:
            # ---- ENTERS HERE WHEN:
            #   A) true_parameter is None (with or without a target_effect_size), OR
            #   B) target_effect_size is provided (even if true_parameter is given)
            # ---------------------------------------------------------------

            if target_effect_size is None:
                # ---- ENTERS THIS INNER BRANCH WHEN:
                #   - optimize was triggered because true_parameter is None
                #   - AND user did not provide target_effect_size
                #   -> use default 0.3
                target_effect_size = 0.3
                warnings.warn(
                    f"Optimizing the true distribution with the target effect size of {target_effect_size}",
                    UserWarning,
                )

            elif true_parameter is not None:
                # ---- ENTERS THIS INNER BRANCH WHEN:
                #   - user provided BOTH true_parameter AND target_effect_size
                #   -> we IGNORE true_parameter and optimize to match the target_effect_size
                warnings.warn(
                    f"Ignoring provided true_parameter={true_parameter}; "
                    f"optimizing for target effect size {target_effect_size}",
                    UserWarning,
                )
                true_parameter = None


            # Perform the optimization of the Dirichlet concentration parameter
            true_dirichlet_parameter = self.optimize_true_parameter(
                num_models=num_models,
                target_effect_size=target_effect_size,
                num_sim=num_sim,
                max_iter=max_iter,
            )

        else:
            # ---- ENTERS HERE WHEN:
            #   - true_parameter IS provided
            #   - AND target_effect_size is NOT provided (i.e., None)
            #   -> no optimization; use the provided true_parameter
            true_dirichlet_parameter = float(true_parameter)

        # 1) Null threshold
        threshold, false_positive_rate = self._run_null(
            num_participants=num_participants,
            num_models=num_models,
            alpha_acceptable=false_positive_acceptable,
            prior_parameter=prior_parameter,
            num_sim=num_sim,
            num_samples_4ep=num_samples_4ep,
        )

        # 2) Population where model 1 is best (batched loop)
        population, _ = self._sample_population_best1(
            num_models=num_models,
            dirichlet_parameter=true_dirichlet_parameter,
            num_sim=num_sim,
            max_iter=max_iter,
        )

        # 3) Group samples → posterior Dirichlet parameters
        group_samples = multinomial_rows(self.rng, num_participants, population)  # (num_sim,K)
        posterior_parameters = group_samples + prior_parameter                    # broadcast add

        # 4) Exceedance (batched)
        exceedance_prob = compute_exceedance(self.rng, posterior_parameters, num_samples_4ep)

        # 5) Power w.r.t. threshold
        power_scalar = float(np.mean(exceedance_prob[:, 0] > threshold))

        metrics = PowerMetrics(
            power=float(power_scalar),
            exceedance_threshold=float(threshold),
            false_positive_rate=float(false_positive_rate),  # compute this alongside threshold
            true_dirichlet_parameter=float(true_dirichlet_parameter),
        )

        data = PowerData(
            posterior_parameters=posterior_parameters,
            exceedance_prob=exceedance_prob,
            population_samples=population,
        )
        config = PowerConfig(
            num_sim=num_sim,
            num_samples_4ep=num_samples_4ep,
            prior_parameter=prior_parameter,
            max_iter=max_iter,
            false_positive_acceptable=false_positive_acceptable,  # your input alpha
            true_parameter=true_parameter,  # resolved value
            target_effect_size=target_effect_size,
        )

        result = PowerResult(metrics=metrics, data=data, config=config)
        return power_scalar, result

    def optimize_true_parameter(
        self,
        num_models: int,
        target_effect_size: float = 0.30,
        tol_effect_size: float = 0.01,
        step_base: float = 0.02,
        start_parameter: float = 1.0,
        num_sim: int = 10_000,
        max_iter: int = 50,
    ) -> float:
        """
        Iteratively adjusts the true Dirichlet parameter so the mean
        effect size (r1 - max(r2..rK)) is within `tol_effectsize` of
        `target_effect_size`.

        Returns
        -------
        best_param : float
        best_effect : float   # mean effect size at best_param
        history : dict        # parameter & effectsize traces
        """
        # first evaluation at the starting parameter
        b = float(start_parameter)
        _, eff = self._sample_population_best1(num_models=num_models, dirichlet_parameter=b, num_sim=num_sim)
        effects = [float(np.mean(eff))]
        parameters = [b]

        # choose initial step direction (if too small -> decrease b)
        step = -step_base if (effects[0] < target_effect_size) else +step_base

        prev_error = effects[0] - target_effect_size

        i = 0
        while (abs(effects[-1] - target_effect_size) > tol_effect_size) and (i < max_iter):
            # propose new b
            b_new = b + step
            # guardrails to keep parameter positive
            b_new = max(b_new, 1e-6)

            _, eff_new = self._sample_population_best1(num_models=num_models, dirichlet_parameter=b_new, num_sim=num_sim)
            mean_eff_new = float(np.mean(eff_new))

            # sign-change detection -> flip and halve step (MATLAB: step = -step/2)
            error_new = mean_eff_new - target_effect_size
            if (error_new > 0) != (prev_error > 0):
                step = -step / 2.0

            # accept move
            b = b_new
            prev_error = error_new
            i += 1

        return b

    # ---------- internal null functions ----------
    def _run_null(
        self,
        num_participants: int,
        num_models: int,
        alpha_acceptable: float,
        prior_parameter: float,
        num_sim: int,
        num_samples_4ep: int,
    ) -> Tuple[float, float]:
        probs = np.full((num_sim, num_models), 1.0 / num_models)
        group_samples = multinomial_rows(self.rng, num_participants, probs)
        posterior_parameters = group_samples + prior_parameter

        exceedance_prob = compute_exceedance(self.rng, posterior_parameters, num_samples_4ep)
        threshold, fpr = self._find_critical_threshold(exceedance_prob, alpha_acceptable)

        return float(threshold), float(fpr)

    def _find_critical_threshold(self, exceedance_prob: Array,
                                alpha_acceptable: float,
                                step: float = 1e-4) -> Tuple[float, float]:
        """
        Sweep thresholds in [1/K, 1] with given step; pick smallest with FP <= alpha.
        Returns (threshold, false_positives_rate).
        """
        K = exceedance_prob.shape[1]
        t_min, t_max = 1.0 / K, 1.0
        thresholds = np.arange(t_min, t_max + step * 0.5, step)
        max_per_row = exceedance_prob.max(axis=1)
        fp_curve = np.array([np.mean(max_per_row >= t) for t in thresholds])

        idxs = np.where(fp_curve <= alpha_acceptable)[0]
        idx = idxs[0] if idxs.size else (thresholds.size - 1)
        return float(thresholds[idx]), float(fp_curve[idx])

    # ---------- true distribution function ----------
    def _sample_population_best1(
            self,
            num_models: int,
            dirichlet_parameter: float = 1.0,
            num_sim: int = 10_000,
            max_iter: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates population-level scenarios from a uniform Dirichlet(dirichlet_parameter)
        until we collect `num_sim` rows where model 1 is the best, or we hit
        `max_iter` batches.

        Returns
        -------
        population_samples : (num_sim, num_models) ndarray
            The corresponding Dirichlet rows.
        effect_size : (num_sim,) ndarray
            For each kept scenario, r[:,0] - max(r[:,1:]) (only when model 1 is best).
        """
        # Accumulators
        effect_list = []
        pop_list = []
        total = 0
        it = 0

        alpha_vec = np.full(num_models, float(dirichlet_parameter), dtype=float)

        # Keep sampling batches until we have enough rows where model 1 is best
        while (total < num_sim) and (it < max_iter):
            # Sample a fresh batch each iteration (size = num_sim, like the MATLAB version)
            r = dirichlet_rows(self.rng, num_sim, alpha_vec)  # (num_sim, K)

            # Identify scenarios where model 1 is best
            is_best1 = r[:, 0] > r[:, 1:].max(axis=1)

            if np.any(is_best1):
                r_keep = r[is_best1]  # rows where model 1 is max
                # Effect size = r1 - max(other models)
                diff1_2nd = r_keep[:, 0] - r_keep[:, 1:].max(axis=1)  # (M,)

                pop_list.append(r_keep)
                effect_list.append(diff1_2nd)
                total += r_keep.shape[0]

            it += 1

        if total == 0:
            # Degenerate fallback to keep shapes consistent (model 1 is pure winner)
            row = np.zeros(num_models, dtype=float)
            row[0] = 1.0
            population_samples = np.tile(row[None, :], (num_sim, 1))
            effect_size = np.ones(num_sim, dtype=float)  # r1 - max(others) = 1 - 0
            return effect_size, population_samples

        # Concatenate and trim to exactly num_sim rows
        population_samples = np.concatenate(pop_list, axis=0)[:num_sim, :]
        effect_size = np.concatenate(effect_list, axis=0)[:num_sim]

        return population_samples, effect_size
