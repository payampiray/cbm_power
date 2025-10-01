# cbm_power/sample_size.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
from dataclasses import dataclass, asdict

from .config import Config
from .optimizer import optimizer_ax_repeated
from .power import Power
from .logger import RunLogger, make_base_path

Array = np.ndarray

@dataclass
class StageResult:
    N: int
    power: List[float]
    optimized: float


@dataclass
class ExhaustiveTrial:
    N: int
    mean_power: float
    power_vec: List[float]


@dataclass
class ExhaustiveResult:
    trials: List[ExhaustiveTrial]
    criterion: str
    replicates_per_N: int


@dataclass
class OutputSummary:
    sample_size: int
    power: float
    dispersion: float
    repetitions: List[float]


@dataclass
class ConfigResult:
    num_models: int
    false_positive_acceptable: float
    prior_parameter: float
    desired_power: float
    optimize_true: bool
    target_effect_size: float
    true_parameter: float
    max_iter: List[int]
    tol: List[float]
    replicates: List[int]
    num_sim: List[int]
    num_sim_4ep: List[int]

    # artifacts merged here
    json_path: str
    log_path: str


@dataclass
class CBMResult:
    output: OutputSummary
    initialization: StageResult
    stages: List[StageResult]
    exhaustive: ExhaustiveResult
    config: ConfigResult


class SampleSize:
    """
    Four-stage sample-size optimization (Stages 0..3).

    For each suggested N at each stage, evaluate the objective `replicates` times
    with independent RNGs and minimize:
        loss(N) = abs(mean(power - desired_power)) + std(power - desired_power)

    Stage 0 uses initializer (null threshold + dp criterion).
    Stages 1–2 use full Power.run_power with Bayesian optimization.
    Stage 3 performs a downward exhaustive search from Stage 2's N, using 10 reps.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = cfg.rng
        self.num_models = cfg.num_models

    def compute_sample_size(self, save_path: Optional[str] = "") -> Dict[str, Any]:
        """
        Run stages 0..3.

        Parameters
        ----------
        save_path : str or None
            If empty/None, results are saved to:
              cbm_YYYYMMDD_HHMMSS.json  (and sibling .log with same stem)
            If provided (with or without extension), its stem is used for both files.

        Returns
        -------
        cbm : dict
            JSON-serializable result containing summary, per-stage detail, exhaustive info, and artifact paths.
        """
        # Resolve base path (no extension). If empty, use timestamped default.
        base = make_base_path(save_path)

        # Tee all prints to console and to <base>.log
        with RunLogger(base):
            cfg = self.cfg
            num_models = self.num_models

            results: List[Dict[str, Any]] = []

            # ---------------- Stage 0 ----------------
            print("+" * 50)
            print("Stage 0 optimization:\n")

            s0 = cfg.stage0  # dict with max_iter, tol, replicates, num_sim, num_sim_4ep

            # Null threshold for (best - second-best) posterior-count difference
            diff_param_thresh = self._run_initializer_null(
                num_participants=5 * num_models,
                num_models=num_models,
                alpha_acceptable=cfg.false_positive_acceptable,
                prior_parameter=cfg.prior_parameter,
                num_sim=s0["num_sim"],
                rng=self.rng,
            )

            # Single eval for one repetition (child RNG supplied by optimizer)
            def fun_stage0_single(n: int, rng_child) -> float:
                return self._run_initializer_power(
                    num_participants=int(n),
                    num_models=num_models,
                    diff_parameters_threshold=diff_param_thresh,
                    prior_parameter=cfg.prior_parameter,
                    num_sim=s0["num_sim"],
                    rng=rng_child,
                )

            res0, _R0, _range0 = optimizer_ax_repeated(  # noqa: F841
                fun_single=fun_stage0_single,
                K=num_models,
                stage=0,
                desired_power=cfg.desired_power,
                max_eval=s0["max_iter"],
                n0=None,
                tol=s0["tol"],
                num_repeat=s0["replicates"],
                master_rng=self.rng,
            )
            results.append(res0)
            N_prev = int(res0["N"])

            # ---------------- Stages 1–2 (BO) ----------------
            true_dirichlet_parameter = 1
            if cfg.target_effect_size is not None:
                rng_child = np.random.default_rng(
                    int(self.rng.integers(0, 2 ** 32 - 1, dtype=np.uint32))
                )
                true_dirichlet_parameter = Power(rng=rng_child).optimize_true_parameter(
                    num_models=num_models,
                    target_effect_size=cfg.target_effect_size,
                )
                print("+" * 50)
                print(f"Optimizing dirichlet parameter of the true distribution\n")

            for stage in [1, 2]:
                print("+" * 50)
                print(f"Stage {stage} optimization:\n")

                s_cfg = getattr(cfg, f"stage{stage}")

                def fun_stage_single(n: int, rng_child, s_cfg=s_cfg) -> float:
                    pwr, _ = Power(rng=rng_child).compute_power(
                        num_participants=int(n),
                        num_models=num_models,
                        false_positive_acceptable=cfg.false_positive_acceptable,
                        true_parameter=true_dirichlet_parameter,
                        target_effect_size=None,
                        prior_parameter=cfg.prior_parameter,
                        num_sim=s_cfg["num_sim"],
                        num_samples_4ep=s_cfg["num_sim_4ep"],
                    )
                    # Optimize to two decimals (effectively near-deterministic)
                    return float(np.round(pwr, 2))

                res, _R, _rng = optimizer_ax_repeated(  # noqa: F841
                    fun_single=fun_stage_single,
                    K=num_models,
                    stage=stage,
                    desired_power=cfg.desired_power,
                    max_eval=s_cfg["max_iter"],
                    n0=N_prev,
                    tol=s_cfg["tol"],
                    num_repeat=s_cfg["replicates"],
                    master_rng=self.rng,
                )
                results.append(res)
                N_prev = int(res["N"])

            # ---------------- Stage 3 (EXHAUSTIVE downward) ----------------
            print("+" * 50)
            print("Stage 3 exhaustive search:\n")

            s3 = cfg.stage3
            exhaustive_trials: List[Dict[str, Any]] = []

            def mean_power_at(n: int) -> Tuple[float, np.ndarray]:
                """Mean power over 10 replicates at sample size n."""
                reps = []
                for _ in range(10):  # fixed 10 replicates for stage 3
                    rng_child = np.random.default_rng(
                        int(self.rng.integers(0, 2**32 - 1, dtype=np.uint32))
                    )
                    pwr, _ = Power(rng=rng_child).compute_power(
                        num_participants=int(n),
                        num_models=num_models,
                        false_positive_acceptable=cfg.false_positive_acceptable,
                        true_parameter=true_dirichlet_parameter,
                        target_effect_size=None,
                        prior_parameter=cfg.prior_parameter,
                        num_sim=s3["num_sim"],
                        num_samples_4ep=s3["num_sim_4ep"],
                    )
                    reps.append(float(pwr))
                return float(np.mean(reps)), np.asarray(reps, dtype=float)

            # Start from N_prev, walk down to find smallest N with two-decimal power ≥ target
            target2 = float(np.round(cfg.desired_power, 2))
            N_probe = int(N_prev)
            best_N = None
            best_power_vec = None
            best_loss = None  # |mean - target| + std at the best

            while N_probe >= 1:
                mean_pwr, reps_vec = mean_power_at(N_probe)
                mean2 = float(np.round(mean_pwr, 2))

                print(
                    f"[exhaustive] N={N_probe}, mean_power={mean_pwr:.4f} (→ {mean2:.2f}), "
                    f"target={target2:.2f}"
                )

                exhaustive_trials.append(
                    {
                        "N": int(N_probe),
                        "mean_power": float(mean_pwr),
                        "power_vec": reps_vec.tolist(),
                    }
                )

                if mean2 >= target2:
                    best_N = N_probe
                    best_power_vec = reps_vec
                    diff = reps_vec - cfg.desired_power
                    best_loss = float(abs(np.mean(diff)) + np.std(diff))
                    N_probe -= 1
                else:
                    break

            # Fallback: if none met the target at two decimals, keep N_prev
            if best_N is None:
                best_N = int(N_prev)
                mean_pwr, best_power_vec = mean_power_at(best_N)
                exhaustive_trials.append(
                    {
                        "N": int(best_N),
                        "mean_power": float(mean_pwr),
                        "power_vec": best_power_vec.tolist(),
                    }
                )
                diff = best_power_vec - cfg.desired_power
                best_loss = float(abs(np.mean(diff)) + np.std(diff))

            res3 = {
                "N": int(best_N),
                "power": best_power_vec,
                "optimized": float(best_loss),
            }
            results.append(res3)
            N_prev = int(best_N)

            # ---------------- Output summary ----------------
            N_optimal = int(N_prev)
            p_last = np.asarray(results[-1]["power"])
            power_mean = float(np.mean(p_last))
            power_sd = float(np.std(p_last))

            cbm = CBMResult(
                output=OutputSummary(
                    sample_size=N_optimal,
                    power=power_mean,
                    dispersion=power_sd,
                    repetitions=p_last.tolist(),
                ),
                initialization=StageResult(
                    N=int(results[0]["N"]),
                    power=np.asarray(results[0]["power"]).tolist(),
                    optimized=float(results[0]["optimized"]),
                ),
                stages=[
                    StageResult(
                        N=int(results[i]["N"]),
                        power=np.asarray(results[i]["power"]).tolist(),
                        optimized=float(results[i]["optimized"]),
                    )
                    for i in range(1, 4)
                ],
                exhaustive=ExhaustiveResult(
                    trials=[ExhaustiveTrial(**trial) for trial in exhaustive_trials],
                    criterion="two-decimal power >= target",
                    replicates_per_N=10,
                ),
                config=ConfigResult(
                    num_models=cfg.num_models,
                    false_positive_acceptable=float(cfg.false_positive_acceptable),
                    optimize_true=cfg.optimize_true,
                    target_effect_size=cfg.target_effect_size,
                    prior_parameter=cfg.prior_parameter,
                    desired_power=float(cfg.desired_power),
                    true_parameter=float(true_dirichlet_parameter),
                    max_iter=list(cfg.max_iter),
                    tol=[float(x) for x in cfg.tol],
                    replicates=list(cfg.replicates),
                    num_sim=list(cfg.num_sim),
                    num_sim_4ep=list(cfg.num_sim_4ep),
                    json_path=f"{base}.json",
                    log_path=f"{base}.log",
                ),
            )

            # Save JSON
            with open(f"{base}.json", "w", encoding="utf-8") as f:
                json.dump(asdict(cbm), f, indent=2)

            # Final console/log dump
            print("\nFinal result:")
            print(cbm)

            return cbm

    # ---------- internal null functions ----------
    def _run_initializer_null(
            self,
            num_participants: int,
            num_models: int,
            alpha_acceptable: float,
            prior_parameter: float,
            num_sim: int,
            rng: np.random.Generator,
    ) -> int:
        """Generate null distribution of (best - second-best) posterior counts and pick threshold."""
        probs = np.full(num_models, 1.0 / num_models)
        group_samples = rng.multinomial(num_participants, probs, size=num_sim)
        posterior_parameters = group_samples + prior_parameter

        sorted_post = np.sort(posterior_parameters, axis=1)[:, ::-1]
        dp = (sorted_post[:, 0] - sorted_post[:, 1]).astype(int)

        max_dp = int(dp.max())
        if max_dp < 1:
            return 1

        ndp = np.array([np.mean(dp == i) for i in range(1, max_dp + 1)])
        idx = np.argmax(ndp <= alpha_acceptable) if np.any(ndp <= alpha_acceptable) else None
        return int(idx + 1) if idx is not None else max_dp


    def _run_initializer_power(
            self,
            num_participants: int,
            num_models: int,
            diff_parameters_threshold: int,
            prior_parameter: float,
            num_sim: int,
            rng: np.random.Generator,
    ) -> float:
        """Estimate power of the initializer stage by sampling population allocations and groups."""
        max_iter = 100
        population_samples: list[Array] = []
        total, it = 0, 0
        alpha_vec = np.ones(num_models, dtype=float)

        while (total < num_sim) and (it < max_iter):
            r = rng.dirichlet(alpha_vec, size=num_sim)
            is_best1 = r[:, 0] > r[:, 1:].max(axis=1)
            keep = r[is_best1]
            if keep.size:
                population_samples.append(keep)
                total += keep.shape[0]
            it += 1

        if total == 0:
            fallback = np.zeros((num_sim, num_models), dtype=float)
            fallback[:, 0] = 1.0
            population = fallback
        else:
            population = np.vstack(population_samples)[:num_sim, :]

        group_samples = np.vstack([rng.multinomial(num_participants, p) for p in population])
        posterior_parameters = group_samples + prior_parameter
        dp = posterior_parameters[:, 0] - posterior_parameters[:, 1:].max(axis=1)

        return float(np.mean(dp > diff_parameters_threshold))
