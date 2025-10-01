from typing import Callable, Tuple, Dict, Any, List
import numpy as np
from ax.service.ax_client import AxClient
from ax.exceptions.core import SearchSpaceExhausted
from ax.service.utils.instantiation import ObjectiveProperties
import contextlib
import logging
logging.getLogger("ax").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"MapKeyToFloat is unable to identify `parameters`",
    category=UserWarning,
    module=r"ax\.adapter\.base",
)

EvalFunSingle = Callable[[int, np.random.Generator], float]


def _stage_search_range(K: int, stage: int, n0: int | None) -> Tuple[int, int]:
    if stage == 0:
        lo, hi = K * 5, K * 500
    elif stage == 1:
        assert n0 is not None, "n0 must be provided for stage 1"
        lo, hi = min(K * 5, round(n0 / 4)), max(K * 500, n0 * 4)
    elif stage == 2:
        assert n0 is not None
        lo, hi = n0 - 10 * K, n0 + 10 * K
    elif stage == 3:
        assert n0 is not None
        lo, hi = n0 - 10, n0 + 9
    else:
        raise ValueError("stage must be 0..3")
    lo = max(1, int(lo))
    hi = max(lo + 1, int(hi))
    return lo, hi


def _evaluate_at_N(
    fun_single: EvalFunSingle,
    n: int,
    num_repeat: int,
    desired_power: float,
    master_rng: np.random.Generator,
) -> Tuple[float, np.ndarray]:
    power_vals: List[float] = []
    for _ in range(num_repeat):
        seed = int(master_rng.integers(0, 2**32 - 1, dtype=np.uint32))
        child = np.random.default_rng(seed)
        p = float(fun_single(int(n), child))
        power_vals.append(p)

    power_vec = np.asarray(power_vals, dtype=float)
    diff = power_vec - desired_power
    loss = float(abs(np.mean(diff)) + np.std(diff))
    eps = 1e-3
    loss_penalized = float(loss + eps * n)
    return loss_penalized, power_vec


def optimizer_ax_repeated(
    fun_single: Callable[[int, np.random.Generator], float],
    K: int,
    stage: int,
    desired_power: float,
    max_eval: int,
    n0: int | None,
    tol: float,
    num_repeat: int,
    master_rng: np.random.Generator,
) -> Tuple[Dict[str, Any], Any, Tuple[int, int]]:
    # ----- Stage-dependent search range -----
    if stage == 0:
        lo, hi = K * 5, K * 500
    elif stage == 1:
        assert n0 is not None
        lo, hi = min(K * 5, round(n0 / 4)), max(K * 500, n0 * 4)
    elif stage == 2:
        assert n0 is not None
        lo, hi = n0 - 10 * K, n0 + 10 * K
    elif stage == 3:
        assert n0 is not None
        lo, hi = n0 - 10, n0 + 9
    else:
        raise ValueError(f"Invalid stage {stage}")
    lo, hi = max(1, int(lo)), max(2, int(hi))
    search_range = (lo, hi)

    # prefer smaller N on ties
    target_max_penalty = 1e-3
    eps = target_max_penalty / max(1, hi - lo)

    def objective(params: Dict[str, Any]) -> float:
        n = int(params["N"])
        reps: List[float] = []
        for _ in range(num_repeat):
            child = np.random.default_rng(int(master_rng.integers(0, 2**32 - 1, dtype=np.uint32)))
            reps.append(float(fun_single(n, child)))
        reps = np.asarray(reps, dtype=float)
        diff = reps - desired_power
        loss = float(abs(np.mean(diff)) + np.std(diff))
        loss_pen = float(loss + eps * n)
        print(f"[trial] N={n}, mean_power={reps.mean():.4f}, loss={loss:.6f}, loss+pen={loss_pen:.6f}")
        return loss_pen

    # ----- Ax setup -----
    ax = AxClient(enforce_sequential_optimization=True, verbose_logging=False)
    ax.create_experiment(
        name=f"stage{stage}_optimizer",
        parameters=[{
            "name": "N",
            "type": "range",
            "bounds": [lo, hi],
            "value_type": "int",
        }],
        objectives={"loss": ObjectiveProperties(minimize=True)},
    )

    if n0 is not None and lo <= int(n0) <= hi:
        try:
            ax.attach_trial({"N": int(n0)})
        except Exception:
            pass

    # ----- Prefer higher baseline jitter during model fits -----
    # Try the public settings API first; fall back to no-op if unavailable.
    try:
        from linear_operator.settings import settings as linop_settings  # public name on recent versions
        jitter_cm = linop_settings.cholesky_jitter(1e-5)  # bump baseline jitter
        # optional: also allow larger Cholesky before switching methods
        maxchol_cm = linop_settings.max_cholesky_size(8192)
        linop_cm = contextlib.ExitStack()
        linop_cm.enter_context(jitter_cm)
        linop_cm.enter_context(maxchol_cm)
    except Exception:
        # older builds or missing API: just no-op
        linop_cm = contextlib.nullcontext()

    # ----- BO loop -----
    num_trials = 0
    tried_any = False
    with linop_cm:
        while num_trials < max_eval:
            try:
                params, tid = ax.get_next_trial()
            except SearchSpaceExhausted:
                print("[ax] Search space exhausted.")
                break
            tried_any = True
            f0 = objective(params)
            ax.complete_trial(trial_index=tid, raw_data={"loss": (f0, 0.01)})
            num_trials += 1
            if f0 < tol:
                print(f"[ax] Early stop: loss {f0:.6f} < tol {tol:.6f}")
                break

    if not tried_any:
        N_fallback = int(np.clip((lo + hi) // 2, lo, hi))
        rngs = [np.random.default_rng(int(master_rng.integers(0, 2**32 - 1, dtype=np.uint32)))
                for _ in range(num_repeat)]
        power_vec = np.array([fun_single(N_fallback, r) for r in rngs], dtype=float)
        diff = power_vec - desired_power
        f_best = float(abs(np.mean(diff)) + np.std(diff) + eps * N_fallback)
        return {"N": N_fallback, "power": power_vec, "optimized": f_best}, ax, search_range

    # ----- best with smallest-N tie break -----
    df = ax.get_trials_data_frame()
    if "N" in df.columns and "mean" in df.columns:
        best_mean = df["mean"].min()
        winners = df[df["mean"] <= best_mean + 1e-12]
        N_best = int(winners["N"].min())
    else:
        best_params, _ = ax.get_best_parameters()
        N_best = int(best_params["N"])

    rngs = [np.random.default_rng(int(master_rng.integers(0, 2**32 - 1, dtype=np.uint32)))
            for _ in range(num_repeat)]
    power_vec = np.array([fun_single(N_best, r) for r in rngs], dtype=float)
    diff = power_vec - desired_power
    f_best = float(abs(np.mean(diff)) + np.std(diff) + eps * N_best)

    return {"N": N_best, "power": power_vec, "optimized": f_best}, ax, search_range

