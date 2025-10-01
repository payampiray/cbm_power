# cbm_power/config.py
from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Any, Union, Optional


@dataclass
class Config:
    """
    Configuration for 4-stage sample-size optimization.

    Stages are indexed 0..3. All per-stage arrays must be length-4.
    RNG is provided from the top level for reproducibility.
    """

    # Required
    num_models: int
    rng: Union[np.random.Generator, int] = 0

    # Globals
    prior_parameter: float = 1.0
    false_positive_acceptable: float = 0.05
    desired_power: float = 0.8
    optimize_true: bool = False
    target_effect_size: Optional[float] = None
    true_parameter: Optional[float] = None

    # Per-stage (length-4)
    max_iter: List[int] = field(default_factory=lambda: [50, 100, 100, None])
    tol: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.015, 0.015])
    replicates: List[int] = field(default_factory=lambda: [1, 5, 20, 10])
    num_sim: List[int] = field(default_factory=lambda: [10_000, 1_000, 1_000, 5_000])
    num_sim_4ep: List[int] = field(default_factory=lambda: [10_000, 1_000, 1_000, 5_000])

    def __post_init__(self):
        # Normalize rng: accept int seed or Generator
        if isinstance(self.rng, int):
            self.rng = np.random.default_rng(self.rng)
        elif not isinstance(self.rng, np.random.Generator):
            raise TypeError(
                f"rng must be int or np.random.Generator (got {type(self.rng)})"
            )

        # If asked optimize the dirichlet parameter with a medium effect size for num_model>3
        if self.optimize_true:
            if self.target_effect_size is None:
                if self.num_models > 3:
                    self.true_parameter = None
                    self.target_effect_size = 0.3
                else:
                    self.true_parameter = 1
                    self.target_effect_size = None

        # Validate all stage-wise arrays have length 4
        self._ensure_len4("max_iter", self.max_iter)
        self._ensure_len4("tol", self.tol)
        self._ensure_len4("replicates", self.replicates)
        self._ensure_len4("num_sim", self.num_sim)
        self._ensure_len4("num_sim_4ep", self.num_sim_4ep)

    @staticmethod
    def _ensure_len4(name: str, x: List[Any]) -> None:
        if not isinstance(x, list) or len(x) != 4:
            raise ValueError(f"`{name}` must be a list of length 4 (got {x!r}).")

    # ---- Convenience accessors ----
    def stage(self, i: int) -> Dict[str, Any]:
        if i < 0 or i > 3:
            raise IndexError("stage index must be in {0,1,2,3}.")
        return dict(
            max_iter=self.max_iter[i],
            tol=self.tol[i],
            replicates=self.replicates[i],
            num_sim=self.num_sim[i],
            num_sim_4ep=self.num_sim_4ep[i],
        )

    @property
    def stage0(self) -> Dict[str, Any]:
        return self.stage(0)

    @property
    def stage1(self) -> Dict[str, Any]:
        return self.stage(1)

    @property
    def stage2(self) -> Dict[str, Any]:
        return self.stage(2)

    @property
    def stage3(self) -> Dict[str, Any]:
        return self.stage(3)
