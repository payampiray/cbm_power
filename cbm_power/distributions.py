from __future__ import annotations
import numpy as np

Array = np.ndarray


def dirichlet_rows(rng: np.random.Generator, num_rows: int, alpha_vec: Array) -> Array:
    """(num_rows, K) Dirichlet draws for a single alpha vector."""
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    return rng.dirichlet(alpha_vec, size=num_rows)


def dirichlet_rows_3x(rng: np.random.Generator, alpha_mat: Array, n: int) -> Array:
    """
    For each row alpha[i,:] in (S,K), draw n Dirichlet samples → (S, K, n).
    Matches MATLAB's (S,K,n) layout (reduce over K).
    """
    alpha_mat = np.asarray(alpha_mat, dtype=float)
    S, K = alpha_mat.shape
    out = np.empty((S, K, n), dtype=float)
    for i in range(S):
        out[i, :, :] = rng.dirichlet(alpha_mat[i, :], size=n).T  # (K,n)
    return out

# --------- Multinomial ---------
def multinomial_rows(rng: np.random.Generator, n: int, probs_rows: Array) -> Array:
    """Row-wise multinomial: probs_rows (R,K) → (R,K) counts."""
    return np.vstack([rng.multinomial(n, p) for p in probs_rows])

# --------- Exceedance (batched) ---------
def compute_exceedance(rng: np.random.Generator,
                       dirichlet_parameters: Array,
                       num_samples_4ep: int,
                       batch: int = 1_000) -> Array:
    """
    Monte Carlo exceedance probabilities for each posterior row.
    dirichlet_parameters: (S,K) → returns (S,K)
    """
    S, K = dirichlet_parameters.shape
    remaining = num_samples_4ep
    argmax_accum = np.empty((S, 0), dtype=np.int16)

    while remaining > 0:
        n = min(batch, remaining)
        samples = dirichlet_rows_3x(rng, dirichlet_parameters, n)  # (S,K,n)
        argmax_idx = np.argmax(samples, axis=1)                    # (S,n) over K
        argmax_accum = np.concatenate([argmax_accum, argmax_idx], axis=1)
        remaining -= n

    exceedance = np.zeros((S, K), dtype=float)
    for k in range(K):
        exceedance[:, k] = np.mean(argmax_accum == k, axis=1)
    return exceedance

# --------- Threshold search ---------

