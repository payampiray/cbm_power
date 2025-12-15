from cbm_power import Power
import numpy as np
# =========================
# Example usage
# =========================
if __name__ == "__main__":

    # Computes power for a study with these parameters (replace N, K with your values)
    N = 410  # number of participants
    K = 10   # number of models

    power = []

    for i in range(5):
        sim = Power(seed=i)
        pwr, result = sim.compute_power(N, K, target_effect_size=None)
        power.append(pwr)

    # Convert to numpy arrays for convenience
    power = np.array(power)
    std_power = np.std(power)
    mean_power = np.mean(power)

    # Display (similar to MATLAB display)
    print(f"Power estimate: {mean_power:.2f} ± {std_power:.2f}")
