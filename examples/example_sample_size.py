from cbm_power import Config
from cbm_power import SampleSize

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # finds optimal sample size for a study with K models
    # automatically saves the results in a json file in the current directory, alongside a log file.
    K = 10

    cfg = Config(num_models=K, optimize_true=True)
    cbm = SampleSize(cfg).compute_sample_size()


