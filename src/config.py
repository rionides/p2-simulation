""" Setting configs."""

# settings 
CONFIG = {
    "seed": 607,
    "repetitions": 200,
    "p_values": [128, 256, 512],
    "theta_values": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3],
    "regimes": ["linear", "sublinear", "fractional"],
    "alpha": 0.4,
    "gamma": 0.75,
    "noise_sigma": 0.5,
    "weak_count": 0,
    "weak_magnitude": 0.25,
    "designs": [
        {"name": "identity", "mu": 0.0, "reference_lines": [1.0]},
        # Wainwright, Section 4, p. 21: equation (39) and Figure 2 use mu=0.10.
        {"name": "toeplitz", "mu": 0.1, "reference_lines": [0.46, 1.84]},
    ],
    "solver_tolerance": 1e-7,
    "max_iterations": 10000,
    "zero_tolerance": 1e-7,
    "log_every": 200,
}

# copy the full settings and replace just four entries for a small example.
QUICK_CONFIG = CONFIG.copy()
QUICK_CONFIG.update({
    "repetitions": 5,
    "p_values": [128],
    "theta_values": [0.3, 1.1, 2.3],
    "log_every": 15,
})


# challenges and new simulations 

# 1 five true variables are weak and the rest stay at +/-1
WEAK_SIGNALS_CONFIG = CONFIG.copy()
WEAK_SIGNALS_CONFIG["weak_count"] = 5

# big big settings
LARGER_P_CONFIG = CONFIG.copy()
LARGER_P_CONFIG.update({
    "p_values": [512, 1024],
    "regimes": ["sublinear", "fractional"],
})

# Names accepted by the command line and Makefile
CONFIGS = {
    "full": CONFIG,
    "quick": QUICK_CONFIG,
    "weak_signals": WEAK_SIGNALS_CONFIG,
    "larger_p": LARGER_P_CONFIG,
}
