import math

# --- Constants and Noise Presets ---
BASE_INTENSITY_SCALAR = 2000.0
MZ_SCALE_FACTOR = 1000.0
PROTON_MASS = 1.007276
NEUTRON_MASS_APPROX = 1.00866491588
FWHM_TO_SIGMA = 2 * math.sqrt(2 * math.log(2))

noise_presets = {
    "Default Noise": {
        "base_noise_level": 0.05, "decay_constant": 200.0, "white_noise_level": 0.002,
        "shot_noise_factor": 0.01, "perlin_scale": 50.0, "perlin_octaves": 4,
        "baseline_wobble_amplitude": 0.1, "baseline_wobble_scale": 100.0,
        "white_noise_decay_constant": 200.0, "baseline_wobble_decay_constant": 300.0,
    },
    "Noisy": {
        "base_noise_level": 5.0, "decay_constant": 250.0, "white_noise_level": 0.05,
        "shot_noise_factor": 0.08, "perlin_scale": 50.0, "perlin_octaves": 4,
        "baseline_wobble_amplitude": 3.0, "baseline_wobble_scale": 500.0,
        "white_noise_decay_constant": 600.0, "baseline_wobble_decay_constant": 1500.0,
    },
}
