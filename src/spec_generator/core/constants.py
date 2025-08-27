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
        "pink_noise_level": 0.05,
    },
    "Noisy": {
        "base_noise_level": 5.0, "decay_constant": 250.0, "white_noise_level": 0.05,
        "shot_noise_factor": 0.08, "perlin_scale": 50.0, "perlin_octaves": 4,
        "baseline_wobble_amplitude": 3.0, "baseline_wobble_scale": 500.0,
        "pink_noise_level": 0.5,
    },
}

# Mass of two hydrogen atoms, lost during the formation of one disulfide bond.
# Using average mass of H (~1.008 Da) -> 2 * H = 2.016 Da
DISULFIDE_MASS_LOSS = 2.016
