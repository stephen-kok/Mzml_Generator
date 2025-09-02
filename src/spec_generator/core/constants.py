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


from pyteomics.mass import std_aa_mass as AMINO_ACID_MASSES

# --- Masses for Fragmentation ---
# Monoisotopic masses of amino acid residues from pyteomics.mass

# Mass of water, ammonia, and carbon monoxide for fragment ion calculation.
# Using monoisotopic masses.
# H = 1.007825, O = 15.994915, N = 14.003074, C = 12.000000
H2O_MASS = 18.010565
NH3_MASS = 17.026549
CO_MASS = 27.994915

# Mass modifications for different fragment ion types.
# These values are added to the sum of residue masses of the fragment.
# This gives the mass of the neutral fragment.
FRAGMENT_ION_MODIFICATIONS = {
    # N-terminal ions
    'a': -CO_MASS,
    'b': 0.0,
    'c': NH3_MASS,
    # C-terminal ions
    'x': CO_MASS,
    'y': H2O_MASS,
    'z': H2O_MASS - NH3_MASS,
}

# Amino acids prone to neutral loss events.
# The keys are the single-letter amino acid codes, and the values are a list
# of mass losses (e.g., H2O, NH3) that can occur.
NEUTRAL_LOSS_RULES = {
    'S': [H2O_MASS],  # Serine -> loss of water
    'T': [H2O_MASS],  # Threonine -> loss of water
    'D': [H2O_MASS],  # Aspartic Acid -> loss of water
    'E': [H2O_MASS],  # Glutamic Acid -> loss of water
    'N': [NH3_MASS],  # Asparagine -> loss of ammonia
    'Q': [NH3_MASS],  # Glutamine -> loss of ammonia
    'K': [NH3_MASS],  # Lysine -> loss of ammonia
    'R': [NH3_MASS],  # Arginine -> loss of ammonia
}

# --- Fragmentation Intensity Rules ---
# Simplified model for factors that enhance fragmentation at a specific bond.
# Based on the "mobile proton" model. Cleavage C-terminal to Proline is
# highly favored.
# The keys are single-letter amino acid codes. The values are multipliers.
FRAGMENTATION_ENHANCEMENT_RULES = {
    'P': 5.0,  # Proline
    'default': 1.0,
}
