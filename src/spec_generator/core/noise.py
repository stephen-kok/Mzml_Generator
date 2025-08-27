import numpy as np
from numba import jit
from numpy.random import default_rng


@jit(nopython=True)
def _generate_perlin_like_noise_numba(n_points: int, mz_values: np.ndarray, scale: float, octaves: int) -> np.ndarray:
    """
    A Numba-JIT compiled function to generate Perlin-like noise extremely fast.
    This function contains the performance-critical loops.
    """
    total_noise = np.zeros(n_points, dtype=np.float64)

    for i in range(octaves):
        amplitude = 1 / (2**i)
        frequency_scale = scale / (2**i)

        num_control_points = int(n_points / frequency_scale)
        if num_control_points < 2:
            num_control_points = 2

        # Create control points for this octave
        control_points_x = np.linspace(mz_values[0], mz_values[-1], num_control_points)
        control_points_y = np.random.uniform(-1.0, 1.0, num_control_points)

        # Linearly interpolate to generate the noise octave
        total_noise += np.interp(mz_values, control_points_x, control_points_y) * amplitude

    return total_noise


def generate_pink_noise(n_points: int, level: float) -> np.ndarray:
    """
    Generates 1/f (pink) noise using the Voss-McCartney algorithm.
    """
    # Number of octaves to sum
    num_octaves = int(np.log2(n_points))

    # Generate white noise for each octave
    white_noises = [np.random.uniform(-1.0, 1.0, n_points) for _ in range(num_octaves)]

    pink_noise = np.zeros(n_points, dtype=np.float64)

    for i in range(num_octaves):
        amplitude = 1 / (2**i)
        # Downsample by taking every 2^i-th element
        downsampled_noise = white_noises[i][::(2**i)]

        # Interpolate back to the original size
        interpolated_noise = np.interp(
            np.linspace(0, len(downsampled_noise) - 1, n_points),
            np.arange(len(downsampled_noise)),
            downsampled_noise
        )
        pink_noise += interpolated_noise * amplitude

    # Normalize and scale
    pink_noise = pink_noise / np.max(np.abs(pink_noise))
    return pink_noise * level


def add_noise(
    mz_values: np.ndarray,
    intensities: np.ndarray,
    base_noise_level: float,
    min_noise_level: float,
    max_intensity: float,
    decay_constant: float,
    white_noise_level: float,
    shot_noise_factor: float,
    perlin_scale: float,
    perlin_octaves: int,
    baseline_wobble_amplitude: float,
    baseline_wobble_scale: float,
    pink_noise_level: float,
    pink_noise_enabled: bool,
    seed: int,
) -> np.ndarray:
    """
    Adds multiple layers of realistic noise to a spectrum.
    """
    rng = default_rng(seed)

    # Seed numpy's legacy random generator for Numba compatibility.
    np.random.seed(seed)
    total_perlin_chem = _generate_perlin_like_noise_numba(len(mz_values), mz_values, perlin_scale, octaves=perlin_octaves)

    # Re-seed for the second call to get different noise pattern
    np.random.seed(seed + 1)
    perlin_wobble = _generate_perlin_like_noise_numba(len(mz_values), mz_values, baseline_wobble_scale, octaves=1)

    # --- Combine all noise sources using vectorized numpy operations ---
    # 1. Chemical noise (low m/z "haystack")
    chemical_noise_level = np.maximum(
        min_noise_level,
        base_noise_level * np.exp(-mz_values / decay_constant) * (1 + 0.5 * total_perlin_chem)
    )
    # 2. White electronic noise
    white_noise = white_noise_level * max_intensity
    # 3. Signal-dependent shot noise
    shot_noise = shot_noise_factor * np.sqrt(np.maximum(0, intensities))

    # Combine noise sources and generate random values
    total_noise_stddev = np.sqrt(chemical_noise_level**2 + white_noise**2 + shot_noise**2)
    random_noise = rng.normal(0, total_noise_stddev, size=intensities.shape)

    # 4. Low-frequency baseline wobble
    baseline_wobble = baseline_wobble_amplitude * perlin_wobble

    # 5. Optional Pink Noise
    pink_noise = 0.0
    if pink_noise_enabled:
        np.random.seed(seed + 2) # Re-seed for another different noise pattern
        pink_noise = generate_pink_noise(len(mz_values), pink_noise_level * max_intensity)

    return intensities + random_noise + baseline_wobble + pink_noise
