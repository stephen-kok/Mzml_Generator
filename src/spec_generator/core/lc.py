import math
import numpy as np
from numpy.random import default_rng
from scipy.special import erfc

from .constants import noise_presets
from .noise import add_noise
from .spectrum import generate_protein_spectrum


def generate_scaled_spectra(
    protein_masses_list: list[float],
    mz_range: np.ndarray,
    mz_step_float: float,
    peak_sigma_mz_float: float,
    intensity_scalars: list[float],
    isotopic_enabled: bool,
    resolution: float,
    progress_callback=None,
) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    """
    Generates a list of clean (noiseless) base spectra for multiple proteins.
    """
    progress_callback = progress_callback or (lambda *args: None)
    all_clean_spectra = []
    num_proteins = len(protein_masses_list)
    progress_per_protein = (50 / num_proteins) if num_proteins > 0 else 0

    try:
        for i, protein_mass in enumerate(protein_masses_list):
            progress_callback('log', f"Generating base spectrum for Protein (Mass: {protein_mass})...\n")

            spectrum = generate_protein_spectrum(
                protein_mass, mz_range, mz_step_float, peak_sigma_mz_float,
                intensity_scalars[i], isotopic_enabled, resolution
            )
            all_clean_spectra.append(spectrum)

            progress_callback('progress_add', progress_per_protein)
        return mz_range, all_clean_spectra
    except Exception as e:
        progress_callback('error', f"Error during base spectrum generation: {e}")
        return None, None


def _get_lc_peak_shape(num_scans: int, apex_scan_index: float, std_dev: float, tau: float) -> np.ndarray:
    """
    Generates an LC peak profile, either Gaussian or Exponentially Modified Gaussian (EMG).
    """
    scan_index_values = np.arange(num_scans)

    if tau <= 1e-6:  # Treat as pure Gaussian for negligible tau
        return np.exp(-((scan_index_values - apex_scan_index) ** 2) / (2 * max(1e-6, std_dev) ** 2))
    else: # Exponentially Modified Gaussian
        # Clamp values to avoid math errors with very small inputs
        tau = max(1e-6, tau)
        std_dev = max(1e-6, std_dev)

        arg = (std_dev**2 - tau * (scan_index_values - apex_scan_index)) / (math.sqrt(2) * std_dev * tau)
        emg = (std_dev / tau) * math.sqrt(math.pi / 2) * np.exp(0.5 * (std_dev / tau)**2 - (scan_index_values - apex_scan_index) / tau) * erfc(arg)

        # Normalize the peak to a maximum of 1.0
        max_val = np.max(emg)
        return emg / max_val if max_val > 0 else emg


def apply_lc_profile_and_noise(
    mz_range: np.ndarray,
    all_clean_spectra: list[np.ndarray],
    num_scans: int,
    gaussian_std_dev: float,
    lc_tailing_factor: float,
    seed: int,
    noise_option: str,
    pink_noise_enabled: bool,
    apex_scans: list[int] | None = None,
    progress_callback=None,
) -> list[np.ndarray]:
    """
    Applies an LC peak shape to a list of base spectra and combines them into
    a single chromatogram. Noise is applied to the final combined data.
    """
    progress_callback = progress_callback or (lambda *args: None)
    if num_scans <= 0:
        progress_callback('error', "Number of scans must be positive.")
        return []

    # If apex_scans aren't provided, all species elute at the center
    if apex_scans is None:
        center_scan = (num_scans - 1) / 2.0
        apex_scans = [center_scan] * len(all_clean_spectra)

    rng = default_rng(seed)
    # This will hold the final data: a list of 1D arrays (scans)
    final_chromatogram = [np.zeros_like(mz_range, dtype=float) for _ in range(num_scans)]
    total_max_intensity = 0

    progress_callback('log', "Applying LC profiles and combining spectra...\n")

    num_proteins = len(all_clean_spectra)
    progress_per_protein = 40 / num_proteins if num_proteins > 0 else 0

    # Generate and apply LC profile for each species, adding it to the final chromatogram
    for i, base_spectrum in enumerate(all_clean_spectra):
        apex_scan_index = apex_scans[i]
        lc_scaling_factors = _get_lc_peak_shape(num_scans, apex_scan_index, gaussian_std_dev, lc_tailing_factor)

        # Apply the unique LC profile to the current base spectrum
        for scan_idx in range(num_scans):
            # Add some random variation per scan to make it more realistic
            scan_variation = rng.normal(loc=1.0, scale=0.05, size=len(mz_range))
            scaled_intensity = base_spectrum * lc_scaling_factors[scan_idx] * scan_variation
            final_chromatogram[scan_idx] += scaled_intensity

        # Keep track of the max intensity for noise calculation
        max_intensity_for_species = np.max(base_spectrum) if base_spectrum.size > 0 else 0
        if max_intensity_for_species > total_max_intensity:
            total_max_intensity = max_intensity_for_species

        progress_callback('progress_add', progress_per_protein)

    # Apply noise to the combined chromatogram
    progress_callback('log', "Applying scan-level noise to combined chromatogram...\n")

    if noise_option != "No Noise":
        noise_params = noise_presets.get(noise_option)
        if noise_params:
            for scan_idx in range(num_scans):
                final_chromatogram[scan_idx] = add_noise(
                    mz_values=mz_range,
                    intensities=final_chromatogram[scan_idx],
                    min_noise_level=0.01,
                    max_intensity=total_max_intensity,
                    pink_noise_enabled=pink_noise_enabled,
                    seed=seed + scan_idx,
                    **noise_params
                )
        else:
            progress_callback('log', f"Warning: Noise preset '{noise_option}' not found. Skipping noise.\n")

    # Add baseline offset and ensure non-negative intensities
    baseline_offset = 10.0
    final_noisy_chromatogram = [np.maximum(0, scan + baseline_offset) for scan in final_chromatogram]

    return final_noisy_chromatogram
