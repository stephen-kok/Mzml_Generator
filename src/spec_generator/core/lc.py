import queue
import numpy as np
from numpy.random import default_rng

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
    update_queue: queue.Queue | None = None,
) -> tuple[np.ndarray | None, list[np.ndarray] | None]:
    """
    Generates a list of clean (noiseless) base spectra for multiple proteins.
    """
    all_clean_spectra = []
    num_proteins = len(protein_masses_list)
    progress_per_protein = (50 / num_proteins) if num_proteins > 0 else 0

    try:
        for i, protein_mass in enumerate(protein_masses_list):
            if update_queue:
                update_queue.put(('log', f"Generating base spectrum for Protein (Mass: {protein_mass})...\n"))

            spectrum = generate_protein_spectrum(
                protein_mass, mz_range, mz_step_float, peak_sigma_mz_float,
                intensity_scalars[i], isotopic_enabled, resolution
            )
            all_clean_spectra.append(spectrum)

            if update_queue:
                update_queue.put(('progress_add', progress_per_protein))
        return mz_range, all_clean_spectra
    except Exception as e:
        if update_queue:
            update_queue.put(('error', f"Error during base spectrum generation: {e}"))
        return None, None


def generate_gaussian_scaled_spectra(
    mz_range: np.ndarray,
    all_clean_spectra: list[np.ndarray],
    num_scans: int,
    gaussian_std_dev: float,
    seed: int,
    noise_option: str,
    update_queue: queue.Queue | None = None,
) -> list[list[np.ndarray]]:
    """
    Applies a Gaussian LC peak shape to a list of base spectra over a
    specified number of scans and adds noise to each scan.
    """
    if num_scans <= 0:
        if update_queue:
            update_queue.put(('log', "Error: Number of scans must be positive.\n"))
        return []

    min_noise_level = 0.01
    baseline_offset = 10.0
    apex_scan_index = (num_scans - 1) / 2.0
    scan_index_values = np.arange(num_scans)
    # Clamp std dev to avoid division by zero
    lc_scaling_factors = np.exp(-((scan_index_values - apex_scan_index)**2) / (2 * max(1e-6, gaussian_std_dev)**2))

    gaussian_scaled_spectra_all_proteins = []
    rng = default_rng(seed)

    num_proteins = len(all_clean_spectra)
    progress_per_scan = (40 / (num_proteins * num_scans)) if (num_proteins * num_scans) > 0 else 0

    if update_queue:
        update_queue.put(('log', "Applying LC profile and scan-level noise...\n"))

    for protein_idx, base_spectrum in enumerate(all_clean_spectra):
        spectra_for_protein = []
        max_intensity_for_noise = np.max(base_spectrum) if base_spectrum.size > 0 else 0

        for scan_idx in range(num_scans):
            # Apply LC peak shape scaling and add some random variation per scan
            scaled_intensity_array = base_spectrum * lc_scaling_factors[scan_idx] * rng.normal(loc=1.0, scale=0.05, size=len(mz_range))

            if noise_option != "No Noise":
                noise_params = noise_presets.get(noise_option)
                if noise_params:
                    # Note: The original add_noise signature had two unused decay constants.
                    # They have been removed from the refactored function.
                    scaled_intensity_array = add_noise(
                        mz_values=mz_range,
                        intensities=scaled_intensity_array,
                        min_noise_level=min_noise_level,
                        max_intensity=max_intensity_for_noise,
                        seed=seed + protein_idx * num_scans + scan_idx,
                        **noise_params
                    )
                elif update_queue:
                    update_queue.put(('log', f"Warning: Noise preset '{noise_option}' not found. Skipping noise.\n"))

            # Add baseline offset and ensure non-negative intensities
            spectra_for_protein.append(np.maximum(0, scaled_intensity_array + baseline_offset))

            if update_queue:
                update_queue.put(('progress_add', progress_per_scan))

        gaussian_scaled_spectra_all_proteins.append(spectra_for_protein)

    return gaussian_scaled_spectra_all_proteins
