import os
import queue
import numpy as np

from ..core.lc import apply_lc_profile_and_noise
from ..core.spectrum import generate_protein_spectrum
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename


def execute_simulation_and_write_mzml(
    protein_masses_str: str,
    mz_step_str: str,
    peak_sigma_mz_str: str,
    mz_range_start_str: str,
    mz_range_end_str: str,
    intensity_scalars: list[float],
    noise_option: str,
    seed: int,
    lc_simulation_enabled: bool,
    num_scans: int,
    scan_interval: float,
    gaussian_std_dev: float,
    lc_tailing_factor: float,
    final_filepath: str,
    isotopic_enabled: bool,
    resolution: float,
    mass_inhomogeneity: float,
    pink_noise_enabled: bool,
    update_queue: queue.Queue | None,
    return_data_only: bool = False,
) -> bool | tuple[np.ndarray, list[np.ndarray]]:
    """
    The main logic for running a simulation and writing the mzML file.
    Orchestrates the spectrum generation, scaling, noise addition, and file writing.
    """
    try:
        # --- Parameter Validation ---
        protein_masses_list = [float(m.strip()) for m in protein_masses_str.split(",")]
        mz_range_start_f = float(mz_range_start_str)
        mz_range_end_f = float(mz_range_end_str)
        mz_step_float = float(mz_step_str)
        peak_sigma_mz_float = float(peak_sigma_mz_str)

        if not protein_masses_list or not all(m > 0 for m in protein_masses_list):
            raise ValueError("Protein masses must be positive numbers.")
        if mz_step_float <= 0:
            raise ValueError("m/z step must be positive.")
        if peak_sigma_mz_float < 0:
            raise ValueError("Peak sigma cannot be negative.")
        if mz_range_start_f >= mz_range_end_f:
            raise ValueError("m/z range start must be less than the end.")
        if resolution < 0:
            raise ValueError("Resolution cannot be negative.")

    except ValueError as e:
        if update_queue:
            update_queue.put(('error', f"Invalid parameters: {e}. Please check inputs."))
        return False

    try:
        if not os.path.basename(final_filepath):
            if update_queue:
                update_queue.put(('error', "Filename template resulted in an empty name."))
            return False

        scans_to_gen = num_scans if lc_simulation_enabled else 1

        if update_queue:
            log_msg = (
                f"  Proteins: {len(protein_masses_list)} ({protein_masses_str})\n"
                f"  m/z Range: {mz_range_start_f}-{mz_range_end_f}, Step: {mz_step_float}\n"
                f"  Isotopes: {'Enabled' if isotopic_enabled else 'Disabled'}, Resolution: {resolution/1000}k\n"
                f"  LC Simulation: {'Enabled' if lc_simulation_enabled else 'Disabled'} ({scans_to_gen} scans)\n"
                f"  Noise: {noise_option}, Seed: {seed}\n"
            )
            update_queue.put(('log', log_msg))
            update_queue.put(('progress_set', 5))

        # --- Core Simulation Steps ---
        mz_range = np.arange(mz_range_start_f, mz_range_end_f + mz_step_float, mz_step_float)

        all_clean_spectra = []
        num_proteins = len(protein_masses_list)
        progress_per_protein = (50 / num_proteins) if num_proteins > 0 else 0

        for i, protein_mass in enumerate(protein_masses_list):
            if update_queue:
                update_queue.put(('log', f"Generating base spectrum for Protein (Mass: {protein_mass})...\n"))

            if mass_inhomogeneity > 0:
                # Simulate a distribution of masses to model peak broadening
                num_samples = 7  # Use a small number of samples for performance
                mass_distribution = np.random.normal(loc=protein_mass, scale=mass_inhomogeneity, size=num_samples)

                total_spectrum = np.zeros_like(mz_range, dtype=float)
                for sub_mass in mass_distribution:
                    # We will need to import generate_protein_spectrum for this
                    spectrum = generate_protein_spectrum(
                        sub_mass, mz_range, mz_step_float, peak_sigma_mz_float,
                        intensity_scalars[i], isotopic_enabled, resolution
                    )
                    total_spectrum += spectrum
                # Average the spectra from the distribution
                clean_spectrum = total_spectrum / num_samples
            else:
                # Original behavior: simulate a single, perfectly defined mass
                clean_spectrum = generate_protein_spectrum(
                    protein_mass, mz_range, mz_step_float, peak_sigma_mz_float,
                    intensity_scalars[i], isotopic_enabled, resolution
                )

            all_clean_spectra.append(clean_spectrum)

            if update_queue:
                update_queue.put(('progress_add', progress_per_protein))

        if not all_clean_spectra:
            # This case should ideally not be reached if validation is correct
            if update_queue:
                update_queue.put(('error', "Spectrum generation failed for an unknown reason."))
            return False

        if update_queue:
            update_queue.put(('progress_set', 55))

        spectra_for_mzml = apply_lc_profile_and_noise(
            mz_range, all_clean_spectra, scans_to_gen, gaussian_std_dev,
            lc_tailing_factor, seed, noise_option, pink_noise_enabled, update_queue
        )

        mzml_content = create_mzml_content_et(
            mz_range, spectra_for_mzml,
            scan_interval if lc_simulation_enabled else 0.0,
            update_queue
        )
        if mzml_content is None:
            return False

        if update_queue:
            update_queue.put(('progress_set', 95))

        # --- File Writing ---
        unique_filepath = create_unique_filename(final_filepath)
        os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)

        if update_queue:
            update_queue.put(('log', f"Writing mzML file to: {os.path.basename(unique_filepath)}\n"))

        if return_data_only:
            return mz_range, spectra_for_mzml

        # --- File Writing ---
        unique_filepath = create_unique_filename(final_filepath)
        os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)

        if update_queue:
            update_queue.put(('log', f"Writing mzML file to: {os.path.basename(unique_filepath)}\n"))

        with open(unique_filepath, "wb") as outfile:
            outfile.write(mzml_content)

        if update_queue:
            update_queue.put(('log', "File successfully created.\n\n"))
            update_queue.put(('progress_set', 100))

        return True

    except Exception as e:
        if update_queue:
            update_queue.put(('error', f"An unexpected error occurred: {e}"))
        # Log to console as well for multiprocessing case where queue is not available
        print(f"An unexpected error occurred: {e}")
        return False
