import os
import queue
import numpy as np

from ..core.lc import generate_scaled_spectra, generate_gaussian_scaled_spectra
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
    final_filepath: str,
    isotopic_enabled: bool,
    resolution: float,
    update_queue: queue.Queue | None,
) -> bool:
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

        _, all_clean_spectra = generate_scaled_spectra(
            protein_masses_list, mz_range, mz_step_float, peak_sigma_mz_float,
            intensity_scalars, isotopic_enabled, resolution, update_queue
        )
        if all_clean_spectra is None:  # An error occurred in generate_scaled_spectra
            return False

        if update_queue:
            update_queue.put(('progress_set', 55))

        spectra_for_mzml = generate_gaussian_scaled_spectra(
            mz_range, all_clean_spectra, scans_to_gen, gaussian_std_dev,
            seed, noise_option, update_queue
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
