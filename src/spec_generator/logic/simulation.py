import os
import queue
import multiprocessing
import itertools
import numpy as np

from ..core.lc import apply_lc_profile_and_noise
from ..core.spectrum import generate_protein_spectrum
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename
from ..config import SpectrumGeneratorConfig
from collections import defaultdict


def _spectrum_generation_worker(
    protein_index: int,
    protein_mass: float,
    intensity_scalar: float,
    config: SpectrumGeneratorConfig,
    mz_range: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    A simple worker that calls the main spectrum generation function and returns
    the result along with the index of the protein it belongs to.
    """
    common = config.common
    spectrum = generate_protein_spectrum(
        protein_mass,
        mz_range,
        common.mz_step,
        common.peak_sigma_mz,
        intensity_scalar,
        common.isotopic_enabled,
        common.resolution,
    )
    return protein_index, spectrum


def _run_simulation(
    config: SpectrumGeneratorConfig, update_queue: queue.Queue | None
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """
    Core simulation logic to generate spectra data.
    """
    common = config.common
    lc = config.lc

    if update_queue:
        log_msg = (
            f"  Proteins: {len(config.protein_masses)} ({', '.join(map(str, config.protein_masses))})\n"
            f"  m/z Range: {common.mz_range_start}-{common.mz_range_end}, Step: {common.mz_step}\n"
            f"  Isotopes: {'Enabled' if common.isotopic_enabled else 'Disabled'}, Resolution: {common.resolution/1000}k\n"
            f"  LC Simulation: {'Enabled' if lc.enabled else 'Disabled'} ({lc.num_scans} scans)\n"
            f"  Noise: {common.noise_option}, Seed: {common.seed}\n"
        )
        update_queue.put(("log", log_msg))
        update_queue.put(("progress_set", 5))

    mz_range = np.arange(
        common.mz_range_start, common.mz_range_end + common.mz_step, common.mz_step
    )

    # Flatten the tasks: each call to generate_protein_spectrum becomes one task.
    # This handles both multi-protein and mass inhomogeneity in one parallel step.
    tasks = []
    num_sub_tasks = []
    for i, protein_mass in enumerate(config.protein_masses):
        if config.mass_inhomogeneity > 0:
            num_samples = 7
            mass_distribution = np.random.normal(
                loc=protein_mass, scale=config.mass_inhomogeneity, size=num_samples
            )
            num_sub_tasks.append(num_samples)
            for sub_mass in mass_distribution:
                tasks.append(
                    (
                        i,
                        sub_mass,
                        config.intensity_scalars[i],
                        config,
                        mz_range,
                    )
                )
        else:
            num_sub_tasks.append(1)
            tasks.append(
                (
                    i,
                    protein_mass,
                    config.intensity_scalars[i],
                    config,
                    mz_range,
                )
            )

    if update_queue:
        update_queue.put(
            ("log", f"Generating {len(tasks)} total spectra in parallel...\n")
        )

    with multiprocessing.Pool() as pool:
        results = pool.starmap(_spectrum_generation_worker, tasks)

    # Group and average the results
    grouped_spectra = defaultdict(list)
    for protein_index, spectrum in results:
        grouped_spectra[protein_index].append(spectrum)

    all_clean_spectra = []
    for i in range(len(config.protein_masses)):
        spectra_for_protein = grouped_spectra[i]
        if num_sub_tasks[i] > 1:
            # Average the spectra for mass inhomogeneity
            avg_spectrum = np.sum(spectra_for_protein, axis=0) / num_sub_tasks[i]
            all_clean_spectra.append(avg_spectrum)
        else:
            all_clean_spectra.append(spectra_for_protein[0])

    if not all_clean_spectra:
        if update_queue:
            update_queue.put(
                ("error", "Spectrum generation failed for an unknown reason.")
            )
        return None

    if update_queue:
        update_queue.put(("progress_set", 55))

    apex_scans = None
    if (
        lc.enabled
        and config.hydrophobicity_scores
        and len(config.hydrophobicity_scores) == len(all_clean_spectra)
    ):
        scores = np.array(config.hydrophobicity_scores)
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score == min_score:
            apex_scans = [int(lc.num_scans / 2)] * len(scores)
        else:
            scan_padding = int(lc.num_scans * 0.1)
            usable_scan_range = lc.num_scans - 2 * scan_padding
            scaled_scans = (
                (scores - min_score) / (max_score - min_score) * usable_scan_range
            )
            apex_scans = [int(s + scan_padding) for s in scaled_scans]

    combined_chromatogram = apply_lc_profile_and_noise(
        mz_range=mz_range,
        all_clean_spectra=all_clean_spectra,
        num_scans=lc.num_scans,
        gaussian_std_dev=lc.gaussian_std_dev,
        lc_tailing_factor=lc.lc_tailing_factor,
        seed=common.seed,
        noise_option=common.noise_option,
        pink_noise_enabled=common.pink_noise_enabled,
        apex_scans=apex_scans,
        update_queue=update_queue,
    )
    return mz_range, [combined_chromatogram]


def run_simulation_for_preview(config: SpectrumGeneratorConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Runs the core simulation and returns a single spectrum for previewing.
    """
    simulation_result = _run_simulation(config, update_queue=None)
    if simulation_result:
        mz_range, spectra_data = simulation_result
        # For preview, we just need the apex scan of the first (and only) chromatogram
        apex_scan_index = (config.lc.num_scans - 1) // 2
        return mz_range, spectra_data[0][apex_scan_index]
    return None

def execute_simulation_and_write_mzml(
    config: SpectrumGeneratorConfig,
    final_filepath: str,
    update_queue: queue.Queue | None,
    return_data_only: bool = False,
) -> bool | tuple[np.ndarray, list[np.ndarray]]:
    """
    The main logic for running a simulation and writing the mzML file.
    Orchestrates the spectrum generation, scaling, noise addition, and file writing.
    """
    try:
        if not config.protein_masses or not all(m > 0 for m in config.protein_masses):
            raise ValueError("Protein masses must be positive numbers.")
        if config.common.mz_step <= 0:
            raise ValueError("m/z step must be positive.")
    except ValueError as e:
        if update_queue:
            update_queue.put(('error', f"Invalid parameters: {e}. Please check inputs."))
        return False

    try:
        simulation_result = _run_simulation(config, update_queue)
        if simulation_result is None:
            return False

        mz_range, spectra_for_mzml = simulation_result

        if return_data_only:
            return mz_range, spectra_for_mzml

        if not os.path.basename(final_filepath):
            if update_queue:
                update_queue.put(('error', "Filename template resulted in an empty name."))
            return False

        mzml_content = create_mzml_content_et(
            mz_range, spectra_for_mzml,
            config.lc.scan_interval if config.lc.enabled else 0.0,
            update_queue
        )
        if mzml_content is None:
            return False

        if update_queue:
            update_queue.put(('progress_set', 95))

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
        print(f"An unexpected error occurred: {e}")
        return False
