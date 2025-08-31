import os
import queue
import numpy as np

from ..core.lc import apply_lc_profile_and_noise
from ..core.spectrum import generate_protein_spectrum
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename
from ..config import SpectrumGeneratorConfig


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
        # --- Unpack config for easier access ---
        common = config.common
        lc = config.lc

        # --- Parameter Validation ---
        if not config.protein_masses or not all(m > 0 for m in config.protein_masses):
            raise ValueError("Protein masses must be positive numbers.")
        if common.mz_step <= 0:
            raise ValueError("m/z step must be positive.")
        if common.peak_sigma_mz < 0:
            raise ValueError("Peak sigma cannot be negative.")
        if common.mz_range_start >= common.mz_range_end:
            raise ValueError("m/z range start must be less than the end.")
        if common.resolution < 0:
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

        if update_queue:
            log_msg = (
                f"  Proteins: {len(config.protein_masses)} ({', '.join(map(str, config.protein_masses))})\n"
                f"  m/z Range: {common.mz_range_start}-{common.mz_range_end}, Step: {common.mz_step}\n"
                f"  Isotopes: {'Enabled' if common.isotopic_enabled else 'Disabled'}, Resolution: {common.resolution/1000}k\n"
                f"  LC Simulation: {'Enabled' if lc.enabled else 'Disabled'} ({lc.num_scans} scans)\n"
                f"  Noise: {common.noise_option}, Seed: {common.seed}\n"
            )
            update_queue.put(('log', log_msg))
            update_queue.put(('progress_set', 5))

        # --- Core Simulation Steps ---
        mz_range = np.arange(common.mz_range_start, common.mz_range_end + common.mz_step, common.mz_step)

        all_clean_spectra = []
        num_proteins = len(config.protein_masses)
        progress_per_protein = (50 / num_proteins) if num_proteins > 0 else 0

        for i, protein_mass in enumerate(config.protein_masses):
            if update_queue:
                update_queue.put(('log', f"Generating base spectrum for Protein (Mass: {protein_mass})...\n"))

            if config.mass_inhomogeneity > 0:
                num_samples = 7
                mass_distribution = np.random.normal(loc=protein_mass, scale=config.mass_inhomogeneity, size=num_samples)
                total_spectrum = np.zeros_like(mz_range, dtype=float)
                for sub_mass in mass_distribution:
                    spectrum = generate_protein_spectrum(
                        sub_mass, mz_range, common.mz_step, common.peak_sigma_mz,
                        config.intensity_scalars[i], common.isotopic_enabled, common.resolution
                    )
                    total_spectrum += spectrum
                clean_spectrum = total_spectrum / num_samples
            else:
                clean_spectrum = generate_protein_spectrum(
                    protein_mass, mz_range, common.mz_step, common.peak_sigma_mz,
                    config.intensity_scalars[i], common.isotopic_enabled, common.resolution
                )

            all_clean_spectra.append(clean_spectrum)

            if update_queue:
                update_queue.put(('progress_add', progress_per_protein))

        if not all_clean_spectra:
            if update_queue:
                update_queue.put(('error', "Spectrum generation failed for an unknown reason."))
            return False

        if update_queue:
            update_queue.put(('progress_set', 55))

        # --- LC Simulation and Noise ---
        apex_scans = None
        if lc.enabled and config.hydrophobicity_scores and len(config.hydrophobicity_scores) == len(all_clean_spectra):
            scores = np.array(config.hydrophobicity_scores)
            min_score, max_score = np.min(scores), np.max(scores)

            if max_score == min_score:
                # All species have the same hydrophobicity, elute in the center
                apex_scans = [int(lc.num_scans / 2)] * len(scores)
            else:
                # Scale scores to scan range. Pad by 10% on each side.
                scan_padding = int(lc.num_scans * 0.1)
                usable_scan_range = lc.num_scans - 2 * scan_padding
                scaled_scans = (scores - min_score) / (max_score - min_score) * usable_scan_range
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
            update_queue=update_queue
        )
        # Wrap the single chromatogram in a list to match the expected format for the mzML writer
        spectra_for_mzml = [combined_chromatogram]

        mzml_content = create_mzml_content_et(
            mz_range, spectra_for_mzml,
            lc.scan_interval if lc.enabled else 0.0,
            update_queue
        )
        if mzml_content is None:
            return False

        if update_queue:
            update_queue.put(('progress_set', 95))

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
        print(f"An unexpected error occurred: {e}")
        return False
