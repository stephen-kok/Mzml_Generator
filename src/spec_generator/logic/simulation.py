import os
import queue
import multiprocessing
import itertools
import numpy as np

from ..core.lc import apply_lc_profile_and_noise
from ..core.spectrum import generate_protein_spectrum
from ..core.types import MSMSSpectrum
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename
from ..config import SpectrumGeneratorConfig
from .retention_time import calculate_apex_scans_from_hydrophobicity
from .fragmentation import generate_fragmentation_events
from ..core.types import FragmentationEvent
from collections import defaultdict


class SimulationRunner:
    """
    Encapsulates the logic for running a full simulation from a configuration object.
    """

    def __init__(self, config: SpectrumGeneratorConfig, progress_callback=None):
        """
        Initializes the SimulationRunner.
        Args:
            config: The configuration object for the simulation.
            progress_callback: An optional function to call with progress updates.
                               It should accept (str, Any) where the first element
                               is the type of update (e.g., 'log', 'progress_set')
                               and the second is the value.
        """
        self.config = config
        self.progress_callback = progress_callback or (lambda *args: None)

    def _prepare_simulation_tasks(self, mz_range: np.ndarray) -> tuple[list, list[int]]:
        """Prepares the list of tasks for the multiprocessing pool."""
        tasks = []
        num_sub_tasks = []
        for i, protein_mass in enumerate(self.config.protein_masses):
            if self.config.mass_inhomogeneity > 0:
                num_samples = self.config.mass_inhomogeneity_samples
                mass_distribution = np.random.normal(
                    loc=protein_mass,
                    scale=self.config.mass_inhomogeneity,
                    size=num_samples,
                )
                num_sub_tasks.append(num_samples)
                for sub_mass in mass_distribution:
                    tasks.append(
                        (i, sub_mass, self.config.intensity_scalars[i], self.config, mz_range)
                    )
            else:
                num_sub_tasks.append(1)
                tasks.append(
                    (i, protein_mass, self.config.intensity_scalars[i], self.config, mz_range)
                )
        return tasks, num_sub_tasks

    def _aggregate_simulation_results(
        self, results: list, num_sub_tasks: list[int]
    ) -> list[np.ndarray]:
        """Groups and averages the results from the multiprocessing pool."""
        grouped_spectra = defaultdict(list)
        for protein_index, spectrum in results:
            grouped_spectra[protein_index].append(spectrum)

        all_clean_spectra = []
        for i in range(len(self.config.protein_masses)):
            spectra_for_protein = grouped_spectra[i]
            if num_sub_tasks[i] > 1:
                avg_spectrum = np.sum(spectra_for_protein, axis=0) / num_sub_tasks[i]
                all_clean_spectra.append(avg_spectrum)
            else:
                all_clean_spectra.append(spectra_for_protein[0])
        return all_clean_spectra

    def run(self) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """
        Executes the core simulation logic to generate spectra data.
        """
        common = self.config.common
        lc = self.config.lc

        log_msg = (
            f"  Proteins: {len(self.config.protein_masses)} ({', '.join(map(str, self.config.protein_masses))})\n"
            f"  m/z Range: {common.mz_range_start}-{common.mz_range_end}, Step: {common.mz_step}\n"
            f"  Isotopes: {'Enabled' if common.isotopic_enabled else 'Disabled'}, Resolution: {common.resolution/1000}k\n"
            f"  LC Simulation: {'Enabled' if lc.enabled else 'Disabled'} ({lc.num_scans} scans)\n"
            f"  Noise: {common.noise_option}, Seed: {common.seed}\n"
        )
        self.progress_callback("log", log_msg)
        self.progress_callback("progress_set", 5)

        mz_range = np.arange(
            common.mz_range_start, common.mz_range_end + common.mz_step, common.mz_step
        )

        tasks, num_sub_tasks = self._prepare_simulation_tasks(mz_range)

        self.progress_callback(
            "log", f"Generating {len(tasks)} total spectra in parallel...\n"
        )

        with multiprocessing.Pool() as pool:
            results = pool.starmap(_spectrum_generation_worker, tasks)

        all_clean_spectra = self._aggregate_simulation_results(results, num_sub_tasks)

        if not all_clean_spectra:
            self.progress_callback(
                "error", "Spectrum generation failed for an unknown reason."
            )
            return None

        self.progress_callback("progress_set", 55)

        apex_scans = None
        if (
            lc.enabled
            and self.config.hydrophobicity_scores
            and len(self.config.hydrophobicity_scores) == len(all_clean_spectra)
        ):
            apex_scans = calculate_apex_scans_from_hydrophobicity(
                scores=self.config.hydrophobicity_scores,
                num_scans=lc.num_scans,
                retention_time_model=lc.retention_time_model,
                rpc_hydrophobicity_coefficient=lc.rpc_hydrophobicity_coefficient,
            )

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
            progress_callback=self.progress_callback,
        )
        return mz_range, [combined_chromatogram]


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


def execute_simulation_and_write_mzml(
    config: SpectrumGeneratorConfig,
    final_filepath: str,
    progress_callback=None,
    return_data_only: bool = False,
) -> bool | tuple[np.ndarray, list[np.ndarray]]:
    """
    The main logic for running a simulation and writing the mzML file.
    Orchestrates the spectrum generation, scaling, noise addition, and file writing.
    """
    progress_callback = progress_callback or (lambda *args: None)
    try:
        if not config.protein_masses or not all(m > 0 for m in config.protein_masses):
            raise ValueError("Protein masses must be positive numbers.")
        if config.common.mz_step <= 0:
            raise ValueError("m/z step must be positive.")
    except ValueError as e:
        progress_callback('error', f"Invalid parameters: {e}. Please check inputs.")
        return False

    try:
        runner = SimulationRunner(config, progress_callback)
        simulation_result = runner.run()

        if simulation_result is None:
            return False

        mz_range, spectra_for_mzml = simulation_result

        if return_data_only:
            return mz_range, spectra_for_mzml

        if not os.path.basename(final_filepath):
            progress_callback('error', "Filename template resulted in an empty name.")
            return False

        # --- MS/MS Data Generation ---
        msms_spectra = []
        if config.common.msms_enabled and config.peptide_sequences:
            self.progress_callback("log", "Generating MS/MS spectra...\n")
            for seq in config.peptide_sequences:
                # Loop through configured precursor charges
                for precursor_charge in config.common.msms_precursor_charges:
                    events = generate_fragmentation_events(
                        sequence=seq,
                        precursor_charge=precursor_charge,
                        ion_types=config.common.msms_ion_types,
                        fragment_charges=config.common.msms_fragment_charges,
                        rt=30.0,  # dummy rt
                        precursor_intensity=1e5,  # dummy intensity
                        config=config,
                    )
                    if events:
                        msms_spectra.append(
                            MSMSSpectrum(
                                rt=30.0,
                                precursor_mz=events[0].precursor_mz,
                                precursor_charge=precursor_charge,
                                fragmentation_events=events,
                            )
                        )

        mzml_content = create_mzml_content_et(
            mz_range,
            spectra_for_mzml,
            config.lc.scan_interval if config.lc.enabled else 0.0,
            progress_callback,
            msms_spectra=msms_spectra,
        )
        if mzml_content is None:
            return False

        progress_callback('progress_set', 95)

        unique_filepath = create_unique_filename(final_filepath)
        os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)

        progress_callback('log', f"Writing mzML file to: {os.path.basename(unique_filepath)}\n")

        with open(unique_filepath, "wb") as outfile:
            outfile.write(mzml_content)

        progress_callback('log', "File successfully created.\n\n")
        progress_callback('progress_set', 100)

        return True

    except Exception as e:
        progress_callback('error', f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        return False
