"""
This module orchestrates the full peptide map simulation process.
"""
import numpy as np
from pyteomics import mass
import multiprocessing
import itertools
import threading
from typing import Optional

from .peptide import digest_sequence
from .retention_time import predict_retention_times
from .charge import predict_charge_states
from ..core.peptide_isotopes import peptide_isotope_calculator
from ..core.spectrum import generate_peptide_spectrum
from ..core.lc import _get_lc_peak_shape # Re-using the private LC shape function
from ..core.noise import add_noise
from ..utils.mzml import create_mzml_content_et
from ..config import PeptideMapSimConfig
from ..core.constants import noise_presets
from ..workers.worker_init import init_worker, _stop_event_worker


def _generate_spectrum_for_peptide_worker(
    peptide: str,
    config: PeptideMapSimConfig,
    mz_range: np.ndarray,
    peak_sigma_mz: float,
) -> tuple[str, np.ndarray] | None:
    """
    Generates a single peptide's spectrum. Designed to be called by a multiprocessing Pool.
    """
    if _stop_event_worker and _stop_event_worker.is_set():
        return None
    total_spectrum = np.zeros_like(mz_range)
    base_intensity = 1000.0  # Placeholder intensity

    if config.predict_charge:
        charge_distribution = predict_charge_states(peptide)
        for charge, rel_intensity in charge_distribution.items():
            if _stop_event_worker and _stop_event_worker.is_set():
                return None
            spectrum_for_charge = generate_peptide_spectrum(
                peptide_sequence=peptide,
                mz_range=mz_range,
                peak_sigma_mz_float=peak_sigma_mz,
                intensity_scalar=base_intensity * rel_intensity,
                resolution=config.common.resolution,
                charge=charge,
            )
            total_spectrum += spectrum_for_charge
    else:
        total_spectrum = generate_peptide_spectrum(
            peptide_sequence=peptide,
            mz_range=mz_range,
            peak_sigma_mz_float=peak_sigma_mz,
            intensity_scalar=base_intensity,
            resolution=config.common.resolution,
            charge=config.charge_state,
        )
    return peptide, total_spectrum


def execute_peptide_map_simulation(
    config: PeptideMapSimConfig,
    final_filepath: str,
    progress_callback=None,
    return_data_only: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    """
    Orchestrates the full peptide map simulation.
    """
    progress_callback = progress_callback or (lambda *args: None)
    try:
        progress_callback('log', "Starting peptide map simulation...\n")
        progress_callback('progress_set', 5)

        # 1. Digestion
        if stop_event and stop_event.is_set(): return False
        progress_callback('log', f"Performing in-silico digestion of sequence...\n")
        peptides = digest_sequence(
            config.sequence,
            missed_cleavages=config.missed_cleavages
        )
        progress_callback('log', f"Generated {len(peptides)} unique peptides.\n")
        progress_callback('progress_set', 15)

        # 2. Retention Time Prediction
        if stop_event and stop_event.is_set(): return False
        progress_callback('log', "Predicting retention times...\n")
        retention_times = predict_retention_times(peptides, total_run_time_minutes=config.lc.run_time)
        progress_callback('progress_set', 25)

        # 3. Generate Base Spectra for all peptides
        if stop_event and stop_event.is_set(): return False
        progress_callback(
            "log", f"Generating base spectra for {len(peptides)} peptides (in parallel)...\n"
        )

        mz_range = np.arange(
            config.common.mz_range_start,
            config.common.mz_range_end,
            config.common.mz_step,
        )
        peak_sigma_mz = config.common.mz_range_end / (
            config.common.resolution * 2.355
        )

        tasks = zip(
            peptides,
            itertools.repeat(config),
            itertools.repeat(mz_range),
            itertools.repeat(peak_sigma_mz),
        )
        results = []
        try:
            with multiprocessing.Pool(initializer=init_worker, initargs=(stop_event,)) as pool:
                async_result = pool.starmap_async(_generate_spectrum_for_peptide_worker, tasks)
                while not async_result.ready():
                    if stop_event and stop_event.is_set():
                        progress_callback("log", "Cancellation received, terminating workers...\n")
                        pool.terminate()
                        pool.join()
                        return False
                    async_result.wait(timeout=0.1)
                results = async_result.get()
        except Exception as e:
            progress_callback("error", f"A multiprocessing error occurred: {e}")
            return False

        if stop_event and stop_event.is_set(): return False

        results = [r for r in results if r is not None]
        if not results:
            progress_callback("log", "Spectrum generation cancelled or failed.\n")
            return False

        peptide_data = [
            {"sequence": peptide, "rt": retention_times[peptide], "spectrum": spectrum}
            for peptide, spectrum in results
        ]
        progress_callback('progress_set', 45)

        # 4. Construct Chromatogram
        if stop_event and stop_event.is_set(): return False
        progress_callback('log', "Constructing chromatogram...\n")

        scan_interval_minutes = config.lc.scan_interval / 60.0
        num_scans = int(config.lc.run_time / scan_interval_minutes)
        final_scans = [np.zeros_like(mz_range) for _ in range(num_scans)]

        lc_peak_scans = int(config.lc.peak_width_seconds / config.lc.scan_interval)
        apex_index = (lc_peak_scans - 1) / 2.0
        std_dev_scans = lc_peak_scans / 6
        tau = 1.0
        lc_shape = _get_lc_peak_shape(lc_peak_scans, apex_index, std_dev_scans, tau)

        for p_data in peptide_data:
            if stop_event and stop_event.is_set(): return False
            apex_scan = int(p_data["rt"] / scan_interval_minutes)
            start_scan = apex_scan - (lc_peak_scans // 2)

            for i in range(lc_peak_scans):
                scan_idx = start_scan + i
                if 0 <= scan_idx < num_scans:
                    final_scans[scan_idx] += p_data["spectrum"] * lc_shape[i]

        progress_callback('progress_set', 70)

        # 5. Add Noise
        if stop_event and stop_event.is_set(): return False
        if config.common.noise_option != "No Noise":
            progress_callback('log', "Adding noise...\n")
            noise_params = noise_presets.get(config.common.noise_option)
            if noise_params:
                max_intensity = max(np.max(s) for s in final_scans if s.size > 0)
                for i, scan in enumerate(final_scans):
                    if stop_event and stop_event.is_set(): return False
                    final_scans[i] = add_noise(
                        mz_values=mz_range,
                        intensities=scan,
                        max_intensity=max_intensity,
                        seed=config.common.seed + i,
                        pink_noise_enabled=config.common.pink_noise,
                        **noise_params
                    )

        progress_callback('progress_set', 85)

        # 6. Write mzML
        if stop_event and stop_event.is_set(): return False
        if return_data_only:
            return (mz_range, final_scans)

        mzml_bytes = create_mzml_content_et(
            mz_range=mz_range,
            run_data=[final_scans],
            scan_interval=scan_interval_minutes,
            progress_callback=progress_callback,
            stop_event=stop_event,
        )

        if mzml_bytes:
            with open(final_filepath, 'wb') as f:
                f.write(mzml_bytes)
            progress_callback('log', f"Successfully wrote mzML file to {final_filepath}\n")
            progress_callback('progress_set', 100)
            progress_callback('finished', '')
            return True
        else:
            raise ValueError("mzML content generation failed.")

    except Exception as e:
        progress_callback('error', f"Peptide map simulation failed: {e}")
        print(f"Peptide map simulation failed: {e}")
        return False
