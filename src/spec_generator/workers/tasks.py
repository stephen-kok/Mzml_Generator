from datetime import datetime
import os
import threading
from typing import Any, Optional
from numpy.random import default_rng
import numpy as np

from ..logic.simulation import execute_simulation_and_write_mzml
from ..logic.binding import execute_binding_simulation
from ..core.isotopes import isotope_calculator
from ..utils.file_io import format_filename
from ..config import SpectrumGeneratorConfig, CovalentBindingConfig
from .worker_init import _stop_event_worker


def run_simulation_task(config: SpectrumGeneratorConfig, return_data_only: bool = False) -> tuple[bool, str] | tuple[np.ndarray, list[np.ndarray]] | None:
    """
    A top-level function that runs in a separate process for the Spectrum Generator.
    This function is designed to be called by `multiprocessing.Pool`.

    Args:
        config: A SpectrumGeneratorConfig object for a single protein simulation.
        return_data_only: If True, returns the raw data instead of writing a file.
    """
    if _stop_event_worker and _stop_event_worker.is_set():
        return None
    try:
        # This function is currently only used by the "General" and "Antibody" tabs,
        # which generate a single mzML file from a single config.
        protein_mass = config.protein_masses[0] if config.protein_masses else 0
        scalar = config.intensity_scalars[0] if config.intensity_scalars else 0

        _, most_abundant_offset = isotope_calculator.get_distribution(protein_mass)
        effective_protein_mass = protein_mass + most_abundant_offset

        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": datetime.now().strftime('%H%M%S'),
            "protein_mass": int(round(effective_protein_mass)),
            "scalar": scalar,
            "scans": config.lc.num_scans,
            "noise": config.common.noise_option.replace(" ", ""),
            "seed": config.common.seed,
            "num_proteins": len(config.protein_masses)
        }
        filename = format_filename(config.common.filename_template, placeholders)
        filepath = os.path.join(config.common.output_directory, filename)

        result = execute_simulation_and_write_mzml(
            config=config,
            final_filepath=filepath,
            progress_callback=None, # No callback needed for batch mode
            return_data_only=return_data_only,
            stop_event=_stop_event_worker
        )

        if _stop_event_worker and _stop_event_worker.is_set():
            return (False, f"--- CANCELLED generation for Protein {int(protein_mass)} Da ---\n")

        if return_data_only:
            return result  # type: ignore

        if result:
            return (True, f"--- Successfully generated file for Protein ~{int(round(effective_protein_mass))} Da ---\n")
        else:
            return (False, f"--- FAILED to generate file for Protein {int(protein_mass)} Da ---\n")

    except Exception as e:
        return (False, f"--- Critical error for Protein {int(config.protein_masses[0])} Da: {e} ---\n")


def run_binding_task(args: tuple[Any, ...]) -> tuple[bool, str] | tuple[np.ndarray, list[np.ndarray]] | None:
    """
    A top-level function that runs in a separate process for the Covalent Binding tab.
    This function is designed to be called by `multiprocessing.Pool`.

    Args:
        args: A tuple containing (compound_name, compound_mass, config, return_data_only).
    """
    if _stop_event_worker and _stop_event_worker.is_set():
        return None
    try:
        (compound_name, compound_mass, config, return_data_only) = args
        common = config.common

        _, most_abundant_offset = isotope_calculator.get_distribution(config.protein_avg_mass)
        effective_protein_mass = config.protein_avg_mass + most_abundant_offset
        rng_scenario, rng_intensity = default_rng(common.seed), default_rng(common.seed + 1)
        total_binding, dar2_of_bound, desc = 0.0, 0.0, "No Binding"

        if rng_scenario.random() < config.prob_binding:
            total_binding = rng_intensity.uniform(*config.total_binding_range)
            desc = f"Binding ({total_binding:.2f}%)"
            if rng_scenario.random() < config.prob_dar2:
                dar2_of_bound = rng_intensity.uniform(*config.dar2_range)
                desc += f", DAR-2 ({dar2_of_bound:.2f}% of bound)"

        message = f"--- Processing: {compound_name} | Scenario: {desc} ---\n"
        if _stop_event_worker and _stop_event_worker.is_set():
            return (False, message + "  CANCELLED\n")

        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": datetime.now().strftime('%H%M%S'),
            "compound_name": compound_name,
            "protein_mass": int(round(effective_protein_mass)),
            "scans": config.lc.num_scans,
            "noise": common.noise_option.replace(" ", ""),
            "seed": common.seed
        }
        filename = format_filename(common.filename_template, placeholders)
        filepath = os.path.join(common.output_directory, filename)

        result, final_filepath = execute_binding_simulation(
            config=config,
            compound_mass=compound_mass,
            total_binding_percentage=total_binding,
            dar2_percentage_of_bound=dar2_of_bound,
            filepath=filepath,
            return_data_only=return_data_only,
            stop_event=_stop_event_worker
        )

        if _stop_event_worker and _stop_event_worker.is_set():
             return (False, message + "  CANCELLED\n")

        if return_data_only:
            return result, final_filepath # This is a tuple (mz_range, spectra)

        if result:
            message += f"  SUCCESS: Wrote to {os.path.basename(final_filepath)}\n"
            return (True, message)
        else:
            message += "  ERROR: Spectrum generation or file write failed.\n"
            return (False, message)

    except Exception as e:
        return (False, f"--- Critical error for compound {args[0]}: {e} ---\n")
