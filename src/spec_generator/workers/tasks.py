from datetime import datetime
import os
from numpy.random import default_rng

from ..logic.simulation import execute_simulation_and_write_mzml
from ..logic.binding import execute_binding_simulation
from ..core.isotopes import isotope_calculator
from ..utils.file_io import format_filename


def run_simulation_task(args: tuple) -> tuple[bool, str]:
    """
    A top-level function that runs in a separate process for the Spectrum Generator.
    This function is designed to be called by `multiprocessing.Pool`.

    Args:
        args: A tuple containing (mass, scalar, common_params, file_seed).
    """
    try:
        (mass, scalar, common_params, file_seed) = args

        # Calculate effective protein mass for filename
        _, most_abundant_offset = isotope_calculator.get_distribution(mass)
        effective_protein_mass = mass + most_abundant_offset

        # Prepare placeholders for filename formatting
        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": datetime.now().strftime('%H%M%S'),
            "protein_mass": int(round(effective_protein_mass)),
            "scalar": scalar,
            "scans": common_params['num_scans'],
            "noise": common_params['noise_option'].replace(" ", ""),
            "seed": file_seed,
            "num_proteins": 1
        }
        filename = format_filename(common_params['filename_template'], placeholders)
        filepath = os.path.join(common_params['output_directory'], filename)

        # Delegate the core work to the simulation logic function
        success = execute_simulation_and_write_mzml(
            protein_masses_str=str(mass),
            mz_step_str=common_params['mz_step'],
            peak_sigma_mz_str=common_params['peak_sigma_mz'],
            mz_range_start_str=common_params['mz_range_start'],
            mz_range_end_str=common_params['mz_range_end'],
            intensity_scalars=[scalar],
            noise_option=common_params['noise_option'],
            seed=file_seed,
            lc_simulation_enabled=common_params['lc_simulation_enabled'],
            num_scans=common_params['num_scans'],
            scan_interval=common_params['scan_interval'],
            gaussian_std_dev=common_params['gaussian_std_dev'],
            final_filepath=filepath,
            isotopic_enabled=common_params['isotopic_enabled'],
            resolution=common_params['resolution'],
            update_queue=None  # No queue for multiprocessing workers, just return status
        )

        if success:
            return (True, f"--- Successfully generated file for Protein ~{int(round(effective_protein_mass))} Da ---\n")
        else:
            return (False, f"--- FAILED to generate file for Protein {int(mass)} Da ---\n")

    except Exception as e:
        # Catch any unexpected errors in the worker process
        return (False, f"--- Critical error for Protein {int(args[0])} Da: {e} ---\n")


def run_binding_task(args: tuple) -> tuple[bool, str]:
    """
    A top-level function that runs in a separate process for the Covalent Binding tab.
    This function is designed to be called by `multiprocessing.Pool`.

    Args:
        args: A tuple containing (compound_name, compound_mass, protein_avg_mass, common_params, file_seed).
    """
    try:
        (compound_name, compound_mass, protein_avg_mass, common_params, file_seed) = args

        # Determine binding scenario randomly for this specific task
        _, most_abundant_offset = isotope_calculator.get_distribution(protein_avg_mass)
        effective_protein_mass = protein_avg_mass + most_abundant_offset
        rng_scenario, rng_intensity = default_rng(file_seed), default_rng(file_seed + 1)
        total_binding, dar2_of_bound, desc = 0.0, 0.0, "No Binding"

        if rng_scenario.random() < common_params['prob_binding']:
            total_binding = rng_intensity.uniform(*common_params['total_binding_range'])
            desc = f"Binding ({total_binding:.2f}%)"
            if rng_scenario.random() < common_params['prob_dar2']:
                dar2_of_bound = rng_intensity.uniform(*common_params['dar2_range'])
                desc += f", DAR-2 ({dar2_of_bound:.2f}% of bound)"

        message = f"--- Processing: {compound_name} | Scenario: {desc} ---\n"

        # Prepare filename and path
        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": datetime.now().strftime('%H%M%S'),
            "compound_name": compound_name,
            "protein_mass": int(round(effective_protein_mass)),
            "scans": common_params['num_scans'],
            "noise": common_params['noise_option'].replace(" ", ""),
            "seed": file_seed
        }
        filename = format_filename(common_params['filename_template'], placeholders)
        filepath = os.path.join(common_params['output_directory'], filename)

        # Update common_params with the per-task seed for reproducibility
        task_common_params = common_params.copy()
        task_common_params['seed'] = file_seed

        # Delegate the core work to the binding simulation logic function
        success, final_filepath = execute_binding_simulation(
            protein_avg_mass=protein_avg_mass,
            compound_mass=compound_mass,
            total_binding_percentage=total_binding,
            dar2_percentage_of_bound=dar2_of_bound,
            filepath=filepath,
            common_params=task_common_params
        )

        if success:
            message += f"  SUCCESS: Wrote to {os.path.basename(final_filepath)}\n"
            return (True, message)
        else:
            message += "  ERROR: Spectrum generation or file write failed.\n"
            return (False, message)

    except Exception as e:
        return (False, f"--- Critical error for compound {args[0]}: {e} ---\n")
