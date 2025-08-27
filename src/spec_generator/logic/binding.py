import os
import numpy as np

from ..core.spectrum import generate_binding_spectrum
from ..core.lc import apply_lc_profile_and_noise
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename
from ..core.constants import BASE_INTENSITY_SCALAR


def execute_binding_simulation(
    protein_avg_mass: float,
    compound_mass: float,
    total_binding_percentage: float,
    dar2_percentage_of_bound: float,
    filepath: str,
    common_params: dict
) -> tuple[bool, str]:
    """
    Executes the logic for a single covalent binding simulation and writes the file.
    Returns a tuple of (success_boolean, final_filepath_string).
    """
    try:
        mz_range = np.arange(
            float(common_params['mz_range_start']),
            float(common_params['mz_range_end']) + float(common_params['mz_step']),
            float(common_params['mz_step'])
        )

        # 1. Generate the clean, combined spectrum (native, DAR-1, DAR-2)
        clean_spec = generate_binding_spectrum(
            protein_avg_mass=protein_avg_mass,
            compound_avg_mass=compound_mass,
            mz_range=mz_range,
            mz_step_float=float(common_params['mz_step']),
            peak_sigma_mz_float=float(common_params['peak_sigma_mz']),
            total_binding_percentage=total_binding_percentage,
            dar2_percentage_of_bound=dar2_percentage_of_bound,
            original_intensity_scalar=BASE_INTENSITY_SCALAR, # Base intensity is a constant
            isotopic_enabled=common_params['isotopic_enabled'],
            resolution=common_params['resolution']
        )

        # 2. Apply LC profile and noise
        scans_to_gen = common_params['num_scans']
        final_spectra = apply_lc_profile_and_noise(
            mz_range=mz_range,
            all_clean_spectra=[clean_spec], # The binding spectrum is treated as a single "protein"
            num_scans=scans_to_gen,
            gaussian_std_dev=common_params['gaussian_std_dev'],
            lc_tailing_factor=common_params.get('lc_tailing_factor', 0.0),
            seed=common_params['seed'],
            noise_option=common_params['noise_option'],
            pink_noise_enabled=common_params.get('pink_noise_enabled', False),
            update_queue=None # No queue for worker processes
        )

        # 3. Create mzML content
        mzml_content = create_mzml_content_et(
            mz_range=mz_range,
            run_data=final_spectra,
            scan_interval=common_params['scan_interval'],
            update_queue=None
        )
        if not mzml_content:
            return False, ""

        # 4. Write to file
        unique_filepath = create_unique_filename(filepath)
        os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)
        with open(unique_filepath, "wb") as f:
            f.write(mzml_content)

        return True, unique_filepath

    except Exception as e:
        # Log error to console, as this runs in a separate process
        print(f"Error in binding simulation for {filepath}: {e}")
        return False, ""
