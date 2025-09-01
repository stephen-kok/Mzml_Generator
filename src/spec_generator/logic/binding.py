import os
import numpy as np

from ..core.spectrum import generate_binding_spectrum
from ..core.lc import apply_lc_profile_and_noise
from ..utils.mzml import create_mzml_content_et
from ..utils.file_io import create_unique_filename
from ..core.constants import BASE_INTENSITY_SCALAR
from ..config import CovalentBindingConfig


def execute_binding_simulation(
    config: CovalentBindingConfig,
    compound_mass: float,
    total_binding_percentage: float,
    dar2_percentage_of_bound: float,
    filepath: str,
    return_data_only: bool = False,
) -> tuple[bool, str] | tuple[np.ndarray, list[np.ndarray]]:
    """
    Executes the logic for a single covalent binding simulation and writes the file.
    Returns a tuple of (success_boolean, final_filepath_string).
    """
    try:
        common = config.common
        lc = config.lc

        mz_range = np.arange(
            common.mz_range_start,
            common.mz_range_end + common.mz_step,
            common.mz_step
        )

        # 1. Generate the clean, combined spectrum (native, DAR-1, DAR-2)
        clean_spec = generate_binding_spectrum(
            protein_avg_mass=config.protein_avg_mass,
            compound_avg_mass=compound_mass,
            mz_range=mz_range,
            mz_step_float=common.mz_step,
            peak_sigma_mz_float=common.peak_sigma_mz,
            total_binding_percentage=total_binding_percentage,
            dar2_percentage_of_bound=dar2_percentage_of_bound,
            original_intensity_scalar=BASE_INTENSITY_SCALAR,
            isotopic_enabled=common.isotopic_enabled,
            resolution=common.resolution
        )

        # 2. Apply LC profile and noise
        final_spectra = apply_lc_profile_and_noise(
            mz_range=mz_range,
            all_clean_spectra=[clean_spec],
            num_scans=lc.num_scans,
            gaussian_std_dev=lc.gaussian_std_dev,
            lc_tailing_factor=lc.lc_tailing_factor,
            seed=common.seed,
            noise_option=common.noise_option,
            pink_noise_enabled=common.pink_noise_enabled,
            progress_callback=None
        )

        # 3. Create mzML content
        mzml_content = create_mzml_content_et(
            mz_range=mz_range,
            run_data=[final_spectra], # Wrap in list for mzML writer
            scan_interval=lc.scan_interval,
            progress_callback=None
        )
        if not mzml_content:
            return False, ""

        # 4. Return data if requested, otherwise write to file
        if return_data_only:
            return mz_range, final_spectra

        unique_filepath = create_unique_filename(filepath)
        os.makedirs(os.path.dirname(unique_filepath), exist_ok=True)
        with open(unique_filepath, "wb") as f:
            f.write(mzml_content)

        return True, unique_filepath

    except Exception as e:
        print(f"Error in binding simulation for {filepath}: {e}")
        return False, ""
