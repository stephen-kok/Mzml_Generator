import threading
import numpy as np

from ..config import CovalentBindingConfig
from ..core.spectrum import generate_protein_spectrum, generate_binding_spectrum
from ..core.lc import apply_lc_profile_and_noise
from ..core.constants import BASE_INTENSITY_SCALAR

class BindingTabLogic:
    def start_plot_generation(self, config: CovalentBindingConfig, task_queue, callback):
        """
        Starts the plot generation in a new thread.
        """
        threading.Thread(
            target=self._worker_generate_and_plot,
            args=(config, task_queue, callback),
            daemon=True
        ).start()

    def _worker_generate_and_plot(self, config: CovalentBindingConfig, task_queue, callback):
        """
        The actual worker function that runs in a thread.
        It simulates a single "average" case for plotting.
        """
        try:
            # Use an arbitrary average mass for the compound for previewing
            compound_avg_mass = 500.0

            total_binding_pct = (config.total_binding_range[0] + config.total_binding_range[1]) / 2
            dar2_pct_of_bound = (config.dar2_range[0] + config.dar2_range[1]) / 2

            mz_range = np.arange(config.common.mz_range_start, config.common.mz_range_end + config.common.mz_step, config.common.mz_step)

            # Generate the combined spectrum
            final_spec_clean = generate_binding_spectrum(
                config.protein_avg_mass, compound_avg_mass, mz_range, config.common.mz_step,
                config.common.peak_sigma_mz, total_binding_pct, dar2_pct_of_bound,
                BASE_INTENSITY_SCALAR, config.common.isotopic_enabled, config.common.resolution
            )

            # Apply LC profile and noise
            chromatogram_scans = apply_lc_profile_and_noise(
                mz_range, [final_spec_clean], config.lc.num_scans, config.lc.gaussian_std_dev,
                config.lc.lc_tailing_factor, config.common.seed, config.common.noise_option, config.common.pink_noise_enabled
            )

            # The data format for the plotter is (mz_range, [list_of_scans])
            result = (mz_range, [chromatogram_scans])
            task_queue.put(('callback', (callback, result)))

        except Exception as e:
            task_queue.put(('error', f"An error occurred during plot generation: {e}"))
            task_queue.put(('callback', (callback, None)))
