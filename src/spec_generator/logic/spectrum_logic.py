import os
import copy
import threading
import multiprocessing
from datetime import datetime

import numpy as np

from ..config import SpectrumGeneratorConfig
from ..utils.file_io import read_protein_list_file, format_filename
from ..logic.simulation import execute_simulation_and_write_mzml
from ..workers.tasks import run_simulation_task
from ..core.spectrum import generate_protein_spectrum
from ..core.lc import apply_lc_profile_and_noise
from ..core.constants import BASE_INTENSITY_SCALAR
from ..utils.ui_helpers import show_plot


class SpectrumTabLogic:
    def __init__(self, app_queue):
        self.app_queue = app_queue

    def generate_spectrum(self, config: SpectrumGeneratorConfig):
        use_file_input = config.protein_list_file
        worker = self._worker_generate_from_protein_file if use_file_input else self._worker_generate_from_manual_input
        threading.Thread(target=worker, args=(config,), daemon=True).start()

    def _worker_generate_from_protein_file(self, config: SpectrumGeneratorConfig):
        try:
            protein_list = read_protein_list_file(config.protein_list_file)
        except (ValueError, FileNotFoundError) as e:
            self.app_queue.put(('error', str(e)))
            self.app_queue.put(('done', None))
            return

        jobs = []
        for i, (mass, scalar) in enumerate(protein_list):
            job_config = copy.deepcopy(config)
            job_config.common.seed = config.common.seed + i
            job_config.protein_masses = [mass]
            job_config.intensity_scalars = [scalar]
            jobs.append(job_config)

        self.app_queue.put(('log', f"Starting batch generation for {len(jobs)} proteins using {os.cpu_count()} processes...\n\n"))
        self.app_queue.put(('progress_max', len(jobs)))
        self.app_queue.put(('progress_set', 0))

        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_simulation_task, jobs)):
                    self.app_queue.put(('log', message))
                    if success:
                        success_count += 1
                    self.app_queue.put(('progress_set', i + 1))
            self.app_queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} mzML files."))
        except Exception as e:
            self.app_queue.put(('error', f"A multiprocessing error occurred: {e}"))
            self.app_queue.put(('done', None))

    def _worker_generate_from_manual_input(self, config: SpectrumGeneratorConfig):
        try:
            if not config.protein_masses:
                raise ValueError("No protein masses entered.")
        except ValueError as e:
            self.app_queue.put(('error', f"Invalid input: {e}"))
            self.app_queue.put(('done', None))
            return

        avg_mass = config.protein_masses[0]
        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'), "time": datetime.now().strftime('%H%M%S'),
            "num_proteins": len(config.protein_masses), "scans": config.lc.num_scans,
            "noise": config.common.noise_option.replace(" ", ""), "seed": config.common.seed,
            "protein_mass": int(round(avg_mass))
        }
        filename = format_filename(config.common.filename_template, placeholders)
        filepath = os.path.join(config.common.output_directory, filename)

        success = execute_simulation_and_write_mzml(config, filepath, self.app_queue)

        if success:
            self.app_queue.put(('done', "mzML file successfully created."))
        else:
            self.app_queue.put(('done', None))

    def preview_spectrum(self, config: SpectrumGeneratorConfig):
        try:
            if not config.protein_masses:
                raise ValueError("Please enter at least one protein mass.")

            protein_avg_mass = config.protein_masses[0]
            mz_range = np.arange(config.common.mz_range_start, config.common.mz_range_end + config.common.mz_step, config.common.mz_step)

            clean_spec = generate_protein_spectrum(
                protein_avg_mass, mz_range, config.common.mz_step, config.common.peak_sigma_mz,
                BASE_INTENSITY_SCALAR, config.common.isotopic_enabled, config.common.resolution
            )

            apex_scan_index = (config.lc.num_scans - 1) // 2
            apex_scan_spectrum = apply_lc_profile_and_noise(
                mz_range, [clean_spec], config.lc.num_scans, config.lc.gaussian_std_dev,
                config.lc.lc_tailing_factor, config.common.seed, config.common.noise_option, config.common.pink_noise_enabled
            )[0][apex_scan_index]

            title = f"Preview (Avg Mass: {protein_avg_mass:.0f} Da, Res: {config.common.resolution/1000}k)"
            show_plot(mz_range, {"Apex Scan Preview": apex_scan_spectrum}, title)
            self.app_queue.put(('preview_done', None))

        except (ValueError, IndexError) as e:
            self.app_queue.put(('error', f"Invalid parameters for preview: {e}"))
        except Exception as e:
            self.app_queue.put(('error', f"An unexpected error occurred during preview: {e}"))
        finally:
            # Ensure the preview button is re-enabled even if plotting fails
            self.app_queue.put(('preview_done', None))
