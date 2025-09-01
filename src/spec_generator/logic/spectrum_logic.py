import os
import copy
import threading
import multiprocessing
from datetime import datetime

import numpy as np

from ..config import SpectrumGeneratorConfig
from ..utils.file_io import read_protein_list_file, format_filename
from ..logic.simulation import execute_simulation_and_write_mzml, run_simulation_for_preview
from ..workers.tasks import run_simulation_task
from ..core.spectrum import generate_protein_spectrum
from ..core.lc import apply_lc_profile_and_noise
from ..core.constants import BASE_INTENSITY_SCALAR
from ..utils.ui_helpers import show_plot, parse_float_entry


class SpectrumTabLogic:
    def __init__(self):
        pass

    def validate_and_prepare_config(self, config_dict: dict, queue) -> SpectrumGeneratorConfig:
        mass_str = config_dict["protein_masses_str"]
        mass_list = [float(m.strip()) for m in mass_str.split(',') if m.strip()]
        scalar_str = config_dict["intensity_scalars_str"]
        scalar_list = [float(s.strip()) for s in scalar_str.split(',') if s.strip()]

        if not scalar_list and mass_list:
            scalar_list = [1.0] * len(mass_list)
        if len(scalar_list) != len(mass_list) and mass_list:
            queue.put(('warning', "Mismatched scalars and masses. Adjusting..."))
            scalar_list = (scalar_list + [1.0] * len(mass_list))[:len(mass_list)]

        return SpectrumGeneratorConfig(
            common=config_dict["common"],
            lc=config_dict["lc"],
            protein_list_file=config_dict["protein_list_file"],
            protein_masses=mass_list,
            intensity_scalars=scalar_list,
            mass_inhomogeneity=parse_float_entry(config_dict["mass_inhomogeneity_str"], "Mass Inhomogeneity")
        )

    def validate_and_prepare_template_data(self, mass_str: str, scalar_str: str, queue):
        masses = [m.strip() for m in mass_str.split(',') if m.strip()]
        scalars = [s.strip() for s in scalar_str.split(',') if s.strip()]

        if not masses:
            raise ValueError("No protein masses entered to save.")

        if len(masses) != len(scalars):
            queue.put(('warning', "Mismatched scalars and masses. Padding with 1.0."))
            scalars = (scalars + ['1.0'] * len(masses))[:len(masses)]

        return masses, scalars

    def generate_spectrum(self, config_dict: dict, task_queue):
        try:
            config = self.validate_and_prepare_config(config_dict, task_queue)
            use_file_input = config.protein_list_file
            worker = self._worker_generate_from_protein_file if use_file_input else self._worker_generate_from_manual_input
            threading.Thread(target=worker, args=(config, task_queue), daemon=True).start()
        except ValueError as e:
            task_queue.put(('error', str(e)))
            task_queue.put(('done', None))

    def start_plot_generation(self, config_dict: dict, task_queue, callback):
        try:
            config = self.validate_and_prepare_config(config_dict, task_queue)
            if not config.protein_masses:
                raise ValueError("No protein masses entered for plotting.")

            pool = multiprocessing.Pool(processes=1)
            pool.apply_async(run_simulation_task, args=(config, True), callback=callback)
            pool.close()

        except Exception as e:
            task_queue.put(('error', f"A processing error occurred: {e}"))
            callback(None)

    def _worker_generate_from_protein_file(self, config: SpectrumGeneratorConfig, task_queue):
        try:
            protein_list = read_protein_list_file(config.protein_list_file)
        except (ValueError, FileNotFoundError) as e:
            task_queue.put(('error', str(e)))
            task_queue.put(('done', None))
            return

        jobs = []
        for i, (mass, scalar) in enumerate(protein_list):
            job_config = copy.deepcopy(config)
            job_config.common.seed = config.common.seed + i
            job_config.protein_masses = [mass]
            job_config.intensity_scalars = [scalar]
            jobs.append(job_config)

        task_queue.put(('log', f"Starting batch generation for {len(jobs)} proteins using {os.cpu_count()} processes...\n\n"))
        task_queue.put(('progress_max', len(jobs)))
        task_queue.put(('progress_set', 0))

        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_simulation_task, jobs)):
                    task_queue.put(('log', message))
                    if success:
                        success_count += 1
                    task_queue.put(('progress_set', i + 1))
            task_queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} mzML files."))
        except Exception as e:
            task_queue.put(('error', f"A multiprocessing error occurred: {e}"))
            task_queue.put(('done', None))

    def _worker_generate_from_manual_input(self, config: SpectrumGeneratorConfig, task_queue):
        try:
            if not config.protein_masses:
                raise ValueError("No protein masses entered.")
        except ValueError as e:
            task_queue.put(('error', f"Invalid input: {e}"))
            task_queue.put(('done', None))
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

        # The lambda function acts as a bridge between the callback system and the GUI queue
        success = execute_simulation_and_write_mzml(
            config, filepath, progress_callback=lambda type, value: task_queue.put((type, value))
        )

        if success:
            task_queue.put(('done', "mzML file successfully created."))
        else:
            task_queue.put(('done', None))

    def preview_spectrum(self, config_dict: dict, task_queue):
        try:
            config = self.validate_and_prepare_config(config_dict, task_queue)
            if not config.protein_masses:
                raise ValueError("Please enter at least one protein mass.")

            # Note: run_simulation_for_preview is now decoupled and does not need a callback.
            simulation_result = run_simulation_for_preview(config)
            if simulation_result:
                mz_range, preview_spectrum = simulation_result
                protein_avg_mass = config.protein_masses[0]
                title = f"Preview (Avg Mass: {protein_avg_mass:.0f} Da, Res: {config.common.resolution/1000}k)"
                show_plot(mz_range, {"Apex Scan Preview": preview_spectrum}, title)

        except (ValueError, IndexError) as e:
            task_queue.put(('error', f"Invalid parameters for preview: {e}"))
        except Exception as e:
            task_queue.put(('error', f"An unexpected error occurred during preview: {e}"))
        finally:
            task_queue.put(('preview_done', None))
