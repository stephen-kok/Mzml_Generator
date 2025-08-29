import os
import csv
import threading
import multiprocessing
import copy
from datetime import datetime
from tkinter import (HORIZONTAL, LEFT, NORMAL, DISABLED, NSEW, SUNKEN, WORD,
                     StringVar, messagebox, filedialog, Text)

from ttkbootstrap import widgets as ttk
from ttkbootstrap.constants import PRIMARY

import numpy as np

from ...utils.ui_helpers import Tooltip, parse_float_entry, show_plot
from ...utils.file_io import read_protein_list_file, format_filename
from ...logic.simulation import execute_simulation_and_write_mzml
from ...workers.tasks import run_simulation_task
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame
from ...core.spectrum import generate_protein_spectrum
from ...core.lc import apply_lc_profile_and_noise
from ...core.constants import BASE_INTENSITY_SCALAR
from ...config import SpectrumGeneratorConfig

class SpectrumTab(BaseTab):
    def create_widgets(self):
        # --- Protein Input Frame ---
        in_frame = ttk.LabelFrame(self.content_frame, text="Protein Parameters", padding=(15, 10))
        in_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        in_frame.columnconfigure(1, weight=1)

        ttk.Label(in_frame, text="Protein List File (.txt):").grid(row=0, column=0, sticky="w", pady=5)
        self.protein_list_file_var = StringVar()
        self.protein_list_file_entry = ttk.Entry(in_frame, textvariable=self.protein_list_file_var)
        self.protein_list_file_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        Tooltip(self.protein_list_file_entry, "Path to a tab-delimited file with protein data.\nMust contain 'Protein' (Average Mass) and 'Intensity' headers.")

        self.protein_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_protein_list, style='Outline.TButton')
        self.protein_list_browse_button.grid(row=0, column=2, sticky="w", pady=5, padx=(5,0))
        Tooltip(self.protein_list_browse_button, "Browse for a protein list file.")

        self.save_template_button = ttk.Button(in_frame, text="Save as Template...", command=self._save_protein_template, style='Outline.TButton')
        self.save_template_button.grid(row=0, column=3, sticky="w", pady=5, padx=(2,5))
        Tooltip(self.save_template_button, "Save the manually entered masses and scalars below as a valid\ntab-delimited template file for future use.")

        ttk.Separator(in_frame, orient=HORIZONTAL).grid(row=1, column=0, columnspan=4, sticky="ew", pady=10)
        ttk.Label(in_frame, text="OR Enter Manually Below", bootstyle="secondary").grid(row=2, column=0, columnspan=4)

        ttk.Label(in_frame, text="Protein Avg. Masses (Da, comma-sep):").grid(row=3, column=0, sticky="w", pady=5)
        self.spectrum_protein_masses_entry = ttk.Entry(in_frame)
        self.spectrum_protein_masses_entry.insert(0, "25000")
        self.spectrum_protein_masses_entry.grid(row=3, column=1, columnspan=3, sticky="ew", pady=5, padx=5)
        Tooltip(self.spectrum_protein_masses_entry, "A comma-separated list of protein AVERAGE masses to simulate.")

        ttk.Label(in_frame, text="Intensity Scalars (comma-sep):").grid(row=4, column=0, sticky="w", pady=5)
        self.intensity_scalars_entry = ttk.Entry(in_frame)
        self.intensity_scalars_entry.insert(0, "1.0")
        self.intensity_scalars_entry.grid(row=4, column=1, columnspan=3, sticky="ew", pady=5, padx=5)
        Tooltip(self.intensity_scalars_entry, "A comma-separated list of relative intensity multipliers.\nMust match the number of protein masses.")

        ttk.Label(in_frame, text="Mass Inhomogeneity (Std. Dev., Da):").grid(row=5, column=0, sticky="w", pady=5)
        self.mass_inhomogeneity_entry = ttk.Entry(in_frame)
        self.mass_inhomogeneity_entry.insert(0, "0.0")
        self.mass_inhomogeneity_entry.grid(row=5, column=1, columnspan=3, sticky="ew", pady=5, padx=5)
        Tooltip(self.mass_inhomogeneity_entry, "Standard deviation of protein mass distribution to simulate conformational broadening.\nSet to 0 to disable. A small value (e.g., 1-5 Da) is recommended.")

        # --- Common Parameters ---
        common_frame = ttk.Frame(self.content_frame)
        common_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=0)
        self.spec_gen_params = create_common_parameters_frame(common_frame, "400.0", "2500.0", "Default Noise")
        self.spec_gen_params['output_directory_var'].set(os.path.join(os.getcwd(), "Mzml Mock Spectra"))
        self.spec_gen_params['filename_template_var'].set("{date}_protein_{protein_mass}_{scans}scans_{noise}.mzML")

        # --- LC Simulation ---
        lc_container, self.spec_gen_lc_params = create_lc_simulation_frame(self.content_frame, False)
        lc_container.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        # --- Action Buttons ---
        button_frame = ttk.Frame(self.content_frame)
        button_frame.grid(row=3, column=0, pady=15)
        self.spectrum_preview_button = ttk.Button(button_frame, text="Preview Spectrum", command=self._preview_spectrum_tab_command, style='Outline.TButton')
        self.spectrum_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_preview_button, "Generate and display a plot of a single spectrum using the current settings.\nUses the first protein mass if multiple are entered.")

        self.spectrum_generate_button = ttk.Button(button_frame, text="Generate mzML File(s)", command=self.generate_spectrum_tab_command, bootstyle=PRIMARY)
        self.spectrum_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_generate_button, "Generate and save .mzML file(s) with the specified parameters.")

        # --- Progress & Output ---
        self.progress_bar = ttk.Progressbar(self.content_frame, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=4, column=0, pady=5, sticky="ew", padx=10)

        out_frame = ttk.Frame(self.content_frame)
        out_frame.grid(row=5, column=0, sticky=NSEW, padx=10, pady=(5, 10))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        self.content_frame.rowconfigure(5, weight=1)
        self.output_text = Text(out_frame, height=10, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.output_text.grid(row=0, column=0, sticky=NSEW)
        scrollbar = ttk.Scrollbar(out_frame, command=self.output_text.yview)
        scrollbar.grid(row=0, column=1, sticky=NSEW)
        self.output_text['yscrollcommand'] = scrollbar.set

        # --- Bindings ---
        self.protein_list_file_var.trace_add("write", self._toggle_protein_inputs)
        self._toggle_protein_inputs()

    def _gather_config(self) -> SpectrumGeneratorConfig:
        common, lc = self._gather_common_params(self.spec_gen_params, self.spec_gen_lc_params)

        mass_str = self.spectrum_protein_masses_entry.get()
        mass_list = [float(m.strip()) for m in mass_str.split(',') if m.strip()]
        scalar_str = self.intensity_scalars_entry.get()
        scalar_list = [float(s.strip()) for s in scalar_str.split(',') if s.strip()]
        if not scalar_list and mass_list:
            scalar_list = [1.0] * len(mass_list)
        if len(scalar_list) != len(mass_list) and mass_list:
            self.app_queue.put(('warning', "Mismatched scalars and masses. Adjusting..."))
            scalar_list = (scalar_list + [1.0] * len(mass_list))[:len(mass_list)]

        return SpectrumGeneratorConfig(
            common=common,
            lc=lc,
            protein_list_file=self.protein_list_file_var.get() or None,
            protein_masses=mass_list,
            intensity_scalars=scalar_list,
            mass_inhomogeneity=parse_float_entry(self.mass_inhomogeneity_entry.get(), "Mass Inhomogeneity")
        )

    def _save_protein_template(self):
        try:
            mass_str = self.spectrum_protein_masses_entry.get()
            scalar_str = self.intensity_scalars_entry.get()
            masses = [m.strip() for m in mass_str.split(',') if m.strip()]
            scalars = [s.strip() for s in scalar_str.split(',') if s.strip()]

            if not masses:
                messagebox.showerror("Error", "No protein masses entered to save.")
                return

            if len(masses) != len(scalars):
                if messagebox.askokcancel("Warning", "The number of masses and intensity scalars do not match. Continue with 1.0 for missing scalars?"):
                    scalars = (scalars + ['1.0'] * len(masses))[:len(masses)]
                else:
                    return

            filepath = filedialog.asksaveasfilename(
                title="Save Protein List Template",
                initialfile="protein_list_template.txt",
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if not filepath:
                return

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['Protein', 'Intensity'])
                writer.writerows(zip(masses, scalars))

            self.app_queue.put(('log', f"Saved template to {os.path.basename(filepath)}\n"))
            self.protein_list_file_var.set(filepath)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save template file.\nError: {e}")

    def _toggle_protein_inputs(self, *args):
        state = DISABLED if self.protein_list_file_var.get() else NORMAL
        self.spectrum_protein_masses_entry.config(state=state)
        self.intensity_scalars_entry.config(state=state)

    def browse_protein_list(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd())
        if filepath:
            self.protein_list_file_var.set(filepath)

    def generate_spectrum_tab_command(self):
        self.spectrum_generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.app_queue.put(('clear_log', None))

        use_file_input = self.protein_list_file_var.get()
        worker = self._worker_generate_from_protein_file if use_file_input else self._worker_generate_from_manual_input

        threading.Thread(target=worker, daemon=True).start()

    def _worker_generate_from_protein_file(self):
        try:
            config = self._gather_config()
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
        self.progress_bar["maximum"] = len(jobs)
        self.progress_bar["value"] = 0

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

    def _worker_generate_from_manual_input(self):
        try:
            config = self._gather_config()
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

    def _preview_spectrum_tab_command(self):
        try:
            config = self._gather_config()
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

        except (ValueError, IndexError) as e:
            messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e:
            messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")

    def on_task_done(self):
        self.spectrum_generate_button.config(state=NORMAL)
