import os
import threading
import copy
from tkinter import (HORIZONTAL, LEFT, NORMAL, DISABLED, NSEW, SUNKEN, WORD,
                     StringVar, messagebox, filedialog, Text)

from ttkbootstrap import widgets as ttk
from ttkbootstrap.constants import PRIMARY

import numpy as np
import multiprocessing

from ...utils.ui_helpers import Tooltip, parse_float_entry, parse_range_entry, show_plot
from ...utils.file_io import read_compound_list_file
from ...workers.tasks import run_binding_task
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame
from ...core.spectrum import generate_protein_spectrum, generate_binding_spectrum
from ...core.lc import apply_lc_profile_and_noise
from ...core.constants import BASE_INTENSITY_SCALAR
from ...config import CovalentBindingConfig

class BindingTab(BaseTab):
    def create_widgets(self):
        # --- Target & Compound Frame ---
        in_frame = ttk.LabelFrame(self.content_frame, text="Target & Compound", padding=(15, 10))
        in_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        in_frame.columnconfigure(1, weight=1)

        ttk.Label(in_frame, text="Protein Avg. Mass (Da):").grid(row=0, column=0, sticky="w", pady=5)
        self.binding_protein_mass_entry = ttk.Entry(in_frame)
        self.binding_protein_mass_entry.insert(0, "25000")
        self.binding_protein_mass_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        Tooltip(self.binding_protein_mass_entry, "The AVERAGE mass of the target protein.")

        ttk.Label(in_frame, text="Compound List File (.txt):").grid(row=1, column=0, sticky="w", pady=5)
        self.compound_list_file_var = StringVar()
        self.compound_list_file_entry = ttk.Entry(in_frame, textvariable=self.compound_list_file_var)
        self.compound_list_file_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        Tooltip(self.compound_list_file_entry, "Path to a tab-delimited file of compounds to test.\nMust contain 'Name' and 'Delta' (Average Mass) headers.")

        self.compound_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_compound_list, style='Outline.TButton')
        self.compound_list_browse_button.grid(row=1, column=2, sticky="w", pady=5, padx=5)
        Tooltip(self.compound_list_browse_button, "Browse for a compound list file.")

        # --- Common Parameters ---
        common_frame = ttk.Frame(self.content_frame)
        common_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=0)
        self.binding_params = create_common_parameters_frame(common_frame, "400.0", "2000.0", "Default Noise")
        self.binding_params['output_directory_var'].set(os.path.join(os.getcwd(), "Intact Covalent Binding Mock Spectra"))
        self.binding_params['filename_template_var'].set("{date}_{compound_name}_on_{protein_mass}_{scans}scans_{noise}.mzML")

        # --- LC Simulation ---
        lc_container, self.binding_lc_params = create_lc_simulation_frame(self.content_frame, False)
        lc_container.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        # --- Binding Probabilities Frame ---
        prob_frame = ttk.LabelFrame(self.content_frame, text="Binding Probabilities", padding=(15, 10))
        prob_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        ttk.Label(prob_frame, text="Probability of Binding:").grid(row=0, column=0, sticky="w", pady=2)
        self.prob_binding_entry = ttk.Entry(prob_frame, width=10)
        self.prob_binding_entry.insert(0, "0.5")
        self.prob_binding_entry.grid(row=0, column=1, sticky="w", pady=2, padx=5)
        Tooltip(self.prob_binding_entry, "The chance (0.0 to 1.0) that any given compound will bind to the protein.")

        ttk.Label(prob_frame, text="Probability of DAR-2 (if Binding):").grid(row=1, column=0, sticky="w", pady=2)
        self.prob_dar2_if_binding_entry = ttk.Entry(prob_frame, width=10)
        self.prob_dar2_if_binding_entry.insert(0, "0.1")
        self.prob_dar2_if_binding_entry.grid(row=1, column=1, sticky="w", pady=2, padx=5)
        Tooltip(self.prob_dar2_if_binding_entry, "If a compound binds, this is the chance (0.0 to 1.0)\nthat it will form a doubly-adducted species (DAR-2).")

        ttk.Label(prob_frame, text="Total Binding % Range (if Binding):").grid(row=2, column=0, sticky="w", pady=2)
        self.total_binding_percentage_range_entry = ttk.Entry(prob_frame)
        self.total_binding_percentage_range_entry.insert(0, "10-50")
        self.total_binding_percentage_range_entry.grid(row=2, column=1, sticky="ew", pady=2, padx=5)
        Tooltip(self.total_binding_percentage_range_entry, "If binding occurs, the total percentage of protein that is modified\nwill be randomly chosen from this range (e.g., '10-50').")

        ttk.Label(prob_frame, text="DAR-2 % Range (of Total Bound):").grid(row=3, column=0, sticky="w", pady=2)
        self.dar2_percentage_of_bound_range_entry = ttk.Entry(prob_frame)
        self.dar2_percentage_of_bound_range_entry.insert(0, "5-20")
        self.dar2_percentage_of_bound_range_entry.grid(row=3, column=1, sticky="ew", pady=2, padx=5)
        Tooltip(self.dar2_percentage_of_bound_range_entry, "If DAR-2 occurs, the percentage of the bound protein that is DAR-2\nwill be randomly chosen from this range (e.g., '5-20').")

        # --- Action Buttons ---
        button_frame = ttk.Frame(self.content_frame)
        button_frame.grid(row=4, column=0, pady=15)
        self.binding_preview_button = ttk.Button(button_frame, text="Preview Binding", command=self._preview_binding_tab_command, style='Outline.TButton')
        self.binding_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_preview_button, "Generate and display a plot of a binding spectrum using average probability values.")

        self.binding_generate_button = ttk.Button(button_frame, text="Generate Binding Spectra", command=self.generate_binding_spectra_command, bootstyle=PRIMARY)
        self.binding_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_generate_button, "Generate an .mzML file for each compound in the list,\nwith binding determined by the specified probabilities.")

        # --- Progress & Output ---
        self.progress_bar = ttk.Progressbar(self.content_frame, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=5, column=0, pady=5, sticky="ew", padx=10)

        out_frame = ttk.Frame(self.content_frame)
        out_frame.grid(row=6, column=0, sticky=NSEW, padx=10, pady=(5, 10))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        self.content_frame.rowconfigure(6, weight=1)
        self.output_text = Text(out_frame, height=10, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.output_text.grid(row=0, column=0, sticky=NSEW)
        scrollbar = ttk.Scrollbar(out_frame, command=self.output_text.yview)
        scrollbar.grid(row=0, column=1, sticky=NSEW)
        self.output_text['yscrollcommand'] = scrollbar.set

    def _gather_config(self) -> CovalentBindingConfig:
        common, lc = self._gather_common_params(self.binding_params, self.binding_lc_params)
        return CovalentBindingConfig(
            common=common,
            lc=lc,
            protein_avg_mass=parse_float_entry(self.binding_protein_mass_entry.get(), "Protein Avg. Mass"),
            compound_list_file=self.compound_list_file_var.get(),
            prob_binding=parse_float_entry(self.prob_binding_entry.get(), "Prob Binding"),
            prob_dar2=parse_float_entry(self.prob_dar2_if_binding_entry.get(), "Prob DAR-2"),
            total_binding_range=parse_range_entry(self.total_binding_percentage_range_entry.get(), "Total Binding %"),
            dar2_range=parse_range_entry(self.dar2_percentage_of_bound_range_entry.get(), "DAR-2 %"),
        )

    def browse_compound_list(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd())
        if filepath:
            self.compound_list_file_var.set(filepath)

    def generate_binding_spectra_command(self):
        self.binding_generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.app_queue.put(('clear_log', None))
        threading.Thread(target=self._worker_generate_binding_spectra, daemon=True).start()

    def _worker_generate_binding_spectra(self):
        try:
            config = self._gather_config()
            compounds = read_compound_list_file(config.compound_list_file)
        except (ValueError, FileNotFoundError) as e:
            self.app_queue.put(('error', str(e)))
            self.app_queue.put(('done', None))
            return

        jobs = []
        for i, (name, mass) in enumerate(compounds):
            job_config = copy.deepcopy(config)
            job_config.common.seed = config.common.seed + i
            jobs.append((name, mass, job_config))

        self.app_queue.put(('log', f"Starting batch generation for {len(jobs)} compounds using {os.cpu_count()} processes...\n\n"))
        self.progress_bar["maximum"] = len(jobs)
        self.progress_bar["value"] = 0

        try:
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_binding_task, jobs)):
                    self.app_queue.put(('log', message))
                    if success:
                        success_count += 1
                    self.app_queue.put(('progress_set', i + 1))
            self.app_queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} binding mzML files."))
        except Exception as e:
            self.app_queue.put(('error', f"A multiprocessing error occurred: {e}"))
            self.app_queue.put(('done', None))

    def _preview_binding_tab_command(self):
        try:
            config = self._gather_config()
            compound_avg_mass = 500.0

            total_binding_pct = (config.total_binding_range[0] + config.total_binding_range[1]) / 2
            dar2_pct_of_bound = (config.dar2_range[0] + config.dar2_range[1]) / 2

            mz_range = np.arange(config.common.mz_range_start, config.common.mz_range_end + config.common.mz_step, config.common.mz_step)

            base_native_spec = generate_protein_spectrum(config.protein_avg_mass, mz_range, config.common, BASE_INTENSITY_SCALAR)
            base_dar1_spec = generate_protein_spectrum(config.protein_avg_mass + compound_avg_mass, mz_range, config.common, BASE_INTENSITY_SCALAR)
            base_dar2_spec = generate_protein_spectrum(config.protein_avg_mass + 2 * compound_avg_mass, mz_range, config.common, BASE_INTENSITY_SCALAR)

            final_spec_clean = generate_binding_spectrum(
                config.protein_avg_mass, compound_avg_mass, mz_range, config.common,
                total_binding_pct, dar2_pct_of_bound,
                BASE_INTENSITY_SCALAR
            )

            apex_scan_index = (config.lc.num_scans - 1) // 2
            final_spec_noisy = apply_lc_profile_and_noise(
                mz_range, [final_spec_clean], config.lc.num_scans, config.lc.gaussian_std_dev,
                config.lc.lc_tailing_factor, config.common.seed, config.common.noise_option, config.common.pink_noise_enabled
            )[0][apex_scan_index]

            plot_data = {"Combined Spectrum (with noise)": final_spec_noisy}
            if np.any(base_native_spec): plot_data["Native Protein (ref)"] = base_native_spec * 0.5
            if np.any(base_dar1_spec): plot_data["DAR-1 Adduct (ref)"] = base_dar1_spec * 0.5
            if np.any(base_dar2_spec): plot_data["DAR-2 Adduct (ref)"] = base_dar2_spec * 0.5

            title = f"Binding Preview (Target Avg Mass: ~{config.protein_avg_mass:.0f} Da, Res: {config.common.resolution/1000}k)"
            show_plot(mz_range, plot_data, title=title)

        except (ValueError, IndexError) as e:
            messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e:
            messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")

    def on_task_done(self):
        self.binding_generate_button.config(state=NORMAL)
