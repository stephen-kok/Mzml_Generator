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
from ...workers.worker_init import init_worker
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame
from ...core.spectrum import generate_protein_spectrum, generate_binding_spectrum
from ...core.lc import apply_lc_profile_and_noise
from ...core.constants import BASE_INTENSITY_SCALAR
from ...config import CovalentBindingConfig


class BindingTab(BaseTab):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_event = None

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
        self.binding_params = create_common_parameters_frame(common_frame, "400.0", "2000.0")
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

        self.binding_generate_button = ttk.Button(button_frame, text="Generate Binding Spectra", command=self.generate_binding_spectra_command, bootstyle=PRIMARY)
        self.binding_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_generate_button, "Generate an .mzML file for each compound in the list,\nwith binding determined by the specified probabilities.")

        self.binding_stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_generation, state=DISABLED)
        self.binding_stop_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_stop_button, "Stop the current generation process.")

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
        self.binding_stop_button.config(state=NORMAL)
        self.progress_bar["value"] = 0
        self.task_queue.put(('clear_log', None))
        self.stop_event = multiprocessing.Event()
        threading.Thread(target=self._worker_generate_binding_spectra, args=(self.stop_event,), daemon=True).start()

    def stop_generation(self):
        if self.stop_event:
            self.task_queue.put(('log', "--- Stop signal sent ---\n"))
            self.stop_event.set()
        self.binding_stop_button.config(state=DISABLED)

    def _worker_generate_binding_spectra(self, stop_event):
        try:
            config = self._gather_config()
            compounds = read_compound_list_file(config.compound_list_file)
        except (ValueError, FileNotFoundError) as e:
            self.task_queue.put(('error', str(e)))
            self.task_queue.put(('done', None))
            return

        # Prepare jobs for the pool
        jobs = []
        for i, (name, mass) in enumerate(compounds):
            job_config = copy.deepcopy(config)
            job_config.common.seed = config.common.seed + i
            # The last arg `False` is `return_data_only`
            jobs.append((name, mass, job_config, False))

        self.task_queue.put(('log', f"Starting batch generation for {len(jobs)} compounds using {os.cpu_count()} processes...\n\n"))
        self.task_queue.put(('progress_max', len(jobs)))
        self.task_queue.put(('progress_set', 0))

        try:
            with multiprocessing.Pool(processes=os.cpu_count(), initializer=init_worker, initargs=(stop_event,)) as pool:
                success_count = 0
                results = pool.imap_unordered(run_binding_task, jobs)
                for i, result in enumerate(results):
                    if stop_event.is_set():
                        self.task_queue.put(('log', "Batch generation cancelled.\n"))
                        break
                    if result:
                        success, message = result
                        self.task_queue.put(('log', message))
                        if success:
                            success_count += 1
                    self.task_queue.put(('progress_set', i + 1))

            if not stop_event.is_set():
                self.task_queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} binding mzML files."))
            else:
                self.task_queue.put(('done', "Batch generation stopped."))
        except Exception as e:
            self.task_queue.put(('error', f"A multiprocessing error occurred: {e}"))
        finally:
            # This ensures the main thread knows the task is done
            self.task_queue.put(('done', None))

    def on_task_done(self):
        self.binding_generate_button.config(state=NORMAL)
        self.binding_stop_button.config(state=DISABLED)
        self.stop_event = None
