import os
import csv
import threading
from tkinter import (HORIZONTAL, LEFT, NORMAL, DISABLED, NSEW, SUNKEN, WORD,
                     StringVar, messagebox, filedialog, Text)

from ttkbootstrap import widgets as ttk
from ttkbootstrap.constants import PRIMARY

from ...utils.ui_helpers import Tooltip, parse_float_entry
from ...config import SpectrumGeneratorConfig
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame
from ...logic.spectrum_logic import SpectrumTabLogic


class SpectrumTab(BaseTab):
    def __init__(self, notebook, style, app_queue):
        super().__init__(notebook, style, app_queue)
        self.logic = SpectrumTabLogic(self.app_queue)

    def create_widgets(self):
        self._create_protein_input_frame()
        self._create_common_parameters_frame()
        self._create_lc_simulation_frame()
        self._create_action_buttons_frame()
        self._create_progress_and_output_frame()
        self._setup_bindings()

    def _create_protein_input_frame(self):
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

    def _create_common_parameters_frame(self):
        common_frame = ttk.Frame(self.content_frame)
        common_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=0)
        self.spec_gen_params = create_common_parameters_frame(common_frame, "400.0", "2500.0", "Default Noise")
        self.spec_gen_params['output_directory_var'].set(os.path.join(os.getcwd(), "Mzml Mock Spectra"))
        self.spec_gen_params['filename_template_var'].set("{date}_protein_{protein_mass}_{scans}scans_{noise}.mzML")

    def _create_lc_simulation_frame(self):
        lc_container, self.spec_gen_lc_params = create_lc_simulation_frame(self.content_frame, False)
        lc_container.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

    def _create_action_buttons_frame(self):
        button_frame = ttk.Frame(self.content_frame)
        button_frame.grid(row=3, column=0, pady=15)
        self.spectrum_preview_button = ttk.Button(button_frame, text="Preview Spectrum", command=self.preview_spectrum_command, style='Outline.TButton')
        self.spectrum_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_preview_button, "Generate and display a plot of a single spectrum using the current settings.\nUses the first protein mass if multiple are entered.")

        self.plot_button = ttk.Button(button_frame, text="Generate & Plot", command=self.generate_and_plot_command, style='Outline.TButton')
        self.plot_button.pack(side=LEFT, padx=5)
        Tooltip(self.plot_button, "Generate the spectrum and view it in the 'Plot Viewer' tab without saving a file.")

        self.spectrum_generate_button = ttk.Button(button_frame, text="Generate mzML File(s)", command=self.generate_spectrum_command, bootstyle=PRIMARY)
        self.spectrum_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_generate_button, "Generate and save .mzML file(s) with the specified parameters.")

    def _create_progress_and_output_frame(self):
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

    def _setup_bindings(self):
        self.protein_list_file_var.trace_add("write", self._toggle_protein_inputs)
        self._toggle_protein_inputs()

    def _gather_config(self) -> dict:
        common, lc = self._gather_common_params(self.spec_gen_params, self.spec_gen_lc_params)
        return {
            "common": common,
            "lc": lc,
            "protein_list_file": self.protein_list_file_var.get() or None,
            "protein_masses_str": self.spectrum_protein_masses_entry.get(),
            "intensity_scalars_str": self.intensity_scalars_entry.get(),
            "mass_inhomogeneity_str": self.mass_inhomogeneity_entry.get(),
        }

    def _save_protein_template(self):
        try:
            config_dict = self._gather_config()
            masses, scalars = self.logic.validate_and_prepare_template_data(
                config_dict['protein_masses_str'],
                config_dict['intensity_scalars_str']
            )

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
        except ValueError as e:
             messagebox.showerror("Error", str(e))
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

    def generate_spectrum_command(self):
        self.spectrum_generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.app_queue.put(('clear_log', None))
        try:
            config_dict = self._gather_config()
            self.logic.generate_spectrum(config_dict)
        except ValueError as e:
            self.app_queue.put(('error', f"Invalid input: {e}"))
            self.on_task_done()

    def preview_spectrum_command(self):
        self.spectrum_preview_button.config(state=DISABLED)
        try:
            config_dict = self._gather_config()
            # Run the preview logic in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.logic.preview_spectrum, args=(config_dict,), daemon=True).start()
        except ValueError as e:
            self.app_queue.put(('error', f"Invalid input for preview: {e}"))
            self.on_preview_done()
        except Exception as e:
            self.app_queue.put(('error', f"An unexpected error occurred: {e}"))
            self.on_preview_done()

    def generate_and_plot_command(self):
        self.plot_button.config(state=DISABLED)
        try:
            config_dict = self._gather_config()
            if config_dict.get("protein_list_file"):
                messagebox.showwarning("Warning", "Plotting is only available for manually entered proteins, not for file-based batch processing.")
                self.on_plot_done()
                return
            self.logic.start_plot_generation(config_dict, self._handle_plot_result)
        except ValueError as e:
            self.app_queue.put(('error', f"Invalid input: {e}"))
            self.on_plot_done()

    def _handle_plot_result(self, result):
        if result:
            self.app_queue.put(('plot_data', result))
        # If result is None, it means an error occurred during process setup
        # The error message should have already been put on the queue by the logic class
        self.on_plot_done()

    def on_task_done(self):
        self.spectrum_generate_button.config(state=NORMAL)

    def on_preview_done(self):
        self.spectrum_preview_button.config(state=NORMAL)

    def on_plot_done(self):
        self.plot_button.config(state=NORMAL)
