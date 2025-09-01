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
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame, create_labeled_entry
from ...logic.spectrum_logic import SpectrumTabLogic
from .. import constants as C


class SpectrumTab(BaseTab):
    def __init__(self, notebook, style, app_controller=None):
        super().__init__(notebook, style, app_controller=app_controller)
        self.logic = SpectrumTabLogic()

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

        self.protein_list_file_var, self.protein_list_file_entry = create_labeled_entry(
            in_frame,
            C.PROTEIN_LIST_FILE_LABEL,
            C.PROTEIN_LIST_FILE_TOOLTIP,
            grid_row=0, grid_column=0,
        )

        self.protein_list_browse_button = ttk.Button(in_frame, text=C.BROWSE_BUTTON_TEXT, command=self.browse_protein_list, style='Outline.TButton')
        self.protein_list_browse_button.grid(row=0, column=2, sticky="w", pady=5, padx=(5,0))
        Tooltip(self.protein_list_browse_button, C.BROWSE_PROTEIN_LIST_TOOLTIP)

        self.save_template_button = ttk.Button(in_frame, text=C.SAVE_TEMPLATE_BUTTON_TEXT, command=self._save_protein_template, style='Outline.TButton')
        self.save_template_button.grid(row=0, column=3, sticky="w", pady=5, padx=(2,5))
        Tooltip(self.save_template_button, C.SAVE_TEMPLATE_TOOLTIP)

        ttk.Separator(in_frame, orient=HORIZONTAL).grid(row=1, column=0, columnspan=4, sticky="ew", pady=10)
        ttk.Label(in_frame, text=C.MANUAL_INPUT_LABEL, bootstyle="secondary").grid(row=2, column=0, columnspan=4)

        self.spectrum_protein_masses_var, self.spectrum_protein_masses_entry = create_labeled_entry(
            in_frame, C.PROTEIN_MASSES_LABEL,
            C.PROTEIN_MASSES_TOOLTIP,
            default_value="25000", grid_row=3
        )
        self.spectrum_protein_masses_entry.grid(columnspan=3)

        self.intensity_scalars_var, self.intensity_scalars_entry = create_labeled_entry(
            in_frame, C.INTENSITY_SCALARS_LABEL,
            C.INTENSITY_SCALARS_TOOLTIP,
            default_value="1.0", grid_row=4
        )
        self.intensity_scalars_entry.grid(columnspan=3)

        self.mass_inhomogeneity_var, self.mass_inhomogeneity_entry = create_labeled_entry(
            in_frame, C.MASS_INHOMOGENEITY_LABEL,
            C.MASS_INHOMOGENEITY_TOOLTIP,
            default_value="0.0", grid_row=5
        )
        self.mass_inhomogeneity_entry.grid(columnspan=3)

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
        self.spectrum_preview_button = ttk.Button(button_frame, text=C.PREVIEW_SPECTRUM_BUTTON_TEXT, command=self.preview_spectrum_command, style='Outline.TButton')
        self.spectrum_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_preview_button, C.PREVIEW_SPECTRUM_TOOLTIP)

        self.plot_button = ttk.Button(button_frame, text=C.GENERATE_PLOT_BUTTON_TEXT, command=self.generate_and_plot_command, style='Outline.TButton')
        self.plot_button.pack(side=LEFT, padx=5)
        Tooltip(self.plot_button, C.GENERATE_PLOT_TOOLTIP)

        self.spectrum_generate_button = ttk.Button(button_frame, text=C.GENERATE_MZML_BUTTON_TEXT, command=self.generate_spectrum_command, bootstyle=PRIMARY)
        self.spectrum_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_generate_button, C.GENERATE_MZML_TOOLTIP)

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
            "protein_masses_str": self.spectrum_protein_masses_var.get(),
            "intensity_scalars_str": self.intensity_scalars_var.get(),
            "mass_inhomogeneity_str": self.mass_inhomogeneity_var.get(),
        }

    def _save_protein_template(self):
        try:
            config_dict = self._gather_config()
            masses, scalars = self.logic.validate_and_prepare_template_data(
                config_dict['protein_masses_str'],
                config_dict['intensity_scalars_str'],
                self.task_queue
            )

            filepath = filedialog.asksaveasfilename(
                title=C.SAVE_TEMPLATE_TITLE,
                initialfile=C.PROTEIN_LIST_TEMPLATE_FILENAME,
                defaultextension=".txt",
                filetypes=C.FILE_TYPES
            )
            if not filepath:
                return

            self.logic.save_protein_template(filepath, masses, scalars)

            self.task_queue.put(('log', f"Saved template to {os.path.basename(filepath)}\n"))
            self.protein_list_file_var.set(filepath)
        except ValueError as e:
             messagebox.showerror(C.INVALID_INPUT_ERROR_TITLE, str(e))
        except Exception as e:
            messagebox.showerror(C.SAVE_ERROR_TITLE, C.SAVE_ERROR_MESSAGE.format(e))

    def _toggle_protein_inputs(self, *args):
        state = DISABLED if self.protein_list_file_var.get() else NORMAL
        self.spectrum_protein_masses_entry.config(state=state)
        self.intensity_scalars_entry.config(state=state)
        self.mass_inhomogeneity_entry.config(state=state)

    def browse_protein_list(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd())
        if filepath:
            self.protein_list_file_var.set(filepath)

    def generate_spectrum_command(self):
        self.spectrum_generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.task_queue.put(('clear_log', None))
        try:
            config_dict = self._gather_config()
            self.logic.generate_spectrum(config_dict, self.task_queue)
        except ValueError as e:
            self.task_queue.put(('error', C.INVALID_INPUT_ERROR.format(e)))
            self.on_task_done()

    def preview_spectrum_command(self):
        self.spectrum_preview_button.config(state=DISABLED)
        try:
            config_dict = self._gather_config()
            # Run the preview logic in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.logic.preview_spectrum, args=(config_dict, self.task_queue), daemon=True).start()
        except ValueError as e:
            self.task_queue.put(('error', C.INVALID_PREVIEW_INPUT_ERROR.format(e)))
            self.on_preview_done()
        except Exception as e:
            self.task_queue.put(('error', C.UNEXPECTED_ERROR_MESSAGE.format(e)))
            self.on_preview_done()

    def generate_and_plot_command(self):
        self.plot_button.config(state=DISABLED)
        try:
            config_dict = self._gather_config()
            if config_dict.get("protein_list_file"):
                messagebox.showwarning(C.PLOT_WARNING_TITLE, C.PLOT_WARNING_MESSAGE)
                self.on_plot_done()
                return
            self.logic.start_plot_generation(config_dict, self.task_queue, self._handle_plot_result)
        except ValueError as e:
            self.task_queue.put(('error', C.INVALID_INPUT_ERROR.format(e)))
            self.on_plot_done()

    def _handle_plot_result(self, result):
        if result and self.app_controller:
            plot_viewer = self.app_controller.get_plot_viewer()
            if plot_viewer:
                plot_viewer.plot_data(result)
                self.app_controller.switch_to_plot_viewer()
        # If result is None, it means an error occurred during process setup
        # The error message should have already been put on the queue by the logic class
        self.on_plot_done()

    def on_task_done(self):
        self.spectrum_generate_button.config(state=NORMAL)

    def on_preview_done(self):
        self.spectrum_preview_button.config(state=NORMAL)

    def on_plot_done(self):
        self.plot_button.config(state=NORMAL)
