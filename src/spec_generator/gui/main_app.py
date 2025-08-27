import csv
import multiprocessing
import os
import queue
import random
import threading
from datetime import datetime
from functools import partial
from tkinter import (HORIZONTAL, LEFT, BOTH, E, EW, W, X, Y, Button, Canvas,
                     Checkbutton, Entry, Frame, Label, LabelFrame, Scrollbar,
                     StringVar, Toplevel, messagebox, filedialog, Text, END,
                     NORMAL, DISABLED, Tk, BooleanVar, NSEW, SUNKEN, WORD, FLAT)

import matplotlib.pyplot as plt
import numpy as np
from ttkbootstrap import Style
from ttkbootstrap.constants import PRIMARY
from ttkbootstrap import widgets as ttk

from ..core.constants import BASE_INTENSITY_SCALAR, noise_presets
from ..core.lc import apply_lc_profile_and_noise
from ..core.spectrum import (generate_binding_spectrum,
                               generate_protein_spectrum)
from ..logic.simulation import execute_simulation_and_write_mzml
from ..utils.file_io import (format_filename, read_compound_list_file,
                               read_protein_list_file)
from ..utils.ui_helpers import (ScrollableFrame, Tooltip, parse_float_entry,
                                  parse_range_entry)
from ..workers.tasks import run_binding_task, run_simulation_task


class CombinedSpectrumSequenceApp:
    def __init__(self, master: Tk):
        self.master = master
        master.title("Simulated Spectrum Generator (Genedata Expressionist - 27Jun2025 v3.6-refactored)")
        try:
            self.style = Style(theme="solar")
        except Exception:
            self.style = Style(theme="litera")

        self.queue = queue.Queue()
        self.process_queue()

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        docs_tab = ttk.Frame(self.notebook)
        self.notebook.add(docs_tab, text="Overview & Docs")
        self.create_docs_tab_content(docs_tab)

        spectrum_tab = ttk.Frame(self.notebook)
        self.notebook.add(spectrum_tab, text="Spectrum Generator")
        self.create_spectrum_generator_tab_content(spectrum_tab)

        binding_tab = ttk.Frame(self.notebook)
        self.notebook.add(binding_tab, text="Covalent Binding")
        self.create_binding_spectra_tab_content(binding_tab)

        master.minsize(650, 600)

    def create_docs_tab_content(self, tab):
        """Creates the content for the documentation and overview tab."""
        frame = ScrollableFrame(tab)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        text_widget = Text(frame.scrollable_frame, wrap=WORD, relief=FLAT, background=self.style.colors.bg)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Define text styles ---
        text_widget.tag_configure("h1", font=("Helvetica", 16, "bold"), spacing3=10)
        text_widget.tag_configure("h2", font=("Helvetica", 12, "bold"), spacing3=8)
        text_widget.tag_configure("bold", font=("Helvetica", 10, "bold"))
        text_widget.tag_configure("body", font=("Helvetica", 10), lmargin1=10, lmargin2=10, spacing1=5)

        # --- Add content ---
        docs_content = [
            ("Spectrum Simulator (v3.6-refactored)\n\n", "h1"),
            ("This tool is designed to generate realistic, simulated mass spectrometry data (.mzML files) for intact proteins and covalent binding screens.\n\n", "body"),

            ("How it Works: The Core Model\n", "h2"),
            ("1. Isotopic Distribution:", "bold"),
            (" The simulation starts by calculating the theoretical isotopic distribution of a given mass using a Poisson distribution based on the 'averagine' model. It correctly uses the user-provided ", "body"),
            ("Average Mass", "bold"),
            (" to derive the corresponding monoisotopic mass.\n", "body"),
            ("2. Charge State Envelope:", "bold"),
            (" It then calculates a realistic charge state envelope. The charge states are centered around the most abundant isotope, ensuring the final deconvoluted mass matches the input average mass.\n", "body"),
            ("3. Peak Generation:", "bold"),
            (" For each isotope in each charge state, a Gaussian peak is generated. The final width of the peak is a combination of the ", "body"),
            ("Instrument Resolution", "bold"),
            (" (which determines the m/z-dependent broadening) and the ", "body"),
            ("Intrinsic Peak Sigma", "bold"),
            (" (which models constant physical effects like Doppler broadening).\n", "body"),
            ("4. Noise Simulation:", "bold"),
            (" Several layers of noise are added for realism, including chemical noise (low m/z 'haystack'), white electronic noise, signal-dependent shot noise, and a low-frequency baseline wobble. The noise generation is highly optimized using Numba.\n\n", "body"),

            ("Tabs Overview\n", "h2"),
            ("Spectrum Generator Tab\n", "bold"),
            ("This tab is for generating spectra for one or more proteins. You can enter masses manually as a comma-separated list or provide a tab-delimited file with 'Protein' and 'Intensity' columns. The 'Save as Template...' button provides an easy way to create a valid file from the manual inputs.\n\n", "body"),
            ("Covalent Binding Tab\n", "bold"),
            ("This tab simulates a covalent binding screen. It takes a single protein average mass and a list of compounds (from a file with 'Name' and 'Delta' columns). For each compound, it probabilistically determines if binding occurs (and to what extent) and generates a corresponding spectrum containing native, singly-adducted (DAR-1), and doubly-adducted (DAR-2) species.\n\n", "body"),

            ("Advanced Parameters\n", "h2"),
            ("Mass Inhomogeneity:", "bold"),
            (" In the 'Spectrum Generator' tab, this parameter models the natural heterogeneity of large molecules. It applies a Gaussian distribution to the input mass, simulating effects like conformational changes that broaden the final deconvoluted peak.\n", "body"),
            ("LC Tailing Factor (tau):", "bold"),
            (" This parameter, found in the 'LC Simulation' frame, controls the shape of the chromatographic peak. A value of 0 results in a perfect Gaussian peak, while larger values create an 'Exponentially Modified Gaussian' (EMG) with a more realistic tail, common in 'bind and elute' experiments.\n", "body"),
            ("1/f (Pink) Noise:", "bold"),
            (" This checkbox in the 'Noise & Output' frame adds an additional layer of 1/f noise, which can more accurately simulate noise from some electronic components.\n\n", "body"),

            ("Performance & Dependencies\n", "h2"),
            ("This application is highly optimized for performance and uses multiple CPU cores for batch processing. To run, it requires several external libraries. You can install them all with pip from the `requirements.txt` file:\n", "body"),
            ("pip install -r requirements.txt", "bold"),

        ]

        for text, tag in docs_content:
            text_widget.insert(END, text, tag)

        text_widget.config(state=DISABLED) # Make text read-only

    def process_queue(self):
        """
        Processes messages from the background threads to update the GUI.
        """
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                active_tab = self.notebook.index(self.notebook.select())

                # Determine which tab's widgets to update
                if active_tab == 1: # Spectrum Generator
                    out_text, prog_bar = self.spectrum_output_text, self.progress_bar
                elif active_tab == 2: # Covalent Binding
                    out_text, prog_bar = self.binding_output_text, self.binding_progress_bar
                else: # Docs tab has no progress bar or output text
                    continue

                if msg_type == 'log':
                    out_text.insert(END, msg_data)
                    out_text.see(END)
                elif msg_type == 'clear_log':
                    out_text.delete('1.0', END)
                elif msg_type == 'progress_set':
                    prog_bar["value"] = msg_data
                elif msg_type == 'progress_add':
                    prog_bar["value"] += msg_data
                elif msg_type == 'error':
                    messagebox.showerror("Error", msg_data)
                    prog_bar["value"] = 0
                elif msg_type == 'warning':
                    messagebox.showwarning("Warning", msg_data)
                elif msg_type == 'done':
                    if active_tab == 1:
                        self.spectrum_generate_button.config(state=NORMAL)
                    else:
                        self.binding_generate_button.config(state=NORMAL)
                    if msg_data:
                        messagebox.showinfo("Complete", msg_data)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)

    def _show_plot(self, mz_range, intensity_data: dict, title: str, xlabel="m/z", ylabel="Intensity"):
        """
        Displays a matplotlib plot of the spectrum.
        """
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(10, 6))

            for label, data in intensity_data.items():
                ax.plot(mz_range, data, label=label, lw=1.5)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            if len(intensity_data) > 1:
                ax.legend()

            ax.grid(True)
            fig.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Plotting Error", f"Failed to show plot. Ensure matplotlib is installed correctly.\nError: {e}")

    def _browse_directory_for_var(self, string_var: StringVar):
        """
        Opens a directory browser and sets the selected path to a StringVar.
        """
        directory = filedialog.askdirectory(initialdir=string_var.get())
        if directory:
            string_var.set(directory)

    def _create_common_parameters_frame(self, parent, mz_start, mz_end, noise):
        params = {}

        # --- m/z Parameters Frame ---
        mz_frame = ttk.LabelFrame(parent, text="m/z Parameters", padding=(15, 10))
        mz_frame.grid(row=0, column=0, sticky=EW, padx=10, pady=5)

        params['isotopic_enabled_var'] = BooleanVar(value=True)
        params['isotopic_enabled_check'] = ttk.Checkbutton(mz_frame, text="Enable Isotopic Distribution", variable=params['isotopic_enabled_var'], bootstyle="primary-round-toggle")
        params['isotopic_enabled_check'].grid(row=0, column=0, columnspan=2, sticky=W, pady=5)
        Tooltip(params['isotopic_enabled_check'], "If enabled, simulates the isotopic distribution for each charge state.\nIf disabled, generates a single peak for each charge state.")

        ttk.Label(mz_frame, text="Instrument Resolution (in thousands):").grid(row=1, column=0, sticky=W, pady=2)
        params['resolution_entry'] = ttk.Entry(mz_frame, width=10)
        params['resolution_entry'].insert(0, "120")
        params['resolution_entry'].grid(row=1, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['resolution_entry'], "The resolving power of the mass analyzer (e.g., 120 for 120,000).\nAffects the width of the generated peaks.")

        ttk.Label(mz_frame, text="Intrinsic Peak Sigma (at 1000 m/z):").grid(row=2, column=0, sticky=W, pady=2)
        params['peak_sigma_mz_entry'] = ttk.Entry(mz_frame, width=10)
        params['peak_sigma_mz_entry'].insert(0, "0.01")
        params['peak_sigma_mz_entry'].grid(row=2, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['peak_sigma_mz_entry'], "The 'natural' peak width (standard deviation) from effects like Doppler broadening,\nindependent of instrument resolution. A small value is recommended.")

        ttk.Label(mz_frame, text="m/z Step:").grid(row=3, column=0, sticky=W, pady=2)
        params['mz_step_entry'] = ttk.Entry(mz_frame, width=10)
        params['mz_step_entry'].insert(0, "0.02")
        params['mz_step_entry'].grid(row=3, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['mz_step_entry'], "The distance between data points in the m/z axis.")

        ttk.Label(mz_frame, text="m/z Range Start:").grid(row=4, column=0, sticky=W, pady=2)
        params['mz_range_start_entry'] = ttk.Entry(mz_frame, width=10)
        params['mz_range_start_entry'].insert(0, mz_start)
        params['mz_range_start_entry'].grid(row=4, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['mz_range_start_entry'], "The minimum m/z value for the spectrum.")

        ttk.Label(mz_frame, text="m/z Range End:").grid(row=5, column=0, sticky=W, pady=2)
        params['mz_range_end_entry'] = ttk.Entry(mz_frame, width=10)
        params['mz_range_end_entry'].insert(0, mz_end)
        params['mz_range_end_entry'].grid(row=5, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['mz_range_end_entry'], "The maximum m/z value for the spectrum.")

        # --- Noise & Output Frame ---
        out_frame = ttk.LabelFrame(parent, text="Noise & Output", padding=(15, 10))
        out_frame.grid(row=1, column=0, sticky=EW, padx=10, pady=5)
        out_frame.columnconfigure(1, weight=1)

        ttk.Label(out_frame, text="Noise Level:").grid(row=0, column=0, sticky=W, pady=2)
        params['noise_option_var'] = StringVar(value=noise)
        params['noise_option_combobox'] = ttk.Combobox(out_frame, textvariable=params['noise_option_var'], values=["No Noise"] + list(noise_presets.keys()), state="readonly", width=15)
        params['noise_option_combobox'].grid(row=0, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['noise_option_combobox'], "Select a preset for the type and amount of noise to add to the spectrum.")

        params['pink_noise_enabled_var'] = BooleanVar(value=False)
        params['pink_noise_enabled_check'] = ttk.Checkbutton(out_frame, text="Enable 1/f Noise", variable=params['pink_noise_enabled_var'], bootstyle="primary-round-toggle")
        params['pink_noise_enabled_check'].grid(row=0, column=2, sticky=W, padx=5)
        Tooltip(params['pink_noise_enabled_check'], "Adds an additional layer of 1/f (pink) noise, which is common in electronic systems.")

        ttk.Label(out_frame, text="Output Directory:").grid(row=1, column=0, sticky=W, pady=2)
        params['output_directory_var'] = StringVar(value=os.getcwd())
        params['output_directory_entry'] = ttk.Entry(out_frame, textvariable=params['output_directory_var'])
        params['output_directory_entry'].grid(row=1, column=1, sticky=EW, pady=2, padx=5)
        Tooltip(params['output_directory_entry'], "The folder where the generated .mzML files will be saved.")

        params['browse_button'] = ttk.Button(out_frame, text="Browse...", command=partial(self._browse_directory_for_var, params['output_directory_var']), style='Outline.TButton')
        params['browse_button'].grid(row=1, column=2, sticky=W, pady=2, padx=5)
        Tooltip(params['browse_button'], "Browse for an output directory.")

        ttk.Label(out_frame, text="Random Seed:").grid(row=2, column=0, sticky=W, pady=2)
        params['seed_var'] = StringVar(value="")
        params['seed_entry'] = ttk.Entry(out_frame, textvariable=params['seed_var'], width=15)
        params['seed_entry'].grid(row=2, column=1, sticky=W, pady=2, padx=5)
        Tooltip(params['seed_entry'], "Seed for the random number generator to ensure reproducible noise.\nLeave blank for a different random seed each time.")
        ttk.Label(out_frame, text="(Leave blank for random)").grid(row=2, column=2, sticky=W, pady=2, padx=5)

        ttk.Label(out_frame, text="Filename Template:").grid(row=3, column=0, sticky=W, pady=2)
        params['filename_template_var'] = StringVar()
        params['filename_template_entry'] = ttk.Entry(out_frame, textvariable=params['filename_template_var'])
        params['filename_template_entry'].grid(row=3, column=1, columnspan=2, sticky=EW, pady=2, padx=5)
        Tooltip(params['filename_template_entry'], "Define the pattern for output filenames using available tags.")

        placeholder_text = "Tags: {date} {time} {protein_mass} {compound_name} {num_proteins} {scalar} {scans} {noise} {seed}"
        ttk.Label(out_frame, text=placeholder_text, wraplength=350, justify=LEFT, bootstyle="secondary").grid(row=4, column=1, columnspan=2, sticky=W, padx=5)

        return params

    def _create_lc_simulation_frame(self, parent, enabled_by_default=False):
        lc_params = {}
        container = ttk.Frame(parent)

        lc_params['enabled_var'] = BooleanVar(value=enabled_by_default)
        lc_check = ttk.Checkbutton(container, text="Enable LC Simulation", variable=lc_params['enabled_var'], bootstyle="primary-round-toggle")
        lc_check.grid(row=0, column=0, sticky=W, pady=(0, 5))
        Tooltip(lc_check, "If enabled, simulates a chromatographic peak by generating multiple scans with varying intensity.\nIf disabled, generates a single scan (spectrum).")

        lc_frame = ttk.LabelFrame(container, text="LC Simulation Parameters", padding=(15, 10))
        lc_frame.grid(row=1, column=0, sticky=EW)

        ttk.Label(lc_frame, text="Number of Scans:").grid(row=0, column=0, sticky=W, pady=2)
        lc_params['num_scans_entry'] = ttk.Entry(lc_frame, width=10)
        lc_params['num_scans_entry'].insert(0, "10")
        lc_params['num_scans_entry'].grid(row=0, column=1, sticky=W, pady=2, padx=5)
        Tooltip(lc_params['num_scans_entry'], "The total number of scans to generate across the LC peak.")

        ttk.Label(lc_frame, text="Scan Interval (min):").grid(row=1, column=0, sticky=W, pady=2)
        lc_params['scan_interval_entry'] = ttk.Entry(lc_frame, width=10)
        lc_params['scan_interval_entry'].insert(0, "0.05")
        lc_params['scan_interval_entry'].grid(row=1, column=1, sticky=W, pady=2, padx=5)
        Tooltip(lc_params['scan_interval_entry'], "The simulated time between consecutive scans.")

        ttk.Label(lc_frame, text="LC Peak Std Dev (scans):").grid(row=2, column=0, sticky=W, pady=2)
        lc_params['gaussian_std_dev_entry'] = ttk.Entry(lc_frame, width=10)
        lc_params['gaussian_std_dev_entry'].insert(0, "1")
        lc_params['gaussian_std_dev_entry'].grid(row=2, column=1, sticky=W, pady=2, padx=5)
        Tooltip(lc_params['gaussian_std_dev_entry'], "The width (standard deviation) of the Gaussian component of the LC peak, in units of scans.")

        ttk.Label(lc_frame, text="LC Tailing Factor (tau):").grid(row=3, column=0, sticky=W, pady=2)
        lc_params['lc_tailing_factor_entry'] = ttk.Entry(lc_frame, width=10)
        lc_params['lc_tailing_factor_entry'].insert(0, "0.0")
        lc_params['lc_tailing_factor_entry'].grid(row=3, column=1, sticky=W, pady=2, padx=5)
        Tooltip(lc_params['lc_tailing_factor_entry'], "The exponential tailing factor (tau) for an Exponentially Modified Gaussian peak shape.\nSet to 0.0 for a pure Gaussian peak.")

        def _toggle():
            state = NORMAL if lc_params['enabled_var'].get() else DISABLED
            for w in lc_frame.winfo_children():
                w.configure(state=state)

        lc_check.config(command=_toggle)
        _toggle() # Set initial state
        return container, lc_params

    def create_spectrum_generator_tab_content(self, tab):
        frame = ScrollableFrame(tab)
        frame.pack(fill="both", expand=True)
        main = frame.scrollable_frame

        # --- Protein Input Frame ---
        in_frame = ttk.LabelFrame(main, text="Protein Parameters", padding=(15, 10))
        in_frame.grid(row=0, column=0, sticky=EW, padx=10, pady=10)
        in_frame.columnconfigure(1, weight=1)

        ttk.Label(in_frame, text="Protein List File (.txt):").grid(row=0, column=0, sticky=W, pady=5)
        self.protein_list_file_var = StringVar()
        self.protein_list_file_entry = ttk.Entry(in_frame, textvariable=self.protein_list_file_var)
        self.protein_list_file_entry.grid(row=0, column=1, sticky=EW, pady=5, padx=5)
        Tooltip(self.protein_list_file_entry, "Path to a tab-delimited file with protein data.\nMust contain 'Protein' (Average Mass) and 'Intensity' headers.")

        self.protein_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_protein_list, style='Outline.TButton')
        self.protein_list_browse_button.grid(row=0, column=2, sticky=W, pady=5, padx=(5,0))
        Tooltip(self.protein_list_browse_button, "Browse for a protein list file.")

        self.save_template_button = ttk.Button(in_frame, text="Save as Template...", command=self._save_protein_template, style='Outline.TButton')
        self.save_template_button.grid(row=0, column=3, sticky=W, pady=5, padx=(2,5))
        Tooltip(self.save_template_button, "Save the manually entered masses and scalars below as a valid\ntab-delimited template file for future use.")

        ttk.Separator(in_frame, orient=HORIZONTAL).grid(row=1, column=0, columnspan=4, sticky=EW, pady=10)
        ttk.Label(in_frame, text="OR Enter Manually Below", bootstyle="secondary").grid(row=2, column=0, columnspan=4)

        ttk.Label(in_frame, text="Protein Avg. Masses (Da, comma-sep):").grid(row=3, column=0, sticky=W, pady=5)
        self.spectrum_protein_masses_entry = ttk.Entry(in_frame)
        self.spectrum_protein_masses_entry.insert(0, "25000")
        self.spectrum_protein_masses_entry.grid(row=3, column=1, columnspan=3, sticky=EW, pady=5, padx=5)
        Tooltip(self.spectrum_protein_masses_entry, "A comma-separated list of protein AVERAGE masses to simulate.")

        ttk.Label(in_frame, text="Intensity Scalars (comma-sep):").grid(row=4, column=0, sticky=W, pady=5)
        self.intensity_scalars_entry = ttk.Entry(in_frame)
        self.intensity_scalars_entry.insert(0, "1.0")
        self.intensity_scalars_entry.grid(row=4, column=1, columnspan=3, sticky=EW, pady=5, padx=5)
        Tooltip(self.intensity_scalars_entry, "A comma-separated list of relative intensity multipliers.\nMust match the number of protein masses.")

        ttk.Label(in_frame, text="Mass Inhomogeneity (Std. Dev., Da):").grid(row=5, column=0, sticky=W, pady=5)
        self.mass_inhomogeneity_entry = ttk.Entry(in_frame)
        self.mass_inhomogeneity_entry.insert(0, "0.0")
        self.mass_inhomogeneity_entry.grid(row=5, column=1, columnspan=3, sticky=EW, pady=5, padx=5)
        Tooltip(self.mass_inhomogeneity_entry, "Standard deviation of protein mass distribution to simulate conformational broadening.\nSet to 0 to disable. A small value (e.g., 1-5 Da) is recommended.")

        # --- Common Parameters ---
        common_frame = ttk.Frame(main)
        common_frame.grid(row=1, column=0, sticky=EW, padx=10, pady=0)
        self.spec_gen_params = self._create_common_parameters_frame(common_frame, "400.0", "2500.0", "Default Noise")
        self.spec_gen_params['output_directory_var'].set(os.path.join(os.getcwd(), "Mzml Mock Spectra"))
        self.spec_gen_params['filename_template_var'].set("{date}_protein_{protein_mass}_{scans}scans_{noise}.mzML")

        # --- LC Simulation ---
        lc_container, self.spec_gen_lc_params = self._create_lc_simulation_frame(main, False)
        lc_container.grid(row=2, column=0, sticky=EW, padx=10, pady=10)

        # --- Action Buttons ---
        button_frame = ttk.Frame(main)
        button_frame.grid(row=3, column=0, pady=15)
        self.spectrum_preview_button = ttk.Button(button_frame, text="Preview Spectrum", command=self._preview_spectrum_tab_command, style='Outline.TButton')
        self.spectrum_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_preview_button, "Generate and display a plot of a single spectrum using the current settings.\nUses the first protein mass if multiple are entered.")

        self.spectrum_generate_button = ttk.Button(button_frame, text="Generate mzML File(s)", command=self.generate_spectrum_tab_command, bootstyle=PRIMARY)
        self.spectrum_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.spectrum_generate_button, "Generate and save .mzML file(s) with the specified parameters.")

        # --- Progress & Output ---
        self.progress_bar = ttk.Progressbar(main, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=4, column=0, pady=5, sticky=EW, padx=10)

        out_frame = ttk.Frame(main)
        out_frame.grid(row=5, column=0, sticky=NSEW, padx=10, pady=(5, 10))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)
        self.spectrum_output_text = Text(out_frame, height=10, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.spectrum_output_text.grid(row=0, column=0, sticky=NSEW)
        scrollbar = ttk.Scrollbar(out_frame, command=self.spectrum_output_text.yview)
        scrollbar.grid(row=0, column=1, sticky=NSEW)
        self.spectrum_output_text['yscrollcommand'] = scrollbar.set

        # --- Bindings ---
        self.protein_list_file_var.trace_add("write", self._toggle_protein_inputs)
        self._toggle_protein_inputs()

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

            self.queue.put(('log', f"Saved template to {os.path.basename(filepath)}\n"))
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
        self.queue.put(('clear_log', None))

        # Decide which worker to run based on whether a file is selected
        use_file_input = self.protein_list_file_var.get()
        worker = self._worker_generate_from_protein_file if use_file_input else self._worker_generate_from_manual_input

        # Start the worker in a new thread to keep the GUI responsive
        threading.Thread(target=worker, daemon=True).start()

    def _get_common_gen_params(self, params_dict, lc_params_dict):
        """
        Gathers and validates all common parameters from the GUI widgets.
        This is a pure data-gathering function.
        """
        params = {}
        # Get values from StringVars and Entry widgets
        # Safely get pink noise status, as this frame is shared
        params['pink_noise_enabled'] = params_dict.get('pink_noise_enabled_var', BooleanVar(value=False)).get()

        for key, widget in params_dict.items():
            if 'var' in key:
                params[key.replace('_var', '')] = widget.get()
            elif 'entry' in key:
                params[key.replace('_entry', '')] = widget.get()

        # Get LC parameters
        params['lc_simulation_enabled'] = lc_params_dict['enabled_var'].get()
        if params['lc_simulation_enabled']:
            params['num_scans'] = int(lc_params_dict['num_scans_entry'].get())
            params['scan_interval'] = float(lc_params_dict['scan_interval_entry'].get())
            params['gaussian_std_dev'] = float(lc_params_dict['gaussian_std_dev_entry'].get())
            params['lc_tailing_factor'] = float(lc_params_dict['lc_tailing_factor_entry'].get())
        else:
            params['num_scans'], params['scan_interval'], params['gaussian_std_dev'], params['lc_tailing_factor'] = 1, 0.0, 0.0, 0.0

        # Handle random seed
        seed_str = params_dict['seed_var'].get().strip()
        if seed_str:
            params['seed'] = int(seed_str)
        else:
            params['seed'] = random.randint(0, 2**32 - 1)
            # Update the GUI to show the seed that was used
            params_dict['seed_var'].set(str(params['seed']))

        params['resolution'] = float(params_dict['resolution_entry'].get()) * 1000
        return params

    def _worker_generate_from_protein_file(self):
        """
        Worker thread for generating spectra from a protein list file.
        Uses multiprocessing for the actual generation.
        """
        try:
            protein_list = read_protein_list_file(self.protein_list_file_var.get())
            common_params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params)
            common_params['mass_inhomogeneity'] = parse_float_entry(self.mass_inhomogeneity_entry.get(), "Mass Inhomogeneity")
        except (ValueError, FileNotFoundError) as e:
            self.queue.put(('error', str(e)))
            self.queue.put(('done', None))
            return

        # Create a job for each protein in the list
        jobs = [(mass, scalar, common_params, common_params['seed'] + i) for i, (mass, scalar) in enumerate(protein_list)]

        self.queue.put(('log', f"Starting batch generation for {len(jobs)} proteins using {os.cpu_count()} processes...\n\n"))
        self.progress_bar["maximum"] = len(jobs)
        self.progress_bar["value"] = 0

        try:
            # Use a process pool to run jobs in parallel
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_simulation_task, jobs)):
                    self.queue.put(('log', message))
                    if success:
                        success_count += 1
                    self.queue.put(('progress_set', i + 1))
            self.queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} mzML files."))
        except Exception as e:
            self.queue.put(('error', f"A multiprocessing error occurred: {e}"))
            self.queue.put(('done', None))

    def _worker_generate_from_manual_input(self):
        """
        Worker thread for generating a single spectrum from manually entered data.
        """
        try:
            common_params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params)
            common_params['mass_inhomogeneity'] = parse_float_entry(self.mass_inhomogeneity_entry.get(), "Mass Inhomogeneity")
            mass_str = self.spectrum_protein_masses_entry.get()
            mass_list = [m.strip() for m in mass_str.split(',') if m.strip()]
            scalar_str = self.intensity_scalars_entry.get()
            scalar_list = [float(s.strip()) for s in scalar_str.split(',') if s.strip()]

            if not mass_list:
                raise ValueError("No protein masses entered.")
            if not scalar_list:
                scalar_list = [1.0] * len(mass_list)
            if len(scalar_list) != len(mass_list):
                self.queue.put(('warning', "Mismatched scalars and masses. Adjusting..."))
                scalar_list = (scalar_list + [1.0] * len(mass_list))[:len(mass_list)]
        except ValueError as e:
            self.queue.put(('error', f"Invalid input: {e}"))
            self.queue.put(('done', None))
            return

        # Prepare filename
        avg_mass = float(mass_list[0])
        placeholders = {
            "date": datetime.now().strftime('%Y-%m-%d'), "time": datetime.now().strftime('%H%M%S'),
            "num_proteins": len(mass_list), "scans": common_params['num_scans'],
            "noise": common_params['noise_option'].replace(" ", ""), "seed": common_params['seed'],
            "protein_mass": int(round(avg_mass))
        }
        filename = format_filename(common_params['filename_template_var'].get(), placeholders)
        filepath = os.path.join(common_params['output_directory'], filename)

        # Call the refactored logic function directly
        success = execute_simulation_and_write_mzml(
            update_queue=self.queue,
            protein_masses_str=mass_str,
            mz_step_str=common_params['mz_step'],
            peak_sigma_mz_str=common_params['peak_sigma_mz'],
            mz_range_start_str=common_params['mz_range_start'],
            mz_range_end_str=common_params['mz_range_end'],
            intensity_scalars=scalar_list,
            noise_option=common_params['noise_option'],
            seed=common_params['seed'],
            lc_simulation_enabled=common_params['lc_simulation_enabled'],
            num_scans=common_params['num_scans'],
            scan_interval=common_params['scan_interval'],
            gaussian_std_dev=common_params['gaussian_std_dev'],
            final_filepath=filepath,
            isotopic_enabled=common_params['isotopic_enabled'],
            resolution=common_params['resolution']
        )

        if success:
            self.queue.put(('done', "mzML file successfully created."))
        else:
            self.queue.put(('done', None))

    def create_binding_spectra_tab_content(self, tab):
        frame = ScrollableFrame(tab)
        frame.pack(fill="both", expand=True)
        main = frame.scrollable_frame

        # --- Target & Compound Frame ---
        in_frame = ttk.LabelFrame(main, text="Target & Compound", padding=(15, 10))
        in_frame.grid(row=0, column=0, sticky=EW, padx=10, pady=10)
        in_frame.columnconfigure(1, weight=1)

        ttk.Label(in_frame, text="Protein Avg. Mass (Da):").grid(row=0, column=0, sticky=W, pady=5)
        self.binding_protein_mass_entry = ttk.Entry(in_frame)
        self.binding_protein_mass_entry.insert(0, "25000")
        self.binding_protein_mass_entry.grid(row=0, column=1, sticky=EW, pady=5, padx=5)
        Tooltip(self.binding_protein_mass_entry, "The AVERAGE mass of the target protein.")

        ttk.Label(in_frame, text="Compound List File (.txt):").grid(row=1, column=0, sticky=W, pady=5)
        self.compound_list_file_var = StringVar()
        self.compound_list_file_entry = ttk.Entry(in_frame, textvariable=self.compound_list_file_var)
        self.compound_list_file_entry.grid(row=1, column=1, sticky=EW, pady=5, padx=5)
        Tooltip(self.compound_list_file_entry, "Path to a tab-delimited file of compounds to test.\nMust contain 'Name' and 'Delta' (Average Mass) headers.")

        self.compound_list_browse_button = ttk.Button(in_frame, text="Browse...", command=self.browse_compound_list, style='Outline.TButton')
        self.compound_list_browse_button.grid(row=1, column=2, sticky=W, pady=5, padx=5)
        Tooltip(self.compound_list_browse_button, "Browse for a compound list file.")

        # --- Common Parameters ---
        common_frame = ttk.Frame(main)
        common_frame.grid(row=1, column=0, sticky=EW, padx=10, pady=0)
        self.binding_params = self._create_common_parameters_frame(common_frame, "400.0", "2000.0", "Default Noise")
        self.binding_params['output_directory_var'].set(os.path.join(os.getcwd(), "Intact Covalent Binding Mock Spectra"))
        self.binding_params['filename_template_var'].set("{date}_{compound_name}_on_{protein_mass}_{scans}scans_{noise}.mzML")

        # --- LC Simulation ---
        lc_container, self.binding_lc_params = self._create_lc_simulation_frame(main, False)
        lc_container.grid(row=2, column=0, sticky=EW, padx=10, pady=10)

        # --- Binding Probabilities Frame ---
        prob_frame = ttk.LabelFrame(main, text="Binding Probabilities", padding=(15, 10))
        prob_frame.grid(row=3, column=0, sticky=EW, padx=10, pady=10)

        ttk.Label(prob_frame, text="Probability of Binding:").grid(row=0, column=0, sticky=W, pady=2)
        self.prob_binding_entry = ttk.Entry(prob_frame, width=10)
        self.prob_binding_entry.insert(0, "0.5")
        self.prob_binding_entry.grid(row=0, column=1, sticky=W, pady=2, padx=5)
        Tooltip(self.prob_binding_entry, "The chance (0.0 to 1.0) that any given compound will bind to the protein.")

        ttk.Label(prob_frame, text="Probability of DAR-2 (if Binding):").grid(row=1, column=0, sticky=W, pady=2)
        self.prob_dar2_if_binding_entry = ttk.Entry(prob_frame, width=10)
        self.prob_dar2_if_binding_entry.insert(0, "0.1")
        self.prob_dar2_if_binding_entry.grid(row=1, column=1, sticky=W, pady=2, padx=5)
        Tooltip(self.prob_dar2_if_binding_entry, "If a compound binds, this is the chance (0.0 to 1.0)\nthat it will form a doubly-adducted species (DAR-2).")

        ttk.Label(prob_frame, text="Total Binding % Range (if Binding):").grid(row=2, column=0, sticky=W, pady=2)
        self.total_binding_percentage_range_entry = ttk.Entry(prob_frame)
        self.total_binding_percentage_range_entry.insert(0, "10-50")
        self.total_binding_percentage_range_entry.grid(row=2, column=1, sticky=EW, pady=2, padx=5)
        Tooltip(self.total_binding_percentage_range_entry, "If binding occurs, the total percentage of protein that is modified\nwill be randomly chosen from this range (e.g., '10-50').")

        ttk.Label(prob_frame, text="DAR-2 % Range (of Total Bound):").grid(row=3, column=0, sticky=W, pady=2)
        self.dar2_percentage_of_bound_range_entry = ttk.Entry(prob_frame)
        self.dar2_percentage_of_bound_range_entry.insert(0, "5-20")
        self.dar2_percentage_of_bound_range_entry.grid(row=3, column=1, sticky=EW, pady=2, padx=5)
        Tooltip(self.dar2_percentage_of_bound_range_entry, "If DAR-2 occurs, the percentage of the bound protein that is DAR-2\nwill be randomly chosen from this range (e.g., '5-20').")

        # --- Action Buttons ---
        button_frame = ttk.Frame(main)
        button_frame.grid(row=4, column=0, pady=15)
        self.binding_preview_button = ttk.Button(button_frame, text="Preview Binding", command=self._preview_binding_tab_command, style='Outline.TButton')
        self.binding_preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_preview_button, "Generate and display a plot of a binding spectrum using average probability values.")

        self.binding_generate_button = ttk.Button(button_frame, text="Generate Binding Spectra", command=self.generate_binding_spectra_command, bootstyle=PRIMARY)
        self.binding_generate_button.pack(side=LEFT, padx=5)
        Tooltip(self.binding_generate_button, "Generate an .mzML file for each compound in the list,\nwith binding determined by the specified probabilities.")

        # --- Progress & Output ---
        self.binding_progress_bar = ttk.Progressbar(main, orient=HORIZONTAL, mode="determinate")
        self.binding_progress_bar.grid(row=5, column=0, pady=5, sticky=EW, padx=10)

        out_frame = ttk.Frame(main)
        out_frame.grid(row=6, column=0, sticky=NSEW, padx=10, pady=(5, 10))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(0, weight=1)
        main.rowconfigure(6, weight=1)
        self.binding_output_text = Text(out_frame, height=10, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.binding_output_text.grid(row=0, column=0, sticky=NSEW)
        scrollbar = ttk.Scrollbar(out_frame, command=self.binding_output_text.yview)
        scrollbar.grid(row=0, column=1, sticky=NSEW)
        self.binding_output_text['yscrollcommand'] = scrollbar.set

    def browse_compound_list(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt;*.tsv")], initialdir=os.getcwd())
        if filepath:
            self.compound_list_file_var.set(filepath)

    def generate_binding_spectra_command(self):
        self.binding_generate_button.config(state=DISABLED)
        self.binding_progress_bar["value"] = 0
        self.queue.put(('clear_log', None))
        threading.Thread(target=self._worker_generate_binding_spectra, daemon=True).start()

    def _worker_generate_binding_spectra(self):
        """
        Worker thread for generating spectra for a covalent binding screen.
        Uses multiprocessing for the actual generation.
        """
        try:
            # Gather and validate all parameters from the GUI
            common_params = self._get_common_gen_params(self.binding_params, self.binding_lc_params)
            protein_mass = parse_float_entry(self.binding_protein_mass_entry.get(), "Protein Avg. Mass")
            compounds = read_compound_list_file(self.compound_list_file_var.get())

            common_params['prob_binding'] = parse_float_entry(self.prob_binding_entry.get(), "Prob Binding")
            common_params['prob_dar2'] = parse_float_entry(self.prob_dar2_if_binding_entry.get(), "Prob DAR-2")
            common_params['total_binding_range'] = parse_range_entry(self.total_binding_percentage_range_entry.get(), "Total Binding %")
            common_params['dar2_range'] = parse_range_entry(self.dar2_percentage_of_bound_range_entry.get(), "DAR-2 %")
        except (ValueError, FileNotFoundError) as e:
            self.queue.put(('error', str(e)))
            self.queue.put(('done', None))
            return

        # Create a job for each compound in the list
        jobs = [(name, mass, protein_mass, common_params, common_params['seed'] + i) for i, (name, mass) in enumerate(compounds)]

        self.queue.put(('log', f"Starting batch generation for {len(jobs)} compounds using {os.cpu_count()} processes...\n\n"))
        self.binding_progress_bar["maximum"] = len(jobs)
        self.binding_progress_bar["value"] = 0

        try:
            # Use a process pool to run jobs in parallel
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                success_count = 0
                for i, (success, message) in enumerate(pool.imap_unordered(run_binding_task, jobs)):
                    self.queue.put(('log', message))
                    if success:
                        success_count += 1
                    self.queue.put(('progress_set', i + 1))
            self.queue.put(('done', f"Batch complete. Generated {success_count} of {len(jobs)} binding mzML files."))
        except Exception as e:
            self.queue.put(('error', f"A multiprocessing error occurred: {e}"))
            self.queue.put(('done', None))

    def _preview_spectrum_tab_command(self):
        """
        Generates and displays a single spectrum preview without saving a file.
        """
        try:
            # Gather parameters
            params = self._get_common_gen_params(self.spec_gen_params, self.spec_gen_lc_params)
            mass_str = self.spectrum_protein_masses_entry.get()
            mass_list = [float(m.strip()) for m in mass_str.split(',') if m.strip()]
            if not mass_list:
                raise ValueError("Please enter at least one protein mass.")

            protein_avg_mass = mass_list[0]
            mz_range = np.arange(float(params['mz_range_start']), float(params['mz_range_end']) + float(params['mz_step']), float(params['mz_step']))

            # Generate spectrum
            clean_spec = generate_protein_spectrum(
                protein_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']),
                BASE_INTENSITY_SCALAR, params['isotopic_enabled'], params['resolution']
            )

            # Apply LC and noise for the apex scan
            apex_scan_index = (params['num_scans'] - 1) // 2
            apex_scan_spectrum = apply_lc_profile_and_noise(
                mz_range, [clean_spec], params['num_scans'], params['gaussian_std_dev'],
                params['lc_tailing_factor'], params['seed'], params['noise_option'], None
            )[0][apex_scan_index]

            title = f"Preview (Avg Mass: {protein_avg_mass:.0f} Da, Res: {params['resolution']/1000}k)"
            self._show_plot(mz_range, {"Apex Scan Preview": apex_scan_spectrum}, title)

        except (ValueError, IndexError) as e:
            messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e:
            messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")

    def _preview_binding_tab_command(self):
        """
        Generates and displays a single binding spectrum preview without saving.
        """
        try:
            # Gather parameters
            params = self._get_common_gen_params(self.binding_params, self.binding_lc_params)
            protein_avg_mass = parse_float_entry(self.binding_protein_mass_entry.get(), "Protein Avg. Mass")
            compound_avg_mass = 500.0  # Use a fixed default for preview

            total_binding_range = parse_range_entry(self.total_binding_percentage_range_entry.get(), "Total Binding %")
            dar2_range = parse_range_entry(self.dar2_percentage_of_bound_range_entry.get(), "DAR-2 %")

            # Use the average of the ranges for a representative preview
            total_binding_pct = (total_binding_range[0] + total_binding_range[1]) / 2
            dar2_pct_of_bound = (dar2_range[0] + dar2_range[1]) / 2

            mz_range = np.arange(float(params['mz_range_start']), float(params['mz_range_end']) + float(params['mz_step']), float(params['mz_step']))
            isotopic_enabled, resolution = params['isotopic_enabled'], params['resolution']

            # Generate reference spectra for plotting
            base_native_spec = generate_protein_spectrum(protein_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)
            base_dar1_spec = generate_protein_spectrum(protein_avg_mass + compound_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)
            base_dar2_spec = generate_protein_spectrum(protein_avg_mass + 2 * compound_avg_mass, mz_range, float(params['mz_step']), float(params['peak_sigma_mz']), BASE_INTENSITY_SCALAR, isotopic_enabled, resolution)

            # Generate the combined spectrum
            final_spec_clean = generate_binding_spectrum(
                protein_avg_mass, compound_avg_mass, mz_range, float(params['mz_step']),
                float(params['peak_sigma_mz']), total_binding_pct, dar2_pct_of_bound,
                BASE_INTENSITY_SCALAR, isotopic_enabled, resolution
            )

            # Apply LC and noise
            apex_scan_index = (params['num_scans'] - 1) // 2
            final_spec_noisy = apply_lc_profile_and_noise(
                mz_range, [final_spec_clean], params['num_scans'], params['gaussian_std_dev'],
                params['lc_tailing_factor'], params['seed'], params['noise_option'], None
            )[0][apex_scan_index]

            # Prepare data for plotting
            plot_data = {"Combined Spectrum (with noise)": final_spec_noisy}
            if np.any(base_native_spec): plot_data["Native Protein (ref)"] = base_native_spec * 0.5
            if np.any(base_dar1_spec): plot_data["DAR-1 Adduct (ref)"] = base_dar1_spec * 0.5
            if np.any(base_dar2_spec): plot_data["DAR-2 Adduct (ref)"] = base_dar2_spec * 0.5

            title = f"Binding Preview (Target Avg Mass: ~{protein_avg_mass:.0f} Da, Res: {resolution/1000}k)"
            self._show_plot(mz_range, plot_data, title=title)

        except (ValueError, IndexError) as e:
            messagebox.showerror("Preview Error", f"Invalid parameters for preview: {e}")
        except Exception as e:
            messagebox.showerror("Preview Error", f"An unexpected error occurred during preview: {e}")
