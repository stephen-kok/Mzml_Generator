import os
from functools import partial
from tkinter import (StringVar, BooleanVar, filedialog, NORMAL, DISABLED, EW, W)

from ttkbootstrap import widgets as ttk

from ..core.constants import noise_presets
from ..utils.ui_helpers import Tooltip

def create_labeled_entry(parent, label_text, tooltip_text, default_value="", entry_width=None, grid_row=0, grid_column=0, sticky="ew", pady=5, padx=5):
    """
    Creates a label, an entry widget, and a tooltip, and grids them in the parent frame.

    Returns the StringVar associated with the entry widget.
    """
    ttk.Label(parent, text=label_text).grid(row=grid_row, column=grid_column, sticky="w", pady=pady)

    entry_var = StringVar(value=default_value)
    entry = ttk.Entry(parent, textvariable=entry_var, width=entry_width)
    entry.grid(row=grid_row, column=grid_column + 1, sticky=sticky, pady=pady, padx=padx)

    Tooltip(entry, tooltip_text)

    return entry_var, entry


def _browse_directory_for_var(string_var: StringVar):
    """
    Opens a directory browser and sets the selected path to a StringVar.
    """
    directory = filedialog.askdirectory(initialdir=string_var.get())
    if directory:
        string_var.set(directory)

def create_common_parameters_frame(parent, mz_start, mz_end, noise="No Noise"):
    """
    Creates and returns a frame containing common simulation parameter widgets.
    """
    params = {}

    # --- m/z Parameters Frame ---
    mz_frame = ttk.LabelFrame(parent, text="m/z Parameters", padding=(15, 10))
    mz_frame.grid(row=0, column=0, sticky=EW, padx=10, pady=5)

    params['isotopic_enabled_var'] = BooleanVar(value=False)
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

    params['browse_button'] = ttk.Button(out_frame, text="Browse...", command=partial(_browse_directory_for_var, params['output_directory_var']), style='Outline.TButton')
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
    ttk.Label(out_frame, text=placeholder_text, wraplength=350, justify="left", bootstyle="secondary").grid(row=4, column=1, columnspan=2, sticky=W, padx=5)

    return params

def create_lc_simulation_frame(parent, enabled_by_default=False):
    """
    Creates and returns a frame containing LC simulation parameter widgets.
    """
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


# --- PTM Editor Widget ---
import tkinter as tk
from ..logic.ptm import DEFAULT_PTMS, Ptm

class PtmEditor(ttk.Frame):
    """
    A reusable widget for configuring a list of Post-Translational Modifications (PTMs).
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.ptm_vars = {}

        # Create a header
        ttk.Label(self, text="Name", font="-weight bold").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        ttk.Label(self, text="Residue", font="-weight bold").grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(self, text="Enabled", font="-weight bold").grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(self, text="Probability", font="-weight bold").grid(row=0, column=3, padx=5, pady=2)

        row = 1
        for name, ptm_template in DEFAULT_PTMS.items():
            enabled_var = BooleanVar(value=False)
            # Use the probability from the template if it exists and is > 0, else default to something sensible
            default_prob = ptm_template.probability if hasattr(ptm_template, 'probability') and ptm_template.probability > 0 else 0.1
            prob_var = tk.DoubleVar(value=default_prob)

            self.ptm_vars[name] = {
                "template": ptm_template,
                "enabled": enabled_var,
                "prob": prob_var
            }

            ttk.Label(self, text=name).grid(row=row, column=0, sticky="w", padx=5)
            ttk.Label(self, text=f"({ptm_template.residue})").grid(row=row, column=1)

            check = ttk.Checkbutton(self, variable=enabled_var, bootstyle="primary-round-toggle")
            check.grid(row=row, column=2, padx=5)

            spinbox = ttk.Spinbox(self, from_=0.0, to=1.0, increment=0.05, textvariable=prob_var, width=6)
            spinbox.grid(row=row, column=3, padx=5)

            # Disable spinbox if not checked
            def _toggle_spinbox_state(sv=spinbox, en_var=enabled_var):
                state = NORMAL if en_var.get() else DISABLED
                sv.configure(state=state)

            check.config(command=_toggle_spinbox_state)
            _toggle_spinbox_state() # Set initial state

            row += 1

    def get_ptm_configs(self) -> list[Ptm]:
        """
        Returns a list of Ptm objects for the currently enabled and configured PTMs.
        """
        configs = []
        for name, ptm_info in self.ptm_vars.items():
            if ptm_info["enabled"].get():
                configs.append(
                    Ptm(
                        name=ptm_info["template"].name,
                        mass_shift=ptm_info["template"].mass_shift,
                        residue=ptm_info["template"].residue,
                        probability=ptm_info["prob"].get()
                    )
                )
        return configs

    def set_ptm_configs(self, configs: list[Ptm]):
        """
        Sets the state of the editor based on a list of Ptm objects.
        """
        # First, disable all
        for ptm_info in self.ptm_vars.values():
            ptm_info["enabled"].set(False)

        # Then, enable and set probabilities for the provided configs
        for config in configs:
            if config.name in self.ptm_vars:
                self.ptm_vars[config.name]["enabled"].set(True)
                self.ptm_vars[config.name]["prob"].set(config.probability)

        # Update spinbox states
        for child in self.winfo_children():
            if isinstance(child, ttk.Checkbutton):
                child.invoke() # This will trigger the command
                child.invoke() # Twice to restore original state
