import os
import threading
from datetime import datetime
from tkinter import (HORIZONTAL, LEFT, NORMAL, DISABLED, NSEW, SUNKEN, WORD,
                     StringVar, messagebox, Text, E, W, END)

from ttkbootstrap import widgets as ttk
from ttkbootstrap.constants import PRIMARY

from ...utils.ui_helpers import Tooltip, show_plot
from ...utils.file_io import format_filename
from ...logic.peptide_map import execute_peptide_map_simulation
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame
from ...config import PeptideMapSimConfig, PeptideMapLCParams

class PeptideMapTab(BaseTab):
    def create_widgets(self):
        # --- Input Frame ---
        input_frame = ttk.LabelFrame(self.content_frame, text="1. Input Sequence & Digestion", padding=(15, 10))
        input_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        input_frame.columnconfigure(1, weight=1)

        # Sequence Input
        seq_label = ttk.Label(input_frame, text="Protein Sequence:")
        seq_label.grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.sequence_text = Text(input_frame, height=10, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.sequence_text.grid(row=1, column=0, columnspan=4, sticky=NSEW, padx=5, pady=5)
        self.sequence_text.insert(END, "TESTPEPTIDEKTESTPEPTIDER") # Example sequence
        input_frame.rowconfigure(1, weight=1)

        # Digestion Params
        digestion_frame = ttk.Frame(input_frame)
        digestion_frame.grid(row=2, column=0, columnspan=4, sticky=W, pady=5)

        cleavages_label = ttk.Label(digestion_frame, text="Missed Cleavages:")
        cleavages_label.pack(side=LEFT, padx=(5,2))
        self.missed_cleavages_var = StringVar(value="2")
        cleavages_entry = ttk.Entry(digestion_frame, textvariable=self.missed_cleavages_var, width=5)
        cleavages_entry.pack(side=LEFT, padx=(0,10))
        Tooltip(cleavages_entry, "Maximum number of missed trypsin cleavage sites (0-2 recommended).")

        charge_label = ttk.Label(digestion_frame, text="Charge State:")
        charge_label.pack(side=LEFT, padx=(5,2))
        self.charge_state_var = StringVar(value="2")
        charge_entry = ttk.Entry(digestion_frame, textvariable=self.charge_state_var, width=5)
        charge_entry.pack(side=LEFT, padx=(0,10))
        Tooltip(charge_entry, "The charge state to simulate for all peptides.")

        # --- Parameters & Generation ---
        gen_frame = ttk.LabelFrame(self.content_frame, text="2. Simulation Parameters & Output", padding=(15, 10))
        gen_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        common_frame = ttk.Frame(gen_frame)
        common_frame.pack(fill="x", expand=True)
        self.peptide_map_params = create_common_parameters_frame(common_frame, "200.0", "2000.0", "Low Noise")
        self.peptide_map_params['output_directory_var'].set(os.path.join(os.getcwd(), "Peptide Map Mock Spectra"))
        self.peptide_map_params['filename_template_var'].set("{date}_{time}_pepmap_sim_{run_time}min_{noise}.mzML")

        # LC Parameters Frame
        lc_container = ttk.LabelFrame(gen_frame, text="LC Parameters", padding=(10, 5))
        lc_container.pack(fill="x", expand=True, pady=10)

        self.lc_run_time_var = StringVar(value="60.0")
        self.lc_scan_interval_var = StringVar(value="1.0")
        self.lc_peak_width_var = StringVar(value="30.0")

        ttk.Label(lc_container, text="Run Time (min):").grid(row=0, column=0, padx=5, pady=2, sticky=W)
        ttk.Entry(lc_container, textvariable=self.lc_run_time_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky=W)
        ttk.Label(lc_container, text="Scan Interval (s):").grid(row=0, column=2, padx=5, pady=2, sticky=W)
        ttk.Entry(lc_container, textvariable=self.lc_scan_interval_var, width=10).grid(row=0, column=3, padx=5, pady=2, sticky=W)
        ttk.Label(lc_container, text="Peak Width (s):").grid(row=0, column=4, padx=5, pady=2, sticky=W)
        ttk.Entry(lc_container, textvariable=self.lc_peak_width_var, width=10).grid(row=0, column=5, padx=5, pady=2, sticky=W)

        action_frame = ttk.Frame(gen_frame)
        action_frame.pack(pady=(10,0))

        self.preview_button = ttk.Button(action_frame, text="Preview BPC", command=self._preview_command, style='Outline.TButton')
        self.preview_button.pack(side=LEFT, padx=5)
        Tooltip(self.preview_button, "Generate and display a plot of the Base Peak Chromatogram.")

        self.generate_button = ttk.Button(action_frame, text="Generate mzML File", command=self._generate_command, bootstyle=PRIMARY)
        self.generate_button.pack(side=LEFT, padx=5)

        # --- Progress & Log ---
        progress_frame = ttk.LabelFrame(self.content_frame, text="3. Progress & Log", padding=(15, 10))
        progress_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=0, column=0, pady=5, sticky="ew")

        self.output_text = Text(progress_frame, height=8, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.output_text.grid(row=1, column=0, sticky=NSEW)
        log_scroll = ttk.Scrollbar(progress_frame, command=self.output_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.output_text['yscrollcommand'] = log_scroll.set

    def _gather_config(self) -> PeptideMapSimConfig:
        common, _ = self._gather_common_params(self.peptide_map_params)

        lc_params = PeptideMapLCParams(
            run_time=float(self.lc_run_time_var.get()),
            scan_interval=float(self.lc_scan_interval_var.get()),
            peak_width_seconds=float(self.lc_peak_width_var.get())
        )

        sequence = self.sequence_text.get("1.0", "end-1c").strip().upper()
        if not sequence:
            raise ValueError("Protein sequence cannot be empty.")

        return PeptideMapSimConfig(
            common=common,
            lc=lc_params,
            sequence=sequence,
            missed_cleavages=int(self.missed_cleavages_var.get()),
            charge_state=int(self.charge_state_var.get())
        )

    def _generate_command(self):
        self.generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.task_queue.put(('clear_log', None))
        threading.Thread(target=self._worker_generate, daemon=True).start()

    def _worker_generate(self):
        try:
            config = self._gather_config()
            placeholders = {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "time": datetime.now().strftime('%H%M%S'),
                "run_time": config.lc.run_time,
                "noise": config.common.noise_option.replace(" ", ""),
                "seed": config.common.seed,
            }
            filename = format_filename(config.common.filename_template, placeholders)
            filepath = os.path.join(config.common.output_directory, filename)

            execute_peptide_map_simulation(
                config=config,
                final_filepath=filepath,
                update_queue=self.task_queue
            )
        except (ValueError, Exception) as e:
            self.task_queue.put(('error', f"Simulation failed: {e}"))
        finally:
            self.task_queue.put(('done', None))

    def _preview_command(self):
        self.preview_button.config(state=DISABLED)
        threading.Thread(target=self._worker_preview, daemon=True).start()

    def _worker_preview(self):
        try:
            config = self._gather_config()
            result = execute_peptide_map_simulation(
                config=config,
                final_filepath="",
                update_queue=self.task_queue,
                return_data_only=True
            )
            if result and isinstance(result, tuple):
                mz_range, run_data = result
                scans = run_data[0] # The writer wraps it in a list
                bpc = [max(scan) if scan.any() else 0 for scan in scans]
                times = [i * config.lc.scan_interval / 60.0 for i in range(len(scans))]
                show_plot(times, {"Base Peak Chromatogram": bpc}, "BPC Preview", xlabel="Time (min)", ylabel="Intensity")
            else:
                self.task_queue.put(('error', "Could not generate preview data."))
        except (ValueError, Exception) as e:
            self.task_queue.put(('error', f"Preview failed: {e}"))
        finally:
            self.task_queue.put(('preview_done', None))

    def on_task_done(self):
        self.generate_button.config(state=NORMAL)

    def on_preview_done(self):
        self.preview_button.config(state=NORMAL)
