import os
import threading
from dataclasses import asdict
from datetime import datetime
from tkinter import (HORIZONTAL, LEFT, NORMAL, DISABLED, NSEW, SUNKEN, WORD,
                     StringVar, messagebox, Text, E)

from ttkbootstrap import widgets as ttk
from ttkbootstrap.constants import PRIMARY

import numpy as np

from ...utils.ui_helpers import Tooltip, parse_float_entry, show_plot
from ...utils.file_io import format_filename
from ...logic.antibody import (generate_assembly_combinations,
                               calculate_assembly_masses,
                               execute_antibody_simulation)
from .base_tab import BaseTab
from ..shared_widgets import create_common_parameters_frame, create_lc_simulation_frame
from ...config import AntibodySimConfig, Chain

class AntibodyTab(BaseTab):
    def create_widgets(self):
        self.chain_entries = []
        self.assembly_abundances = {}

        # --- Chain Input Frame ---
        chain_frame = ttk.LabelFrame(self.content_frame, text="1. Chain Definition", padding=(15, 10))
        chain_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        chain_frame.columnconfigure(1, weight=1)

        self.chain_inner_frame = ttk.Frame(chain_frame)
        self.chain_inner_frame.grid(row=0, column=0, columnspan=4, sticky="ew")

        button_frame = ttk.Frame(chain_frame)
        button_frame.grid(row=1, column=0, columnspan=4, pady=(5,0))
        add_hc_button = ttk.Button(button_frame, text="Add Heavy Chain", command=lambda: self.add_chain_row("HC"), style='Outline.TButton')
        add_hc_button.pack(side=LEFT, padx=5)
        add_lc_button = ttk.Button(button_frame, text="Add Light Chain", command=lambda: self.add_chain_row("LC"), style='Outline.TButton')
        add_lc_button.pack(side=LEFT, padx=5)

        self.add_chain_row("HC", "PEPTIDE")
        self.add_chain_row("LC", "SEQUENCE")

        # --- Assemblies Frame ---
        assemblies_frame = ttk.LabelFrame(self.content_frame, text="2. Generated Assemblies & Abundance", padding=(15, 10))
        assemblies_frame.grid(row=1, column=0, sticky=NSEW, padx=10, pady=10)
        assemblies_frame.columnconfigure(0, weight=1)
        assemblies_frame.rowconfigure(0, weight=1)
        self.content_frame.rowconfigure(1, weight=1)

        cols = ("Assembly", "Mass (Da)", "Bonds", "Abundance")
        self.assemblies_tree = ttk.Treeview(assemblies_frame, columns=cols, show='headings', height=8)
        for col in cols:
            self.assemblies_tree.heading(col, text=col)
        self.assemblies_tree.column("Assembly", width=150)
        self.assemblies_tree.column("Mass (Da)", width=100, anchor=E)
        self.assemblies_tree.column("Bonds", width=50, anchor=E)
        self.assemblies_tree.column("Abundance", width=100, anchor=E)
        self.assemblies_tree.grid(row=0, column=0, sticky=NSEW)

        tree_scroll = ttk.Scrollbar(assemblies_frame, orient="vertical", command=self.assemblies_tree.yview)
        self.assemblies_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.assemblies_tree.bind("<Double-1>", self._on_treeview_double_click)

        preview_button = ttk.Button(assemblies_frame, text="Generate / Refresh Assemblies", command=self._preview_assemblies_command, style='Outline.TButton')
        preview_button.grid(row=1, column=0, columnspan=2, pady=(10,0))
        Tooltip(preview_button, "Generate the list of possible antibody assemblies and their masses based on the chains defined above.")

        # --- Parameters & Generation ---
        gen_frame = ttk.LabelFrame(self.content_frame, text="3. Simulation Parameters & Output", padding=(15, 10))
        gen_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        common_frame = ttk.Frame(gen_frame)
        common_frame.pack(fill="x", expand=True)
        self.antibody_params = create_common_parameters_frame(common_frame, "400.0", "4000.0", "Default Noise")
        self.antibody_params['output_directory_var'].set(os.path.join(os.getcwd(), "Antibody Mock Spectra"))
        self.antibody_params['filename_template_var'].set("{date}_{time}_antibody_sim_{scans}scans_{noise}.mzML")

        lc_container, self.antibody_lc_params = create_lc_simulation_frame(gen_frame, True)
        lc_container.pack(fill="x", expand=True, pady=10)

        action_frame = ttk.Frame(gen_frame)
        action_frame.pack(pady=(10,0))

        self.antibody_generate_button = ttk.Button(action_frame, text="Generate mzML File", command=self.generate_antibody_spectra_command, bootstyle=PRIMARY)
        self.antibody_generate_button.pack(side=LEFT, padx=5)

        # --- Progress & Log ---
        progress_frame = ttk.LabelFrame(self.content_frame, text="4. Progress & Log", padding=(15, 10))
        progress_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=0, column=0, pady=5, sticky="ew")

        self.output_text = Text(progress_frame, height=8, wrap=WORD, relief=SUNKEN, borderwidth=1)
        self.output_text.grid(row=1, column=0, sticky=NSEW)
        log_scroll = ttk.Scrollbar(progress_frame, command=self.output_text.yview)
        log_scroll.grid(row=1, column=1, sticky="ns")
        self.output_text['yscrollcommand'] = log_scroll.set

    def _gather_config(self) -> AntibodySimConfig:
        common, lc = self._gather_common_params(self.antibody_params, self.antibody_lc_params)

        chains = []
        for entry in self.chain_entries:
            seq = entry['seq_var'].get().strip().upper()
            name = entry['name_var'].get().strip()
            if not seq or not name:
                raise ValueError("All chain sequences and names must be provided.")
            chains.append(Chain(type=entry['type'], name=name, seq=seq))

        if not self.assembly_abundances:
            raise ValueError("Please generate the assemblies first.")

        ordered_names = [self.assemblies_tree.item(item_id)['values'][0] for item_id in self.assemblies_tree.get_children()]
        abundances = {name: parse_float_entry(self.assembly_abundances[name].get(), f"Abundance for {name}") for name in ordered_names}

        return AntibodySimConfig(
            common=common,
            lc=lc,
            chains=chains,
            assembly_abundances=abundances
        )

    def add_chain_row(self, chain_type, sequence=""):
        entry_data = {}

        def remove_chain_row():
            entry_to_remove = next((item for item in self.chain_entries if item['id'] == entry_data['id']), None)
            if not entry_to_remove: return
            for widget in entry_to_remove['widgets']: widget.destroy()
            self.chain_entries.remove(entry_to_remove)
            for i, entry in enumerate(self.chain_entries):
                for col, widget in enumerate(entry['widgets']):
                    widget.grid(row=i, column=col, sticky="w" if col !=1 else "ew", pady=2, padx=5)

        row = len(self.chain_entries)

        chain_label = ttk.Label(self.chain_inner_frame, text=f"{chain_type} Sequence:")
        chain_label.grid(row=row, column=0, sticky="w", pady=2, padx=5)

        seq_var = StringVar(value=sequence)
        chain_seq_entry = ttk.Entry(self.chain_inner_frame, textvariable=seq_var, width=40)
        chain_seq_entry.grid(row=row, column=1, sticky="ew", pady=2, padx=5)

        name_var = StringVar(value=f"{chain_type}{row+1}")
        chain_name_entry = ttk.Entry(self.chain_inner_frame, textvariable=name_var, width=10)
        chain_name_entry.grid(row=row, column=2, sticky="w", pady=2, padx=5)

        remove_button = ttk.Button(self.chain_inner_frame, text="X", command=remove_chain_row, width=2, bootstyle="danger-outline")
        remove_button.grid(row=row, column=3, sticky="w", pady=2, padx=5)

        entry_data = {
            'id': id(seq_var), 'type': chain_type, 'seq_var': seq_var, 'name_var': name_var,
            'widgets': [chain_label, chain_seq_entry, chain_name_entry, remove_button]
        }
        self.chain_entries.append(entry_data)
        self.chain_inner_frame.columnconfigure(1, weight=1)

    def _preview_assemblies_command(self):
        self.output_text.delete('1.0', "end")
        self.app_queue.put(('log', "Generating assembly preview...\n"))

        try:
            for item in self.assemblies_tree.get_children():
                self.assemblies_tree.delete(item)
            self.assembly_abundances.clear()

            chains = []
            for entry in self.chain_entries:
                seq = entry['seq_var'].get().strip().upper()
                name = entry['name_var'].get().strip()
                if not seq or not name:
                    raise ValueError("All chain sequences and names must be provided.")
                chains.append(Chain(type=entry['type'], name=name, seq=seq))

            if not chains:
                raise ValueError("No chains defined.")

            # Convert chain objects to dicts for the logic functions
            chains_as_dicts = [asdict(c) for c in chains]

            assemblies = generate_assembly_combinations(chains_as_dicts)
            assemblies_with_mass = calculate_assembly_masses(chains_as_dicts, assemblies)
            # Sort by mass in ascending order
            assemblies_with_mass.sort(key=lambda x: x['mass'])

            for assembly in assemblies_with_mass:
                name = assembly['name']
                mass_str = f"{assembly['mass']:.2f}"
                bonds_str = str(assembly['bonds'])
                abundance_var = StringVar(value="1.0")
                self.assembly_abundances[name] = abundance_var
                self.assemblies_tree.insert("", "end", values=(name, mass_str, bonds_str, abundance_var.get()))

            self.app_queue.put(('log', f"Successfully generated {len(assemblies_with_mass)} species. You can now edit their relative abundances.\n"))

        except (ValueError, Exception) as e:
            self.app_queue.put(('error', str(e)))

    def generate_antibody_spectra_command(self):
        self.antibody_generate_button.config(state=DISABLED)
        self.progress_bar["value"] = 0
        self.app_queue.put(('clear_log', None))
        threading.Thread(target=self._worker_generate_antibody_spectra, daemon=True).start()

    def _on_treeview_double_click(self, event):
        region = self.assemblies_tree.identify_region(event.x, event.y)
        if region != "cell": return
        column_id = self.assemblies_tree.identify_column(event.x)
        if column_id != "#4": return
        item_id = self.assemblies_tree.identify_row(event.y)
        assembly_name = self.assemblies_tree.item(item_id, "values")[0]
        x, y, width, height = self.assemblies_tree.bbox(item_id, column_id)
        entry_var = self.assembly_abundances[assembly_name]
        entry = ttk.Entry(self.assemblies_tree, textvariable=entry_var, justify='right')
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(event=None):
            new_value = entry_var.get()
            self.assemblies_tree.set(item_id, column_id, new_value)
            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def _worker_generate_antibody_spectra(self):
        try:
            config = self._gather_config()

            placeholders = {
                "date": datetime.now().strftime('%Y-%m-%d'),
                "time": datetime.now().strftime('%H%M%S'),
                "scans": config.lc.num_scans,
                "noise": config.common.noise_option.replace(" ", ""),
                "seed": config.common.seed,
            }
            filename = format_filename(config.common.filename_template, placeholders)
            filepath = os.path.join(config.common.output_directory, filename)

            success = execute_antibody_simulation(
                config=config,
                final_filepath=filepath,
                update_queue=self.app_queue
            )

            if success:
                self.app_queue.put(('done', "Antibody mzML file successfully created."))
            else:
                self.app_queue.put(('done', None))

        except (ValueError, Exception) as e:
            self.app_queue.put(('error', f"Simulation failed: {e}"))
            self.app_queue.put(('done', None))

    def on_task_done(self):
        self.antibody_generate_button.config(state=NORMAL)
