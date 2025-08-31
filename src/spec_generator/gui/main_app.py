import tkinter as tk
from tkinter import messagebox
from ttkbootstrap import Style, widgets as ttk
from ttkbootstrap.constants import BOTH

from .tabs.docs_tab import DocsTab
from .tabs.plot_viewer_tab import PlotViewerTab
from .tabs.spectrum_tab import SpectrumTab
from .tabs.binding_tab import BindingTab
from .tabs.antibody_tab import AntibodyTab
from .tabs.peptide_map_tab import PeptideMapTab
from .tabs.base_tab import BaseTab

class CombinedSpectrumSequenceApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Simulated Spectrum Generator (v4.0-refactored)")
        try:
            self.style = Style(theme="solar")
        except tk.TclError:
            self.style = Style(theme="litera")

        self.tabs = {}  # To hold references to tab instances by name

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Create and add tabs in a specific order
        self.add_tab(DocsTab, "Overview & Docs")
        self.add_tab(PlotViewerTab, "Plot Viewer")
        self.add_tab(SpectrumTab, "Spectrum Generator")
        self.add_tab(BindingTab, "Covalent Binding")
        self.add_tab(AntibodyTab, "Antibody Simulation")
        self.add_tab(PeptideMapTab, "Peptide Map")

        master.minsize(650, 600)

    def add_tab(self, tab_class, text):
        # Pass a reference to this main app instance to all BaseTabs
        if issubclass(tab_class, BaseTab):
            tab = tab_class(self.notebook, self.style, app_controller=self)
        else:
            tab = tab_class(self.notebook, self.style)

        self.notebook.add(tab, text=text)
        self.tabs[text] = tab

    def get_plot_viewer(self):
        """Returns the instance of the PlotViewerTab."""
        return self.tabs.get("Plot Viewer")

    def switch_to_plot_viewer(self):
        """Switches the active notebook tab to the Plot Viewer."""
        plot_tab = self.get_plot_viewer()
        if plot_tab:
            self.notebook.select(plot_tab)
