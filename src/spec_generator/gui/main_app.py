import queue
import tkinter as tk
from tkinter import messagebox
from ttkbootstrap import Style, widgets as ttk
from ttkbootstrap.constants import BOTH

from .tabs.docs_tab import DocsTab
from .tabs.spectrum_tab import SpectrumTab
from .tabs.binding_tab import BindingTab
from .tabs.antibody_tab import AntibodyTab
from .tabs.peptide_map_tab import PeptideMapTab

class CombinedSpectrumSequenceApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Simulated Spectrum Generator (v4.0-refactored)")
        try:
            self.style = Style(theme="solar")
        except tk.TclError:
            self.style = Style(theme="litera")

        self.queue = queue.Queue()
        self.tabs = {}  # To hold references to tab instances

        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Create and add tabs
        self.add_tab(DocsTab, "Overview & Docs")
        self.add_tab(SpectrumTab, "Spectrum Generator")
        self.add_tab(BindingTab, "Covalent Binding")
        self.add_tab(AntibodyTab, "Antibody Simulation")
        self.add_tab(PeptideMapTab, "Peptide Map")

        self._setup_queue_handlers()
        self.process_queue()
        master.minsize(650, 600)

    def add_tab(self, tab_class, text):
        tab = tab_class(self.notebook, self.style, self.queue)
        self.notebook.add(tab, text=text)
        self.tabs[text] = tab

    def get_active_tab(self):
        try:
            selected_tab_name = self.notebook.tab(self.notebook.select(), "text")
            return self.tabs.get(selected_tab_name)
        except tk.TclError:
            return None # No tab selected

    def _setup_queue_handlers(self):
        self.queue_handlers = {
            'log': self._handle_log,
            'clear_log': self._handle_clear_log,
            'progress_set': self._handle_progress_set,
            'progress_add': self._handle_progress_add,
            'progress_max': self._handle_progress_max,
            'error': self._handle_error,
            'warning': self._handle_warning,
            'done': self._handle_done,
            'preview_done': self._handle_preview_done,
        }

    def _handle_log(self, msg_data, active_tab):
        output_text, _ = active_tab.get_log_widgets()
        if output_text:
            output_text.insert("end", msg_data)
            output_text.see("end")

    def _handle_clear_log(self, msg_data, active_tab):
        output_text, _ = active_tab.get_log_widgets()
        if output_text:
            output_text.delete('1.0', "end")

    def _handle_progress_set(self, msg_data, active_tab):
        _, progress_bar = active_tab.get_log_widgets()
        if progress_bar:
            progress_bar["value"] = msg_data

    def _handle_progress_add(self, msg_data, active_tab):
        _, progress_bar = active_tab.get_log_widgets()
        if progress_bar:
            progress_bar["value"] += msg_data

    def _handle_progress_max(self, msg_data, active_tab):
        _, progress_bar = active_tab.get_log_widgets()
        if progress_bar:
            progress_bar["maximum"] = msg_data

    def _handle_error(self, msg_data, active_tab):
        _, progress_bar = active_tab.get_log_widgets()
        messagebox.showerror("Error", msg_data)
        if progress_bar:
            progress_bar["value"] = 0

    def _handle_warning(self, msg_data, active_tab):
        messagebox.showwarning("Warning", msg_data)

    def _handle_done(self, msg_data, active_tab):
        if active_tab:
            active_tab.on_task_done()
        if msg_data:
            messagebox.showinfo("Complete", msg_data)

    def _handle_preview_done(self, msg_data, active_tab):
        if active_tab and hasattr(active_tab, 'on_preview_done'):
            active_tab.on_preview_done()

    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                active_tab = self.get_active_tab()

                if not active_tab:
                    continue

                handler = self.queue_handlers.get(msg_type)
                if handler:
                    handler(msg_data, active_tab)

        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)
