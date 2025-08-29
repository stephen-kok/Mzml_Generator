import queue
import tkinter as tk
from tkinter import messagebox
from ttkbootstrap import Style, widgets as ttk
from ttkbootstrap.constants import BOTH

from .tabs.docs_tab import DocsTab
from .tabs.spectrum_tab import SpectrumTab
from .tabs.binding_tab import BindingTab
from .tabs.antibody_tab import AntibodyTab

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

    def process_queue(self):
        try:
            while True:
                msg_type, msg_data = self.queue.get_nowait()
                active_tab = self.get_active_tab()

                if not active_tab:
                    continue

                output_text, progress_bar = active_tab.get_log_widgets()

                if msg_type == 'log':
                    if output_text:
                        output_text.insert("end", msg_data)
                        output_text.see("end")
                elif msg_type == 'clear_log':
                    if output_text:
                        output_text.delete('1.0', "end")
                elif msg_type == 'progress_set':
                    if progress_bar:
                        progress_bar["value"] = msg_data
                elif msg_type == 'progress_add':
                    if progress_bar:
                        progress_bar["value"] += msg_data
                elif msg_type == 'error':
                    messagebox.showerror("Error", msg_data)
                    if progress_bar:
                        progress_bar["value"] = 0
                elif msg_type == 'warning':
                    messagebox.showwarning("Warning", msg_data)
                elif msg_type == 'done':
                    if active_tab:
                        active_tab.on_task_done()
                    if msg_data:
                        messagebox.showinfo("Complete", msg_data)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.process_queue)
