import tkinter as tk
import numpy as np
from ttkbootstrap import widgets as ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotViewerTab(ttk.Frame):
    def __init__(self, master, style, queue, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.style = style
        self.queue = queue

        # Create a figure and a set of subplots
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=self.style.colors.bg)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=3.0)
        self._style_plot()

        # Create a canvas to display the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()

        # Create a toolbar for the plot
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.ax.plot([]) # Start with an empty plot
        self.canvas.draw()


    def _style_plot(self):
        self.ax.set_facecolor(self.style.colors.inputbg)
        self.ax.spines['bottom'].set_color(self.style.colors.fg)
        self.ax.spines['top'].set_color(self.style.colors.fg)
        self.ax.spines['left'].set_color(self.style.colors.fg)
        self.ax.spines['right'].set_color(self.style.colors.fg)
        self.ax.tick_params(axis='x', colors=self.style.colors.fg)
        self.ax.tick_params(axis='y', colors=self.style.colors.fg)
        self.ax.yaxis.label.set_color(self.style.colors.fg)
        self.ax.xaxis.label.set_color(self.style.colors.fg)
        self.ax.title.set_color(self.style.colors.fg)
        self.fig.patch.set_facecolor(self.style.colors.bg)
        if self.toolbar:
            self.toolbar.config(background=self.style.colors.bg)
            for button in self.toolbar.winfo_children():
                button.config(background=self.style.colors.bg, foreground=self.style.colors.fg)


    def plot_data(self, data):
        """
        Clears the current plot and draws a new one based on the received data.
        """
        self.ax.clear()
        self._style_plot()
        mz_array, spectra_data = data
        total_intensity = np.zeros_like(mz_array)
        is_lc = False

        # spectra_data can be a list of chromatograms (from simulation_task)
        # or a single chromatogram (from binding_task)
        if isinstance(spectra_data, list):
            chromatograms = spectra_data
        else:
            chromatograms = [spectra_data] # Wrap single chromatogram in a list

        for chrom in chromatograms:
            if chrom.ndim == 2:
                if chrom.shape[0] > 1:
                    is_lc = True
                total_intensity += chrom.sum(axis=0)
            else:
                total_intensity += chrom

        self.ax.plot(mz_array, total_intensity, color=self.style.colors.primary)

        if is_lc:
            self.ax.set_title("Total Ion Current (Sum of All Scans)")
        else:
            self.ax.set_title("Mass Spectrum")

        self.ax.set_xlabel("m/z")
        self.ax.set_ylabel("Intensity")
        self.canvas.draw()


    def get_log_widgets(self):
        """
        This tab does not have standard log widgets.
        Returns None to comply with the main app's interface.
        """
        return None, None
