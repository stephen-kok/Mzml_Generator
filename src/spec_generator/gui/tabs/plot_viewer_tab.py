import tkinter as tk
import numpy as np
from ttkbootstrap import widgets as ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PlotViewerTab(ttk.Frame):
    def __init__(self, master, style, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.style = style

        # Create a figure and a set of subplots
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=3.0)

        # Create a canvas to display the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # Create a toolbar for the plot
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.update()

        # Crosshair feature
        self.crosshair_v = self.ax.axvline(0, color='grey', lw=0.5, linestyle='--', visible=False)
        self.crosshair_h = self.ax.axhline(0, color='grey', lw=0.5, linestyle='--', visible=False)
        self.crosshair_text = self.ax.text(
            0.02, 0.95, '', transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=self.style.colors.inputbg, alpha=0.5)
        )
        self.crosshair_text.set_visible(False)

        self._style_plot()

        # Layout the widgets
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ax.plot([]) # Start with an empty plot
        self.canvas.draw()

        # Connect motion and leave events for crosshair
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)

        # Peak annotation feature
        self.canvas.mpl_connect('button_press_event', self.on_click_annotate)
        self.annotations = []
        self.mz_array = None
        self.plotted_intensity = None

        # LC Scan Viewer Feature
        self.is_lc_data = False
        self.lc_data = None
        self.num_scans = 0
        self.lc_controls_frame = ttk.Frame(self)

        # Widgets for the LC controls frame
        self.scan_slider_var = tk.IntVar()
        self.scan_slider = ttk.Scale(self.lc_controls_frame, from_=0, to=100, orient='horizontal', variable=self.scan_slider_var, command=self._on_slider_change)
        self.scan_label_var = tk.StringVar(value="Scan: -/-")
        self.scan_label = ttk.Label(self.lc_controls_frame, textvariable=self.scan_label_var)

        self.scan_entry_var = tk.StringVar()
        self.scan_entry = ttk.Entry(self.lc_controls_frame, textvariable=self.scan_entry_var, width=5)
        self.scan_entry.bind('<Return>', self._on_scan_entry)

        self.range_start_var = tk.StringVar()
        self.range_end_var = tk.StringVar()
        self.range_start_entry = ttk.Entry(self.lc_controls_frame, textvariable=self.range_start_var, width=5)
        self.range_end_entry = ttk.Entry(self.lc_controls_frame, textvariable=self.range_end_var, width=5)
        self.sum_range_button = ttk.Button(self.lc_controls_frame, text="Sum Range", command=self._on_sum_range)

        # Layout for the LC controls
        self.scan_label.pack(side=tk.LEFT, padx=5)
        self.scan_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(self.lc_controls_frame, text="Go to:").pack(side=tk.LEFT, padx=(10, 2))
        self.scan_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.lc_controls_frame, text="Range:").pack(side=tk.LEFT, padx=(10, 2))
        self.range_start_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(self.lc_controls_frame, text="-").pack(side=tk.LEFT, padx=2)
        self.range_end_entry.pack(side=tk.LEFT, padx=2)
        self.sum_range_button.pack(side=tk.LEFT, padx=5)


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
            for widget in self.toolbar.winfo_children():
                if isinstance(widget, (tk.Button, tk.Checkbutton, tk.Label)):
                    widget.config(background=self.style.colors.bg, foreground=self.style.colors.fg)

        if hasattr(self, 'crosshair_text'):
            self.crosshair_text.get_bbox_patch().set_facecolor(self.style.colors.inputbg)
            self.crosshair_text.set_color(self.style.colors.fg)


    def plot_data(self, data):
        """
        Clears the current plot and draws a new one based on the received data.
        Orchestrates the UI for LC vs. non-LC data.
        """
        self.ax.clear()
        self.annotations = []
        self._style_plot()

        if not data:
            self.ax.plot([])
            self.ax.set_title("No data to display")
            self.canvas.draw()
            return

        self.mz_array, list_of_chromatograms = data

        if not list_of_chromatograms:
            self.ax.plot([])
            self.ax.set_title("No data to display")
            self.canvas.draw()
            return

        # The viewer plots one chromatogram at a time. Get the first one.
        # It's a list of 1D arrays (scans).
        chromatogram_scans = list_of_chromatograms[0]

        # Check if it's LC data (more than one scan)
        if isinstance(chromatogram_scans, list) and len(chromatogram_scans) > 1:
            self.is_lc_data = True
            # Convert list of scans to a 2D numpy array for easier handling
            self.lc_data = np.array(chromatogram_scans)
            self.num_scans = self.lc_data.shape[0]

            self._setup_lc_controls()
            self.lc_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(2,5), before=self.canvas.get_tk_widget())

            # Initially plot the Total Ion Current (TIC)
            total_intensity = self.lc_data.sum(axis=0)

            self.plotted_intensity = total_intensity
            self.ax.plot(self.mz_array, self.plotted_intensity, color=self.style.colors.primary)
            self.ax.set_title("Total Ion Current (Sum of All Scans)")
        else: # Not LC data, or only one scan
            self.is_lc_data = False
            self.lc_controls_frame.pack_forget()
            self.lc_data = None

            # If it's a list, take the first element, otherwise it's the spectrum itself.
            single_spectrum = chromatogram_scans[0] if isinstance(chromatogram_scans, list) else chromatogram_scans

            self.plotted_intensity = single_spectrum
            self.ax.plot(self.mz_array, self.plotted_intensity, color=self.style.colors.primary)
            self.ax.set_title("Mass Spectrum")

        self.ax.set_xlabel("m/z")
        self.ax.set_ylabel("Intensity")
        self.canvas.draw()


    def on_mouse_move(self, event):
        """Shows and moves the crosshair with the mouse."""
        if not event.inaxes:
            # If the mouse leaves the axes, hide the crosshair
            if self.crosshair_v.get_visible():
                self.on_mouse_leave(event)
            return

        # Make crosshair visible if it's not
        if not self.crosshair_v.get_visible():
            self.crosshair_v.set_visible(True)
            self.crosshair_h.set_visible(True)
            self.crosshair_text.set_visible(True)

        x, y = event.xdata, event.ydata
        # For axvline/axhline, set_xdata/set_ydata expects a sequence.
        self.crosshair_v.set_xdata([x, x])
        self.crosshair_h.set_ydata([y, y])
        self.crosshair_text.set_text(f'm/z: {x:,.2f}\nIntensity: {y:,.2f}')
        self.canvas.draw_idle()

    def on_mouse_leave(self, event):
        """Hides the crosshair when the mouse leaves the axes."""
        if self.crosshair_v.get_visible():
            self.crosshair_v.set_visible(False)
            self.crosshair_h.set_visible(False)
            self.crosshair_text.set_visible(False)
            self.canvas.draw_idle()


    def on_click_annotate(self, event):
        """Annotates a peak on the plot when clicked."""
        # Ignore clicks outside the plot area or if no data is plotted
        if not event.inaxes or self.mz_array is None or self.plotted_intensity is None:
            return

        # Ignore right-clicks, middle-clicks, etc.
        if event.button != 1:
            return

        # Find the index of the closest data point to the click
        x_click = event.xdata
        idx = (np.abs(self.mz_array - x_click)).argmin()

        x_data = self.mz_array[idx]
        y_data = self.plotted_intensity[idx]

        # Create and style the annotation
        annotation_text = f"{x_data:.2f}, {y_data:,.0f}"
        annotation = self.ax.annotate(
            annotation_text,
            xy=(x_data, y_data),
            xytext=(0, 25),
            textcoords="offset points",
            ha='center',
            va='bottom',
            bbox=dict(boxstyle="round,pad=0.4", fc=self.style.colors.warning, ec=self.style.colors.fg, lw=0.5, alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", facecolor=self.style.colors.fg, edgecolor=self.style.colors.fg)
        )
        annotation.draggable()

        self.annotations.append(annotation)
        self.canvas.draw_idle()


    def _setup_lc_controls(self):
        """Configures the LC control widgets based on the loaded data."""
        self.scan_slider.config(from_=0, to=self.num_scans - 1)
        self.scan_slider_var.set(0)
        self.scan_label_var.set(f"Scan: 1 / {self.num_scans}")
        self.scan_entry_var.set("1")
        self.range_start_var.set("1")
        self.range_end_var.set(str(self.num_scans))

    def _on_slider_change(self, value):
        """Callback for when the slider is moved."""
        scan_idx = int(float(value))
        self._plot_lc_data(start_scan=scan_idx, end_scan=scan_idx)

    def _on_scan_entry(self, event=None):
        """Callback for when a scan is entered in the entry box."""
        try:
            scan_num = int(self.scan_entry_var.get())
            if 1 <= scan_num <= self.num_scans:
                scan_idx = scan_num - 1
                self.scan_slider_var.set(scan_idx)
                self._plot_lc_data(start_scan=scan_idx, end_scan=scan_idx)
        except ValueError:
            pass # Ignore non-integer input

    def _on_sum_range(self):
        """Callback for the 'Sum Range' button."""
        try:
            start_num = int(self.range_start_var.get())
            end_num = int(self.range_end_var.get())
            if 1 <= start_num <= end_num <= self.num_scans:
                self._plot_lc_data(start_scan=start_num - 1, end_scan=end_num - 1)
        except ValueError:
            pass # Ignore non-integer input

    def _plot_lc_data(self, start_scan, end_scan):
        """Updates the plot to show a single scan or a sum of a range."""
        if not self.is_lc_data:
            return

        for ann in self.annotations:
            ann.remove()
        self.annotations = []

        intensity_slice = np.zeros_like(self.mz_array)
        for chrom in self.lc_data:
            if chrom.ndim == 2:
                intensity_slice += chrom[start_scan : end_scan + 1, :].sum(axis=0)

        self.plotted_intensity = intensity_slice
        line = self.ax.lines[0]
        line.set_ydata(self.plotted_intensity)

        if start_scan == end_scan:
            title = f"Scan: {start_scan + 1}"
            self.scan_label_var.set(f"Scan: {start_scan + 1} / {self.num_scans}")
            self.scan_slider_var.set(start_scan)
            self.scan_entry_var.set(str(start_scan + 1))
        else:
            title = f"Sum of Scans: {start_scan + 1} - {end_scan + 1}"
        self.ax.set_title(title)

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
