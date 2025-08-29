import tkinter as tk
from tkinter import ttk


class Tooltip:
    """
    Creates a tooltip for a given widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return

        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip_window, text=self.text, background="#FFFFE0",
            foreground="black", relief="solid", borderwidth=1,
            padding=5, justify=tk.LEFT
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None


class ScrollableFrame(ttk.Frame):
    """
    A scrollable frame widget that can contain other widgets.
    """
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)

        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel scrolling for different platforms
        canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows/macOS
        canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux scroll up
        canvas.bind_all("<Button-5>", self._on_mousewheel)    # Linux scroll down

    def _on_mousewheel(self, event):
        # Determine scroll direction and magnitude
        if event.num == 5:
            delta = 1
        elif event.num == 4:
            delta = -1
        else: # Windows/macOS
            delta = -1 if event.delta > 0 else 1

        # The first child of the main frame is the canvas
        self.winfo_children()[0].yview_scroll(delta, "units")


def parse_float_entry(entry_value: str, name: str) -> float:
    """ Safely parses a string to a float, raising a ValueError with a descriptive message on failure. """
    try:
        return float(entry_value)
    except ValueError:
        raise ValueError(f"Invalid value for {name}. Please enter a number.")


def parse_range_entry(entry_value: str, name: str) -> tuple[float, float]:
    """ Parses a string in 'start-end' format into a tuple of two floats. """
    try:
        parts = entry_value.split('-')
        if len(parts) != 2:
            raise ValueError("Incorrect format.")

        start = float(parts[0])
        end = float(parts[1])

        if start > end:
            raise ValueError("Start of range cannot be greater than the end.")

        return start, end
    except (ValueError, IndexError):
        raise ValueError(f"Invalid range for {name}. Use 'start-end' format (e.g., '10-50').")


def show_plot(mz_range, intensity_data: dict, title: str, xlabel="m/z", ylabel="Intensity"):
    """
    Displays a matplotlib plot of the spectrum.
    """
    import matplotlib.pyplot as plt
    from tkinter import messagebox

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
