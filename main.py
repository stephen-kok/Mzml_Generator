import tkinter as tk
import multiprocessing
from src.spec_generator.gui.main_app import CombinedSpectrumSequenceApp

def main():
    """
    Initializes and runs the main application.
    """
    # freeze_support() is necessary for multiprocessing to work correctly when
    # the application is frozen into an executable (e.g., with PyInstaller).
    multiprocessing.freeze_support()

    root = tk.Tk()
    app = CombinedSpectrumSequenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
