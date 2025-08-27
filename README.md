# Simulated Spectrum Generator

This application generates simulated mass spectrometry data (`.mzML` files) for intact proteins and covalent binding screens. It provides a graphical user interface (GUI) to control the simulation parameters, including protein mass, instrument resolution, noise levels, and liquid chromatography (LC) peak shape.

## Features

- **Spectrum Generator:** Simulate spectra for one or more proteins with specified masses and relative intensities.
- **Covalent Binding Simulator:** Simulate a covalent binding screen for a target protein against a list of compounds, with probabilistic binding outcomes.
- **Realistic Data:** Includes isotopic distributions, charge state envelopes, and multiple configurable noise sources.
- **Batch Processing:** File generation is handled using `multiprocessing` to leverage multiple CPU cores and speed up the creation of large datasets.
- **GUI:** An easy-to-use interface built with Tkinter and ttkbootstrap.

## Installation

To run this application, you need Python 3.10 or newer.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, simply execute the `main.py` script from the root of the repository:

```bash
python main.py
```

## Project Structure

The codebase has been refactored from a single script into a modular structure for better maintainability and scalability.

```
.
├── main.py                 # Main entry point to run the application
├── requirements.txt        # Python dependencies
├── .gitignore              # Files and directories to be ignored by Git
├── src/
│   └── spec_generator/
│       ├── __init__.py
│       ├── core/           # Core scientific logic (isotopes, noise, spectrum math, etc.)
│       ├── gui/            # GUI components and the main application class
│       ├── logic/          # High-level application logic and simulation orchestration
│       ├── utils/          # Helper functions (file I/O, mzML creation, UI widgets)
│       └── workers/        # Top-level functions for multiprocessing tasks
└── tests/                  # Unit tests for the core logic
```
