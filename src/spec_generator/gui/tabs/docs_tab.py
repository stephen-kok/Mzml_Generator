from tkinter import Text, END, DISABLED, WORD, FLAT
from .base_tab import BaseTab

class DocsTab(BaseTab):
    def create_widgets(self):
        """Creates the content for the documentation and overview tab."""
        # The BaseTab already provides a self.content_frame which is scrollable
        text_widget = Text(self.content_frame, wrap=WORD, relief=FLAT, background=self.style.colors.bg)
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Define text styles ---
        text_widget.tag_configure("h1", font=("Helvetica", 16, "bold"), spacing3=10)
        text_widget.tag_configure("h2", font=("Helvetica", 12, "bold"), spacing3=8)
        text_widget.tag_configure("bold", font=("Helvetica", 10, "bold"))
        text_widget.tag_configure("body", font=("Helvetica", 10), lmargin1=10, lmargin2=10, spacing1=5)

        # --- Add content ---
        docs_content = [
            ("Spectrum Simulator (v3.6-refactored)\n\n", "h1"),
            ("This tool is designed to generate realistic, simulated mass spectrometry data (.mzML files) for intact proteins and covalent binding screens.\n\n", "body"),
            ("How it Works: The Core Model\n", "h2"),
            ("1. Isotopic Distribution:", "bold"),
            (" The simulation starts by calculating the theoretical isotopic distribution of a given mass using a Poisson distribution based on the 'averagine' model. It correctly uses the user-provided ", "body"),
            ("Average Mass", "bold"),
            (" to derive the corresponding monoisotopic mass.\n", "body"),
            ("2. Charge State Envelope:", "bold"),
            (" It then calculates a realistic charge state envelope. The charge states are centered around the most abundant isotope, ensuring the final deconvoluted mass matches the input average mass.\n", "body"),
            ("3. Peak Generation:", "bold"),
            (" For each isotope in each charge state, a Gaussian peak is generated. The final width of the peak is a combination of the ", "body"),
            ("Instrument Resolution", "bold"),
            (" (which determines the m/z-dependent broadening) and the ", "body"),
            ("Intrinsic Peak Sigma", "bold"),
            (" (which models constant physical effects like Doppler broadening).\n", "body"),
            ("4. Noise Simulation:", "bold"),
            (" Several layers of noise are added for realism, including chemical noise (low m/z 'haystack'), white electronic noise, signal-dependent shot noise, and a low-frequency baseline wobble. The noise generation is highly optimized using Numba.\n\n", "body"),
            ("Tabs Overview\n", "h2"),
            ("Spectrum Generator Tab\n", "bold"),
            ("This tab is for generating spectra for one or more proteins. You can enter masses manually as a comma-separated list or provide a tab-delimited file with 'Protein' and 'Intensity' columns. The 'Save as Template...' button provides an easy way to create a valid file from the manual inputs.\n\n", "body"),
            ("Covalent Binding Tab\n", "bold"),
            ("This tab simulates a covalent binding screen. It takes a single protein average mass and a list of compounds (from a file with 'Name' and 'Delta' columns). For each compound, it probabilistically determines if binding occurs (and to what extent) and generates a corresponding spectrum containing native, singly-adducted (DAR-1), and doubly-adducted (DAR-2) species.\n\n", "body"),
            ("Advanced Parameters\n", "h2"),
            ("Mass Inhomogeneity:", "bold"),
            (" In the 'Spectrum Generator' tab, this parameter models the natural heterogeneity of large molecules. It applies a Gaussian distribution to the input mass, simulating effects like conformational changes that broaden the final deconvoluted peak.\n", "body"),
            ("LC Tailing Factor (tau):", "bold"),
            (" This parameter, found in the 'LC Simulation' frame, controls the shape of the chromatographic peak. A value of 0 results in a perfect Gaussian peak, while larger values create an 'Exponentially Modified Gaussian' (EMG) with a more realistic tail, common in 'bind and elute' experiments.\n", "body"),
            ("1/f (Pink) Noise:", "bold"),
            (" This checkbox in the 'Noise & Output' frame adds an additional layer of 1/f noise, which can more accurately simulate noise from some electronic components.\n\n", "body"),
            ("Performance & Dependencies\n", "h2"),
            ("This application is highly optimized for performance and uses multiple CPU cores for batch processing. To run, it requires several external libraries. You can install them all with pip from the `requirements.txt` file:\n", "body"),
            ("pip install -r requirements.txt", "bold"),
        ]

        for text, tag in docs_content:
            text_widget.insert(END, text, tag)

        text_widget.config(state=DISABLED) # Make text read-only
