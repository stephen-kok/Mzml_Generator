# Simulated Spectrum Generator

This application generates simulated mass spectrometry data (`.mzML` files) for intact proteins and covalent binding screens. It provides a graphical user interface (GUI) to control the simulation parameters, including protein mass, instrument resolution, noise levels, and liquid chromatography (LC) peak shape.

## Features

- **Spectrum Generator:** Simulate spectra for one or more proteins with specified masses and relative intensities.
- **Covalent Binding Simulator:** Simulate a covalent binding screen for a target protein against a list of compounds, with probabilistic binding outcomes.
- **Realistic Data:** Includes isotopic distributions, charge state envelopes, and multiple configurable noise sources.
- **Batch Processing:** File generation is handled using `multiprocessing` to leverage multiple CPU cores and speed up the creation of large datasets.
- **GUI:** An easy-to-use interface built with Tkinter and ttkbootstrap.

### Advanced Simulation Features
- **Mass Inhomogeneity:** Simulate protein conformational broadening by defining a standard deviation for the average protein mass.
- **EMG Peak Shape:** Model more realistic chromatographic profiles by applying an Exponentially Modified Gaussian (EMG) shape with a configurable tailing factor.
- **1/f Noise:** Optionally add an extra layer of `1/f` (pink) noise for more realistic electronic noise simulation.

## Simulation Modules

The application provides several distinct simulation modes, each tailored for a specific analytical scenario. The following sections provide detailed documentation for each module and its parameters.

### General Spectrum Generator

This is the most fundamental simulation mode. It generates a mass spectrum for one or more species with defined masses and intensities.

**Parameters:**

*   **Protein Masses (`protein_masses`)**: A list of the average molecular weights (in Daltons) of the proteins or molecules you want to simulate.
*   **Intensity Scalars (`intensity_scalars`)**: A corresponding list of relative abundance values for each protein mass. The final intensities in the spectrum will be scaled according to these values.
*   **Mass Inhomogeneity**: A standard deviation (in Daltons) applied to each protein's mass. This simulates conformational or chemical heterogeneity, resulting in broadened peaks. The simulation achieves this by averaging several spectra generated from a normal distribution around the specified average mass.
*   **Isotopic Distribution (`isotopic_enabled`)**: If enabled, the simulation will calculate and generate a full isotopic distribution for each protein's charge state envelope, resulting in a more realistic spectrum. If disabled, only a single peak at the average mass of each charge state is generated.
*   **Resolution**: The mass spectrometer's resolving power (e.g., 60000). This directly influences the width of the peaks. Higher resolution leads to narrower, sharper peaks.
*   **Peak Sigma (`peak_sigma_mz`)**: An additional peak width parameter, defined in m/z units. It allows for peak broadening that is independent of the instrument's resolution setting.
*   **m/z Range (`mz_range_start`, `mz_range_end`)**: The start and end points of the mass-to-charge range to be simulated.
*   **m/z Step (`mz_step`)**: The sampling interval or "step size" across the m/z range. A smaller value (e.g., 0.01) results in a higher-resolution data file but increases generation time and file size.
*   **Noise (`noise_option`)**: The type of random noise to add to the spectrum's baseline.
    *   `None`: No noise is added.
    *   `Uniform`: Noise is drawn from a uniform distribution.
    *   `Gaussian`: Noise is drawn from a normal (Gaussian) distribution.
*   **1/f (Pink) Noise (`pink_noise_enabled`)**: If enabled, an additional layer of `1/f` noise is added. This type of noise is common in electronic systems and can make the baseline appear more realistic.
*   **Seed (`seed`)**: An integer used to initialize the random number generator. Using the same seed will produce the exact same noise and mass inhomogeneity samples, ensuring reproducibility.

### Antibody Simulation

This module is designed for the *in silico* analysis of monoclonal antibodies (mAbs) and bispecific antibodies. It automates the generation of various antibody species that can arise from a given set of heavy chains (HC) and light chains (LC), including fragments and fully assembled forms (e.g., H2L2).

The simulation first calculates all plausible combinatorial assemblies from the provided chains. It then calculates the precise average mass for each assembly, accounting for disulfide bond formation and optional post-translational modifications (PTMs). Finally, it simulates a single, combined mass spectrum representing the entire mixture.

**Chain Definition:**

The primary input is a list of antibody chains, where each chain is defined by:

*   **Type**: Whether the chain is a `HC` (Heavy Chain) or `LC` (Light Chain).
*   **Name**: A short, unique identifier for the chain (e.g., "HC1", "LC_A"). This name is used to construct the names of the final assemblies.
*   **Sequence**: The full amino acid sequence of the chain. This is used to calculate the chain's mass and total cysteine count.
*   **Pyro-Glu (`pyro_glu`)**: A boolean flag. If `True` and the sequence starts with 'E' (Glutamic acid) or 'Q' (Glutamine), the mass corresponding to a water loss will be subtracted to model pyroglutamate formation.
*   **K-Loss (`k_loss`)**: A boolean flag. If `True` and the sequence ends with 'K' (Lysine), the C-terminal lysine is removed to model this common modification.

**Assembly Generation and Mass Calculation:**

The simulator automatically generates a comprehensive list of all possible species based on the input chains, including:
*   Individual chains (HC, LC)
*   Chain dimers (H2, L2, H-L)
*   Fragments (H2L)
*   Fully assembled antibodies (H2L2)

For each assembly, the mass is calculated by:
1.  Summing the masses of the constituent chains (including any PTMs).
2.  Counting the total number of cysteine ('C') residues.
3.  Calculating the number of disulfide bonds (floor of total cysteines / 2).
4.  Subtracting the mass of two hydrogen atoms for each disulfide bond formed.

**Parameters:**

*   **Chains (`chains`)**: The list of chain definitions as described above.
*   **Assembly Abundances (`assembly_abundances`)**: A dictionary where keys are the names of the generated assemblies (e.g., "HC1HC1LC_ALC_A") and values are their desired relative abundances (intensities). This allows you to control the final ratio of the different species in the simulated spectrum.
*   All parameters from the **General Spectrum Generator** and **LC Simulation** are also applicable.

### Covalent Binding Simulator

This module simulates a covalent binding screen, where a target protein is mixed with a library of chemical compounds. The simulation models the stochastic nature of binding, allowing for the formation of the target protein with zero, one (DAR1), or two (DAR2) compounds attached.

**Parameters:**

*   **Protein Average Mass (`protein_avg_mass`)**: The average molecular weight of the primary target protein.
*   **Compound List File (`compound_list_file`)**: A path to a text file containing a list of compound masses, with one mass per line. The simulation will iterate through this list, creating a separate mzML file for each compound.
*   **Probability of Binding (`prob_binding`)**: The overall probability (from 0.0 to 1.0) that a given compound will bind to the target protein at all.
*   **Probability of DAR2 (`prob_dar2`)**: Of the proteins that *do* bind, this is the probability (from 0.0 to 1.0) that a second binding event will occur, forming a Drug-to-Antibody Ratio (DAR) 2 species. The probability of DAR1 is implicitly `1.0 - prob_dar2`.
*   **Total Binding Range (`total_binding_range`)**: A tuple `(min, max)` that defines a random +/- mass shift to be applied to the `prob_binding` for each compound. This adds variability to the binding probability across a screen.
*   **DAR2 Range (`dar2_range`)**: A tuple `(min, max)` that defines a random +/- mass shift to be applied to the `prob_dar2` for each compound.
*   All parameters from the **General Spectrum Generator** and **LC Simulation** are also applicable.

### Peptide Map Simulator

This module simulates a bottom-up proteomics workflow. It takes a protein sequence, performs an *in silico* tryptic digest, and then generates a spectrum for the resulting peptides.

**Parameters:**

*   **Sequence (`sequence`)**: The full amino acid sequence of the protein to be digested.
*   **Missed Cleavages (`missed_cleavages`)**: The maximum number of missed trypsin cleavage sites allowed within a single peptide. Trypsin cleaves after Lysine (K) and Arginine (R), unless followed by Proline (P). A value of 0 means only fully cleaved peptides are generated. A value of 1 allows for peptides containing a single missed cleavage site.
*   **Charge State (`charge_state`)**: The specific charge state to simulate for the peptides.
*   **Predict Charge (`predict_charge`)**: If enabled, the simulation will attempt to predict a plausible charge state for each peptide based on its sequence, overriding the fixed `charge_state` parameter.
*   All parameters from the **General Spectrum Generator** and **LC Simulation** are also applicable.

### Liquid Chromatography (LC) Simulation

When enabled, the simulator can generate a full LC-MS run, creating a multi-scan `.mzML` file that represents how species elute from a chromatography column over time.

**Elution Order:**

In the **Antibody Simulation**, the elution order of the different species is determined by their **hydrophobicity**. The module calculates a Kyte-Doolittle hydrophobicity score for each generated assembly based on its amino acid sequence. The species are then eluted in order of increasing hydrophobicity, with the most hydrophobic species eluting last.

**Parameters:**

*   **LC Enabled (`enabled`)**: A boolean flag to turn the entire LC simulation on or off. If `False`, a single, combined spectrum is generated.
*   **Number of Scans (`num_scans`)**: The total number of mass spectra (scans) to generate in the mzML file. This corresponds to the duration of the LC run.
*   **Scan Interval (`scan_interval`)**: The time (in seconds) between each scan.
*   **Peak Width (`gaussian_std_dev`)**: The standard deviation of the chromatographic peak in the time dimension, measured in number of scans. A larger value results in broader peaks.
*   **Tailing Factor (`lc_tailing_factor`)**: A parameter to simulate asymmetric peak shapes. A value of `1.0` produces a perfect Gaussian peak. Values greater than `1.0` introduce a "tail" to the peak, creating a more realistic Exponentially Modified Gaussian (EMG) shape.

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

    The project dependencies are defined in `pyproject.toml`. You can install them directly using pip:
    ```bash
    pip install .
    ```
    For development purposes (e.g., to run tests), you can install the project in "editable" mode along with the testing dependencies:
    ```bash
    pip install -e .[test]
    ```

## Usage

Once installed, you can run the application by executing the installed script:
```bash
spec-generator
```
Alternatively, from the root of the repository, you can run the main module directly:
```bash
python -m src.spec_generator.main
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
