from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from .logic.ptm import Ptm

@dataclass
class CommonParams:
    """
    Data class for common parameters shared across simulation types.
    """
    isotopic_enabled: bool
    resolution: float
    peak_sigma_mz: float
    mz_step: float
    mz_range_start: float
    mz_range_end: float
    noise_option: str
    pink_noise_enabled: bool
    output_directory: str
    seed: int
    filename_template: str
    # Tech debt remediation: Add MS/MS config
    msms_enabled: bool = False
    msms_ion_types: List[str] = field(default_factory=lambda: ['y', 'b'])
    msms_fragment_charges: List[int] = field(default_factory=lambda: [1])
    msms_precursor_charges: List[int] = field(default_factory=lambda: [2])


@dataclass
class LCParams:
    """
    Data class for LC simulation parameters.
    """
    enabled: bool
    num_scans: int
    scan_interval: float
    gaussian_std_dev: float
    lc_tailing_factor: float
    retention_time_model: str = "rpc"
    rpc_hydrophobicity_coefficient: float = 0.05

@dataclass
class SpectrumGeneratorConfig:
    """
    Configuration for the Spectrum Generator simulation.
    """
    common: CommonParams
    lc: LCParams
    protein_list_file: str | None
    protein_masses: List[float]
    intensity_scalars: List[float]
    mass_inhomogeneity: float
    hydrophobicity_scores: List[float] | None = None
    peptide_sequences: List[str] | None = None
    # Tech debt remediation: Add inhomogeneity samples
    mass_inhomogeneity_samples: int = 7

@dataclass
class CovalentBindingConfig:
    """
    Configuration for the Covalent Binding simulation.
    """
    common: CommonParams
    lc: LCParams
    protein_avg_mass: float
    compound_list_file: str
    prob_binding: float
    prob_dar2: float
    total_binding_range: Tuple[float, float]
    dar2_range: Tuple[float, float]

@dataclass
class Chain:
    """
    Represents a single antibody chain.
    """
    type: str
    name: str
    seq: str
    pyro_glu: bool
    k_loss: bool
    ptms: List[Ptm] = field(default_factory=list)

@dataclass
class AntibodySimConfig:
    """
    Configuration for the Antibody simulation.
    """
    common: CommonParams
    lc: LCParams
    chains: List[Chain]
    assembly_abundances: Dict[str, float]


@dataclass
class PeptideMapLCParams:
    """
    Data class for LC parameters specific to peptide map simulations.
    """
    run_time: float  # in minutes
    scan_interval: float  # in seconds
    peak_width_seconds: float # peak width at base in seconds


@dataclass
class PeptideMapSimConfig:
    """
    Configuration for the Peptide Map simulation.
    """
    common: CommonParams
    lc: PeptideMapLCParams
    sequence: str
    missed_cleavages: int
    charge_state: int
    predict_charge: bool = False
