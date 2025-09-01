from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class Spectrum:
    """
    Represents a mass spectrum.

    Attributes:
        mz: A numpy array of m/z values.
        intensity: A numpy array of intensity values.
    """
    mz: np.ndarray
    intensity: np.ndarray

@dataclass
class FragmentationEvent:
    """
    Represents a single fragmentation event.

    Attributes:
        precursor_mz: The m/z of the precursor ion.
        precursor_charge: The charge of the precursor ion.
        rt: The retention time of the fragmentation event.
        intensity: The intensity of the precursor ion.
        fragments: A Spectrum object containing the fragment ions.
    """

    precursor_mz: float
    precursor_charge: int
    rt: float
    intensity: float
    fragments: Spectrum

@dataclass
class MSMSSpectrum:
    """
    Represents an MS/MS spectrum, containing a list of fragmentation events.
    """
    rt: float
    precursor_mz: float
    precursor_charge: int
    fragmentation_events: List[FragmentationEvent] = field(default_factory=list)
