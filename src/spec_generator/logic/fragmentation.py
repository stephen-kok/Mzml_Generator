"""
This module provides functions for generating fragment ions from peptide sequences.
"""
from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..core.constants import (
    AMINO_ACID_MASSES,
    FRAGMENT_ION_MODIFICATIONS,
    NEUTRAL_LOSS_RULES,
    PROTON_MASS,
    FRAGMENTATION_ENHANCEMENT_RULES,
)
from ..core.types import Spectrum, FragmentationEvent


from typing import Optional

@dataclass
class FragmentIon:
    """Represents the definition of a fragment ion before isotope calculation."""
    sequence: str
    ion_type: str
    charge: int
    intensity: float
    neutral_mass: float
    neutral_loss: Optional[str] = None


def _calculate_prefix_masses(
    sequence: str, ptms: Optional[dict[int, float]] = None
) -> list[float]:
    """
    Calculates the cumulative mass from the N-terminus for each residue,
    including any PTMs.
    """
    ptms = ptms or {}
    prefix_masses = [0.0] * len(sequence)
    current_mass = 0.0
    for i, aa in enumerate(sequence):
        current_mass += AMINO_ACID_MASSES.get(aa, 0.0)
        if i in ptms:
            current_mass += ptms[i]
        prefix_masses[i] = current_mass
    return prefix_masses


def _calculate_suffix_masses(
    sequence: str, ptms: Optional[dict[int, float]] = None
) -> list[float]:
    """
    Calculates the cumulative mass from the C-terminus for each residue,
    including any PTMs.
    """
    ptms = ptms or {}
    suffix_masses = [0.0] * len(sequence)
    current_mass = 0.0
    for i, aa in enumerate(reversed(sequence)):
        current_mass += AMINO_ACID_MASSES.get(aa, 0.0)
        # Original index from the left
        original_index = len(sequence) - 1 - i
        if original_index in ptms:
            current_mass += ptms[original_index]
        suffix_masses[original_index] = current_mass
    return suffix_masses


def _generate_n_terminal_ions(
    sequence: str, prefix_masses: list[float], ion_type: str
) -> List[Tuple[str, float, float, Optional[str]]]:
    """
    Generates definitions for N-terminal ions, including neutral losses.
    Returns tuples of (sequence, intensity, neutral_mass, neutral_loss_formula).
    """
    fragments = []
    modification = FRAGMENT_ION_MODIFICATIONS.get(ion_type)
    if modification is None:
        return []

    for i in range(len(prefix_masses) - 1):
        neutral_mass = prefix_masses[i] + modification
        fragment_sequence = sequence[: i + 1]
        cleavage_aa = sequence[i]

        intensity_factor = FRAGMENTATION_ENHANCEMENT_RULES.get(
            cleavage_aa, FRAGMENTATION_ENHANCEMENT_RULES["default"]
        )
        base_intensity = 100.0 * intensity_factor
        fragments.append((fragment_sequence, base_intensity, neutral_mass, None))

        # Handle neutral losses
        possible_losses = set()
        for aa in fragment_sequence:
            if aa in NEUTRAL_LOSS_RULES:
                for loss_formula in NEUTRAL_LOSS_RULES[aa]:
                    possible_losses.add(loss_formula)

        for loss_formula in possible_losses:
            # Note: neutral_mass is not adjusted here. The composition will be
            # adjusted later during spectrum generation.
            loss_intensity = base_intensity * 0.2
            fragments.append((fragment_sequence, loss_intensity, neutral_mass, loss_formula))

    return fragments


def _generate_c_terminal_ions(
    sequence: str, suffix_masses: list[float], ion_type: str
) -> List[Tuple[str, float, float, Optional[str]]]:
    """
    Generates definitions for C-terminal ions, including neutral losses.
    Returns tuples of (sequence, intensity, neutral_mass, neutral_loss_formula).
    """
    fragments = []
    modification = FRAGMENT_ION_MODIFICATIONS.get(ion_type)
    if modification is None:
        return []

    for i in range(len(suffix_masses) - 1):
        neutral_mass = suffix_masses[i + 1] + modification
        fragment_sequence = sequence[i + 1 :]
        cleavage_aa = sequence[i + 1]

        intensity_factor = FRAGMENTATION_ENHANCEMENT_RULES.get(
            cleavage_aa, FRAGMENTATION_ENHANCEMENT_RULES["default"]
        )
        base_intensity = 100.0 * intensity_factor
        fragments.append((fragment_sequence, base_intensity, neutral_mass, None))

        # Handle neutral losses
        possible_losses = set()
        for aa in fragment_sequence:
            if aa in NEUTRAL_LOSS_RULES:
                for loss_formula in NEUTRAL_LOSS_RULES[aa]:
                    possible_losses.add(loss_formula)

        for loss_formula in possible_losses:
            loss_intensity = base_intensity * 0.2
            fragments.append((fragment_sequence, loss_intensity, neutral_mass, loss_formula))

    return fragments


def generate_fragment_ions(
    sequence: str,
    ion_types: list[str],
    charges: list[int],
    ptms: Optional[dict[int, float]] = None,
) -> List[FragmentIon]:
    """
    Generates a list of FragmentIon objects for specified ion types and charges.
    """
    if not sequence:
        return []

    fragment_definitions = []
    n_term_ion_types = {"a", "b", "c"}
    c_term_ion_types = {"x", "y", "z"}

    prefix_masses = _calculate_prefix_masses(sequence, ptms)
    suffix_masses = _calculate_suffix_masses(sequence, ptms)

    # Generate sequence/intensity definitions
    base_fragments = {}  # Use dict to store unique sequence definitions
    for ion_type in set(ion_types):
        if ion_type in n_term_ion_types:
            ion_defs = _generate_n_terminal_ions(sequence, prefix_masses, ion_type)
        elif ion_type in c_term_ion_types:
            ion_defs = _generate_c_terminal_ions(sequence, suffix_masses, ion_type)
        else:
            continue

        for frag_seq, intensity, neutral_mass, neutral_loss in ion_defs:
            # Store the most intense definition for a given sequence, ion type, and loss
            key = (frag_seq, ion_type, neutral_loss)
            if key not in base_fragments or intensity > base_fragments[key][0]:
                base_fragments[key] = (intensity, neutral_mass)

    # Create charged ions from definitions
    for (frag_seq, ion_type, neutral_loss), (intensity, neutral_mass) in base_fragments.items():
        for charge in charges:
            if charge == 0:
                continue
            fragment_definitions.append(
                FragmentIon(
                    sequence=frag_seq,
                    ion_type=ion_type,
                    charge=charge,
                    intensity=intensity,
                    neutral_mass=neutral_mass,
                    neutral_loss=neutral_loss,
                )
            )

    return fragment_definitions


# This import is circular, so we need to move it inside the function
# from ..core.spectrum import generate_fragment_spectrum
import numpy as np


def generate_fragmentation_events(
    sequence: str,
    precursor_charge: int,
    ion_types: list[str],
    fragment_charges: list[int],
    rt: float,
    precursor_intensity: float,
    config, # Pass the whole config object for simplicity
) -> list[FragmentationEvent]:
    """
    Generates a list of fragmentation events for a peptide, including a
    fully realized fragment spectrum.
    """
    # To avoid circular import at the module level
    from ..core.spectrum import generate_fragment_spectrum

    if not sequence:
        return []

    common = config.common
    mz_range = np.arange(
        common.mz_range_start, common.mz_range_end + common.mz_step, common.mz_step
    )

    # Generate the fragment spectrum array
    fragment_intensity_array = generate_fragment_spectrum(
        peptide_sequence=sequence,
        mz_range=mz_range,
        peak_sigma_mz_float=common.peak_sigma_mz,
        intensity_scalar=1.0,  # Intensity is handled by precursor_intensity
        resolution=common.resolution,
        ion_types=ion_types,
        fragment_charges=fragment_charges,
    )

    # Create the Spectrum object for the FragmentationEvent
    fragments = Spectrum(
        mz=mz_range,
        intensity=fragment_intensity_array,
    )

    # Calculate precursor m/z for the event
    total_mass = sum(AMINO_ACID_MASSES.get(aa, 0.0) for aa in sequence)
    precursor_mz = (total_mass + precursor_charge * PROTON_MASS) / precursor_charge

    event = FragmentationEvent(
        precursor_mz=precursor_mz,
        precursor_charge=precursor_charge,
        rt=rt,
        intensity=precursor_intensity,
        fragments=fragments,
    )

    return [event]
