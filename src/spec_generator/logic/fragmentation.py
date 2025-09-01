"""
This module provides functions for generating fragment ions from peptide sequences.
"""
from typing import Optional
from ..core.constants import (
    AMINO_ACID_MASSES,
    FRAGMENT_ION_MODIFICATIONS,
    NEUTRAL_LOSS_RULES,
    PROTON_MASS,
)


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
    sequence: str, prefix_masses: list[float], ion_type: str, charges: list[int]
) -> list[float]:
    """Generates m/z values for N-terminal ions (a, b, c), including neutral losses."""
    fragment_mzs = []
    modification = FRAGMENT_ION_MODIFICATIONS.get(ion_type)
    if modification is None:
        return []

    for i in range(len(prefix_masses) - 1):
        neutral_mass = prefix_masses[i] + modification
        fragment_sequence = sequence[: i + 1]

        # Add the primary ion
        for charge in charges:
            if charge == 0: continue
            mz = (neutral_mass + charge * PROTON_MASS) / charge
            fragment_mzs.append(mz)

        # Handle neutral losses
        possible_losses = set()
        for aa in fragment_sequence:
            if aa in NEUTRAL_LOSS_RULES:
                for loss in NEUTRAL_LOSS_RULES[aa]:
                    possible_losses.add(loss)

        for loss_mass in possible_losses:
            loss_neutral_mass = neutral_mass - loss_mass
            for charge in charges:
                if charge == 0: continue
                mz = (loss_neutral_mass + charge * PROTON_MASS) / charge
                fragment_mzs.append(mz)

    return fragment_mzs


def _generate_c_terminal_ions(
    sequence: str, suffix_masses: list[float], ion_type: str, charges: list[int]
) -> list[float]:
    """Generates m/z values for C-terminal ions (x, y, z), including neutral losses."""
    fragment_mzs = []
    modification = FRAGMENT_ION_MODIFICATIONS.get(ion_type)
    if modification is None:
        return []

    for i in range(len(suffix_masses) - 1):
        neutral_mass = suffix_masses[i + 1] + modification
        fragment_sequence = sequence[i + 1 :]

        # Add the primary ion
        for charge in charges:
            if charge == 0: continue
            mz = (neutral_mass + charge * PROTON_MASS) / charge
            fragment_mzs.append(mz)

        # Handle neutral losses
        possible_losses = set()
        for aa in fragment_sequence:
            if aa in NEUTRAL_LOSS_RULES:
                for loss in NEUTRAL_LOSS_RULES[aa]:
                    possible_losses.add(loss)

        for loss_mass in possible_losses:
            loss_neutral_mass = neutral_mass - loss_mass
            for charge in charges:
                if charge == 0: continue
                mz = (loss_neutral_mass + charge * PROTON_MASS) / charge
                fragment_mzs.append(mz)

    return fragment_mzs


def generate_fragment_ions(
    sequence: str,
    ion_types: list[str],
    charges: list[int],
    ptms: Optional[dict[int, float]] = None,
) -> list[float]:
    """
    Generates a list of m/z values for specified fragment ions.

    Args:
        sequence: The amino acid sequence of the peptide.
        ion_types: A list of ion types to generate (e.g., ['b', 'y']).
        charges: A list of charge states to consider for each fragment.
        ptms: A dictionary of post-translational modifications, where keys are
              0-based residue indices and values are the mass shifts.

    Returns:
        A list of calculated m/z values for the fragment ions.
    """
    if not sequence:
        return []

    fragment_mzs = []
    n_term_ion_types = {'a', 'b', 'c'}
    c_term_ion_types = {'x', 'y', 'z'}

    prefix_masses = _calculate_prefix_masses(sequence, ptms)
    suffix_masses = _calculate_suffix_masses(sequence, ptms)

    for ion_type in ion_types:
        if ion_type in n_term_ion_types:
            fragment_mzs.extend(
                _generate_n_terminal_ions(sequence, prefix_masses, ion_type, charges)
            )
        elif ion_type in c_term_ion_types:
            fragment_mzs.extend(
                _generate_c_terminal_ions(sequence, suffix_masses, ion_type, charges)
            )

    return sorted(set(fragment_mzs))
