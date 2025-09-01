"""
This module provides functions for generating fragment ions from peptide sequences.
"""
from ..core.constants import (
    AMINO_ACID_MASSES,
    FRAGMENT_ION_MODIFICATIONS,
    PROTON_MASS,
)


def generate_fragment_ions(
    sequence: str, ion_types: list[str], charges: list[int]
) -> list[float]:
    """
    Generates a list of m/z values for specified fragment ions.

    Args:
        sequence: The amino acid sequence of the peptide.
        ion_types: A list of ion types to generate (e.g., ['b', 'y']).
        charges: A list of charge states to consider for each fragment.

    Returns:
        A list of calculated m/z values for the fragment ions.
    """
    if not sequence:
        return []

    fragment_mzs = []
    n_term_ion_types = {'b', 'c'}
    c_term_ion_types = {'y', 'z'}

    # Pre-calculate prefix sums of residue masses for efficient N-terminal ion calculation
    prefix_masses = [0.0] * len(sequence)
    current_mass = 0.0
    for i, aa in enumerate(sequence):
        current_mass += AMINO_ACID_MASSES.get(aa, 0.0)
        prefix_masses[i] = current_mass

    # Pre-calculate suffix sums of residue masses for efficient C-terminal ion calculation
    suffix_masses = [0.0] * len(sequence)
    current_mass = 0.0
    # Iterate backwards to get sums from C-terminus
    for i, aa in enumerate(reversed(sequence)):
        current_mass += AMINO_ACID_MASSES.get(aa, 0.0)
        suffix_masses[len(sequence) - 1 - i] = current_mass

    for ion_type in ion_types:
        modification = FRAGMENT_ION_MODIFICATIONS.get(ion_type)
        if modification is None:
            continue  # Skip unsupported ion types

        # Iterate through all possible cleavage points (len(sequence) - 1 points)
        for i in range(len(sequence) - 1):
            neutral_mass = 0.0
            if ion_type in n_term_ion_types:
                # Fragment is from N-terminus, e.g., b1, b2, ...
                # The fragment length is i + 1, corresponding to prefix_masses[i]
                neutral_mass = prefix_masses[i] + modification
            elif ion_type in c_term_ion_types:
                # Fragment is from C-terminus, e.g., y1, y2, ...
                # The fragment length is len(sequence) - (i + 1)
                # This corresponds to suffix_masses[i+1]
                neutral_mass = suffix_masses[i + 1] + modification

            if neutral_mass > 0:
                for charge in charges:
                    if charge == 0: continue
                    mz = (neutral_mass + charge * PROTON_MASS) / charge
                    fragment_mzs.append(mz)

    return sorted(set(fragment_mzs))
