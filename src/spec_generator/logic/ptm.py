"""
This module defines the data structures and core logic for handling
Post-Translational Modifications (PTMs). It allows for the stochastic
application of PTMs to a protein sequence to calculate a total mass shift.
"""
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class Ptm:
    """
    Represents a single Post-Translational Modification.

    Attributes:
        name: The common name of the modification (e.g., "Oxidation").
        mass_shift: The mass difference (in Daltons) to be added to the peptide.
        residue: The one-letter amino acid code that this PTM targets.
        probability: The chance (0.0 to 1.0) that this PTM will occur at any given target site.
    """
    name: str
    mass_shift: float
    residue: str
    probability: float = 0.0

# A library of common PTMs that can be used as defaults in the GUI.
# Users can select from this list and customize the probability.
DEFAULT_PTMS = {
    "Oxidation": Ptm(name="Oxidation", mass_shift=15.994915, residue="M"),
    "Deamidation": Ptm(name="Deamidation", mass_shift=0.984016, residue="N"),
    "Guanidination": Ptm(name="Guanidination", mass_shift=42.0218, residue="K"),
    "Methylation": Ptm(name="Methylation", mass_shift=14.01565, residue="K"),
}

def calculate_ptm_mass_shift(sequence: str, ptm_configs: List[Ptm]) -> float:
    """
    Calculates the total mass shift for a sequence based on a list of PTMs,
    applied stochastically.

    Args:
        sequence: The amino acid sequence of the protein/peptide.
        ptm_configs: A list of Ptm objects, each configured with a specific
                     probability for this simulation.

    Returns:
        The total mass shift from all applied PTMs.
    """
    total_mass_shift = 0.0
    if not ptm_configs:
        return total_mass_shift

    # Create a map of residue -> list of PTMs for efficient lookup
    ptm_map = {}
    for ptm in ptm_configs:
        if ptm.residue not in ptm_map:
            ptm_map[ptm.residue] = []
        ptm_map[ptm.residue].append(ptm)

    # Iterate through each amino acid in the sequence
    for amino_acid in sequence:
        # Check if this amino acid is a target for any configured PTMs
        if amino_acid in ptm_map:
            # For each PTM that targets this amino acid, roll the dice
            for ptm in ptm_map[amino_acid]:
                if ptm.probability > 0 and random.random() < ptm.probability:
                    total_mass_shift += ptm.mass_shift

    return total_mass_shift
