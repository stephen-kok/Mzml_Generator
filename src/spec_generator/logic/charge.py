"""
This module provides functions for predicting peptide charge states.
"""
import numpy as np

def predict_charge_states(peptide_sequence: str) -> dict[int, float]:
    """
    Predicts a distribution of charge states for a given peptide sequence.

    The prediction is based on the number of basic amino acid residues
    (K, R, H), which are likely to be protonated. The most likely charge
    state is assumed to be `1 + number_of_basic_residues`.

    A normal distribution of intensities is generated around this primary
    charge state.

    Args:
        peptide_sequence: The amino acid sequence of the peptide.

    Returns:
        A dictionary where keys are charge states (int) and values are
        their relative intensities (float). Returns {2: 1.0} for empty
        or very short sequences as a default.
    """
    if not peptide_sequence or len(peptide_sequence) < 3:
        return {2: 1.0}  # Default for very short/empty sequences

    # Count basic residues
    num_basic_residues = sum(peptide_sequence.count(res) for res in "KRH")

    # Primary charge is typically 1 (N-terminus) + number of basic residues
    primary_charge = 1 + num_basic_residues

    # Generate a small range of charges around the primary charge
    # e.g., for primary_charge = 3, charges will be [2, 3, 4]
    charges = np.arange(
        max(1, primary_charge - 1),
        primary_charge + 2
    )

    # Generate relative intensities using a normal distribution
    # The peak of the distribution is at the primary_charge
    mean = primary_charge
    std_dev = 0.5  # Controls the spread of the distribution

    # Using a simplified normal distribution calculation
    intensities = np.exp(-0.5 * ((charges - mean) / std_dev) ** 2)

    # Normalize intensities so the max is 1.0
    if np.max(intensities) > 0:
        intensities /= np.max(intensities)

    # Filter out very low intensity charges
    charge_distribution = {
        int(charge): intensity
        for charge, intensity in zip(charges, intensities)
        if intensity > 0.1
    }

    # Ensure the primary charge is always included if possible
    if not charge_distribution and primary_charge >= 1:
        charge_distribution[primary_charge] = 1.0

    # Handle cases where all charges were filtered out
    if not charge_distribution:
        return {2: 1.0} # Fallback

    return charge_distribution
