"""
This module provides functions for predicting peptide retention times based on
hydrophobicity.
"""
import numpy as np

# Kyte-Doolittle hydrophobicity scale
# Reference: J. Mol. Biol. (1982) 157, 105-132
KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

def predict_retention_times(peptides: list[str], total_run_time_minutes: float = 60.0) -> dict[str, float]:
    """
    Predicts the retention time for a list of peptides based on their
    Kyte-Doolittle hydrophobicity. The scores are then linearly scaled
    to fit within the specified total run time.

    Args:
        peptides: A list of peptide amino acid sequences.
        total_run_time_minutes: The total duration of the LC gradient.

    Returns:
        A dictionary mapping each peptide sequence to its predicted retention time in minutes.
    """
    if not peptides:
        return {}

    scores = []
    for peptide in peptides:
        # Sum the hydrophobicity score for each amino acid in the peptide
        # Use a default of 0 for any non-standard amino acids
        score = sum(KYTE_DOOLITTLE.get(aa.upper(), 0) for aa in peptide)
        scores.append(score)

    scores = np.array(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    # Avoid division by zero if all peptides have the same score
    if max_score == min_score:
        # If all scores are the same, place them in the middle of the run
        return {peptide: total_run_time_minutes / 2 for peptide in peptides}

    # Linearly scale the scores to the run time
    scaled_rts = (scores - min_score) / (max_score - min_score) * total_run_time_minutes

    return dict(zip(peptides, scaled_rts))


def calculate_apex_scans_from_hydrophobicity(
    scores: list[float],
    num_scans: int,
    retention_time_model: str = "linear",
    rpc_hydrophobicity_coefficient: float = 0.05,
) -> list[int]:
    """
    Calculates the apex scan for each species based on its hydrophobicity score.
    Supports different models for scaling retention time.

    Args:
        scores: A list of hydrophobicity scores.
        num_scans: The total number of scans in the LC run.
        retention_time_model: The model to use for scaling ('linear' or 'rpc').
        rpc_hydrophobicity_coefficient: Coefficient for the RPC model.

    Returns:
        A list of integer scan indices for the apex of elution.
    """
    if not scores:
        return []

    scores_arr = np.array(scores, dtype=float)
    min_score, max_score = np.min(scores_arr), np.max(scores_arr)

    if max_score == min_score:
        return [int(num_scans / 2)] * len(scores)

    scan_padding = int(num_scans * 0.1)
    usable_scan_range = num_scans - 2 * scan_padding

    if retention_time_model == "rpc":
        # Reversed-Phase Chromatography (RPC) model
        # Transform scores to be non-negative
        normalized_scores = scores_arr - min_score
        # Apply exponential scaling
        exp_scores = np.exp(normalized_scores * rpc_hydrophobicity_coefficient)
        min_exp_score, max_exp_score = np.min(exp_scores), np.max(exp_scores)

        if max_exp_score == min_exp_score:
             scaled_scans = np.full_like(scores_arr, usable_scan_range / 2)
        else:
            scaled_scans = (
                (exp_scores - min_exp_score) / (max_exp_score - min_exp_score)
            ) * usable_scan_range

    else:  # Linear model
        scaled_scans = (
            (scores_arr - min_score) / (max_score - min_score) * usable_scan_range
        )

    return [int(s + scan_padding) for s in scaled_scans]
