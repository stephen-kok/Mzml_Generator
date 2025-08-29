"""
This module provides a more accurate isotope distribution calculator for peptides,
based on their elemental composition, using pyteomics.
"""
import numpy as np
from pyteomics import mass

class PeptideIsotopeCalculator:
    """
    Calculates the theoretical isotopic distribution of a peptide based on its
    elemental composition using pyteomics.
    """
    def get_distribution(self, sequence: str, charge: int = 1) -> list[tuple[float, float]]:
        """
        Calculates the isotopic distribution for a given peptide sequence.

        Args:
            sequence: The amino acid sequence of the peptide.
            charge: The charge state of the peptide.

        Returns:
            A list of (m/z, relative_intensity) tuples, where the most
            abundant peak has an intensity of 1.0.
        """
        if not sequence:
            return []

        try:
            # isotopologues gives an iterator over (Composition, abundance) tuples
            dist = list(mass.isotopologues(
                sequence=sequence,
                report_abundance=True,
                overall_threshold=1e-6 # Prune very low abundance peaks
            ))

            if not dist:
                return []

            # Calculate m/z for each isotopic composition
            mzs = [mass.calculate_mass(composition=comp, charge=charge) for comp, abund in dist]
            abundances = np.array([abund for comp, abund in dist])

            max_abundance = np.max(abundances)
            if max_abundance < 1e-12:
                return []

            # Normalize intensities so the most abundant is 1.0
            relative_intensities = abundances / max_abundance

            # Sort by m/z before returning
            sorted_distribution = sorted(zip(mzs, relative_intensities), key=lambda x: x[0])

            return sorted_distribution

        except Exception:
            # Fallback for invalid sequences
            return []

# Singleton instance to be used across the application
peptide_isotope_calculator = PeptideIsotopeCalculator()
