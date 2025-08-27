import numpy as np
from scipy.stats import poisson

from .constants import NEUTRON_MASS_APPROX


class IsotopeCalculator:
    """
    Calculates the theoretical isotopic distribution of a given mass using a
    Poisson distribution based on the 'averagine' model.
    """
    AVERAGINE_MASS_PER_NEUTRON_PROB = 1110.0

    def get_distribution(self, mass: float, num_peaks: int = 40) -> tuple[list[tuple[float, float]], float]:
        """
        Calculates the isotopic distribution for a given mass.

        Args:
            mass: The average mass of the protein/molecule.
            num_peaks: The maximum number of isotopic peaks to calculate.

        Returns:
            A tuple containing:
            - A list of (mass_offset, relative_intensity) tuples.
            - The mass offset of the most abundant isotope.
        """
        if mass <= 0:
            return [(0.0, 1.0)], 0.0

        lambda_val = mass / self.AVERAGINE_MASS_PER_NEUTRON_PROB

        # Estimate required number of peaks to avoid unnecessary calculation
        k = np.arange(num_peaks)
        probabilities = poisson.pmf(k, lambda_val)

        significant_indices = np.where(probabilities > 1e-5)[0]
        if len(significant_indices) > 0:
            last_significant_index = significant_indices[-1]
        else:
            # Fallback for very large masses where initial check might fail
            last_significant_index = int(lambda_val) + 5

        num_peaks_to_calc = min(num_peaks, last_significant_index + 3)

        # Recalculate with a more appropriate number of peaks
        k = np.arange(num_peaks_to_calc)
        probabilities = poisson.pmf(k, lambda_val)

        mass_offsets = k * NEUTRON_MASS_APPROX

        max_prob = np.max(probabilities)
        if max_prob < 1e-12:  # Avoid division by zero or near-zero
            return [(0.0, 1.0)], 0.0

        normalized_probs = probabilities / max_prob

        most_abundant_isotope_index = np.argmax(probabilities)
        most_abundant_offset = mass_offsets[most_abundant_isotope_index]

        distribution = list(zip(mass_offsets, normalized_probs))

        return distribution, most_abundant_offset

# Singleton instance to be used across the application
isotope_calculator = IsotopeCalculator()
