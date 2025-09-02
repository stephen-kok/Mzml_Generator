import unittest
import numpy as np
from spec_generator.core.spectrum import generate_fragment_spectrum
from spec_generator.logic.fragmentation import generate_fragment_ions

class TestSpectrum(unittest.TestCase):
    def test_generate_fragment_spectrum_with_isotopes(self):
        """
        Test that generate_fragment_spectrum produces a spectrum with isotopic
        distributions for fragment ions.
        """
        sequence = "PEPTIDE"
        ion_types = ["y"]
        fragment_charges = [1]
        mz_range = np.arange(100.0, 1000.0, 0.01)

        # First, find out how many unique fragment ions we expect.
        fragment_definitions = generate_fragment_ions(
            sequence=sequence,
            ion_types=ion_types,
            charges=fragment_charges,
        )
        num_fragment_ions = len(fragment_definitions)
        self.assertGreater(num_fragment_ions, 0, "Should generate at least one fragment ion")

        # Now, generate the full spectrum.
        spectrum = generate_fragment_spectrum(
            peptide_sequence=sequence,
            mz_range=mz_range,
            peak_sigma_mz_float=0.01,
            intensity_scalar=1.0,
            resolution=10000,
            ion_types=ion_types,
            fragment_charges=fragment_charges,
        )

        self.assertIsNotNone(spectrum)
        self.assertIsInstance(spectrum, np.ndarray)

        # The total number of peaks should be greater than the number of fragment ions,
        # because each ion now has an isotopic distribution.
        num_peaks = np.count_nonzero(spectrum)
        # This is a probabilistic test, but for a simple peptide it should hold true.
        # A simple check is that num_peaks > num_fragment_ions. A more robust one
        # might be num_peaks > num_fragment_ions * 1.5, assuming at least an A+1 peak
        # for most fragments.
        self.assertGreater(num_peaks, num_fragment_ions,
                         "Spectrum should have more peaks than fragment ions due to isotopes.")

        # Check for a specific, known isotopic peak.
        # The y1 ion is 'E' + H2O. Let's find its expected m/z.
        # pyteomics can calculate this for us.
        from pyteomics import mass
        y1_comp = mass.Composition("E") + mass.Composition({"H": 2, "O": 1})
        dist = list(mass.isotopologues(composition=y1_comp, report_abundance=True))

        self.assertGreater(len(dist), 1, "y1 ion should have at least 2 isotopic peaks")

        # Get the m/z of the two most abundant isotopes by calculating them
        dist.sort(key=lambda x: x[1], reverse=True)
        y1_mono_mz = mass.calculate_mass(composition=dist[0][0], charge=1)
        y1_a_plus_1_mz = mass.calculate_mass(composition=dist[1][0], charge=1)

        # Check if these m/z values have non-zero intensity in the spectrum
        mono_idx = np.searchsorted(mz_range, y1_mono_mz)
        a_plus_1_idx = np.searchsorted(mz_range, y1_a_plus_1_mz)

        # Check within a small tolerance
        self.assertGreater(np.sum(spectrum[mono_idx-2:mono_idx+2]), 0,
                         "Monoisotopic peak for y1 ion should be present.")
        self.assertGreater(np.sum(spectrum[a_plus_1_idx-2:a_plus_1_idx+2]), 0,
                         "A+1 isotopic peak for y1 ion should be present.")

if __name__ == "__main__":
    unittest.main()
