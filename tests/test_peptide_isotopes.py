import unittest
from spec_generator.core.peptide_isotopes import PeptideIsotopeCalculator

class TestPeptideIsotopeCalculator(unittest.TestCase):

    def setUp(self):
        """Set up a new PeptideIsotopeCalculator instance for each test."""
        self.calculator = PeptideIsotopeCalculator()

    def test_empty_sequence(self):
        """Test that an empty sequence returns an empty list."""
        result = self.calculator.get_distribution("")
        self.assertEqual(result, [])

    def test_invalid_sequence(self):
        """Test that a sequence with only invalid characters returns an empty list."""
        result = self.calculator.get_distribution("123")
        self.assertEqual(result, [])

    def test_most_abundant_peak_is_normalized(self):
        """Test that the most abundant peak has a relative intensity of 1.0."""
        distribution = self.calculator.get_distribution("PEPTIDE")
        intensities = [p[1] for p in distribution]
        self.assertAlmostEqual(max(intensities), 1.0, places=5, msg="Maximum intensity should be normalized to 1.0")

    def test_glycine_charge_1_quantitative(self):
        """
        Test the isotopic distribution for Glycine (G) at charge 1.
        This test locks in the expected behavior of the pyteomics backend.
        """
        distribution = self.calculator.get_distribution("G", charge=1)

        from pyteomics.mass import calculate_mass
        expected_mz = calculate_mass(sequence="G", charge=1)
        self.assertAlmostEqual(
            distribution[0][0], expected_mz, places=3,
            msg="Monoisotopic m/z for Glycine is incorrect."
        )

        # This value is based on the known output of pyteomics for this input.
        expected_a1_abundance = 0.00365
        self.assertAlmostEqual(
            distribution[1][1], expected_a1_abundance, places=3,
            msg="A+1 peak abundance for Glycine does not match regression baseline."
        )

    def test_peptide_distribution_is_plausible(self):
        """
        Test that the isotope distribution for a peptide is plausible.
        - The first peak should match the theoretical monoisotopic m/z.
        - The distribution should be normalized to 1.0.
        """
        sequence = "PEPTIDE"
        charge = 2
        distribution = self.calculator.get_distribution(sequence, charge=charge)

        from pyteomics.mass import calculate_mass
        expected_mono_mz = calculate_mass(sequence=sequence, charge=charge)

        self.assertAlmostEqual(
            expected_mono_mz, distribution[0][0], places=2,
            msg="Monoisotopic m/z of the first peak is incorrect."
        )

        intensities = [p[1] for p in distribution]
        self.assertAlmostEqual(
            max(intensities), 1.0, places=5,
            msg="The most abundant peak should be normalized to 1.0."
        )

if __name__ == '__main__':
    unittest.main()
