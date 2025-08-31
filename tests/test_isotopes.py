import unittest
from spec_generator.core.isotopes import IsotopeCalculator

class TestIsotopeCalculator(unittest.TestCase):

    def setUp(self):
        """Set up a new IsotopeCalculator instance for each test."""
        self.calculator = IsotopeCalculator()

    def test_get_distribution_zero_mass(self):
        """
        Test that a mass of 0 returns a single peak at offset 0 with intensity 1.0.
        """
        distribution, offset = self.calculator.get_distribution(0)
        self.assertEqual(distribution, [(0.0, 1.0)])
        self.assertEqual(offset, 0.0)

    def test_get_distribution_negative_mass(self):
        """
        Test that a negative mass behaves the same as a zero mass.
        """
        distribution, offset = self.calculator.get_distribution(-100)
        self.assertEqual(distribution, [(0.0, 1.0)])
        self.assertEqual(offset, 0.0)

    def test_get_distribution_for_large_mass(self):
        """
        Test that for a large mass, the distribution is plausible.
        - The offset should be positive.
        - The distribution should contain multiple peaks.
        - The most abundant peak should have an intensity of 1.0.
        """
        distribution, offset = self.calculator.get_distribution(25000)

        self.assertIsInstance(distribution, list)
        self.assertGreater(offset, 0, "Offset for a large mass should be positive.")
        self.assertGreater(len(distribution), 1, "Distribution for a large mass should have multiple peaks.")

        intensities = [p[1] for p in distribution]
        self.assertAlmostEqual(
            max(intensities), 1.0, places=5,
            msg="The most abundant peak should be normalized to 1.0."
        )

if __name__ == '__main__':
    unittest.main()
