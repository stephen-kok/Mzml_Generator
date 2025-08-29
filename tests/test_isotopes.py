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

    def test_distribution_most_abundant_peak(self):
        """
        Test that for a simple case, the most abundant peak is the first one (offset 0).
        For a very small mass, the probability of having even one neutron is very low.
        """
        # A mass so small that the +1 isotope should be negligible
        distribution, offset = self.calculator.get_distribution(10)

        # The most abundant peak should be the first one (monoisotopic)
        self.assertAlmostEqual(offset, 0.0, places=5)

        # The distribution should start with the highest intensity peak (1.0)
        self.assertEqual(distribution[0][1], 1.0)

        # The second peak should have a very low intensity
        self.assertLess(distribution[1][1], 0.01)

    def test_distribution_returns_correct_format(self):
        """
        Test that the returned distribution is a list of tuples (float, float)
        and the offset is a float.
        """
        distribution, offset = self.calculator.get_distribution(25000)

        self.assertIsInstance(distribution, list)
        self.assertTrue(len(distribution) > 1)
        self.assertIsInstance(distribution[0], tuple)
        self.assertIsInstance(distribution[0][0], float)
        self.assertIsInstance(distribution[0][1], float)

        self.assertIsInstance(offset, float)

if __name__ == '__main__':
    unittest.main()
