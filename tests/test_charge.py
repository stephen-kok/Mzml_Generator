import unittest
from spec_generator.logic.charge import predict_charge_states

class TestChargePrediction(unittest.TestCase):

    def test_predict_charge_states_no_basic_residues(self):
        """
        Test with a peptide that has no basic residues (K, R, H).
        The primary charge should be 1.
        """
        peptide = "GAVLIMFWPSTCYNQE"
        distribution = predict_charge_states(peptide)
        # Expects {1: 1.0, 2: 0.22...} -> filters to {1: 1.0}
        self.assertIn(1, distribution)
        self.assertAlmostEqual(distribution[1], 1.0, places=2)
        # Charge 2 should have a lower intensity
        if 2 in distribution:
            self.assertLess(distribution[2], distribution[1])

    def test_predict_charge_states_one_basic_residue(self):
        """
        Test with a peptide containing one basic residue (K).
        The primary charge should be 2.
        """
        peptide = "GAVKIMFWPSTCYNQE"
        distribution = predict_charge_states(peptide)
        # Expects charges around 2, e.g., {1: 0.22, 2: 1.0, 3: 0.22}
        self.assertIn(2, distribution)
        self.assertAlmostEqual(distribution[2], 1.0, places=2)
        self.assertIn(1, distribution)
        self.assertIn(3, distribution)
        self.assertLess(distribution[1], distribution[2])
        self.assertLess(distribution[3], distribution[2])

    def test_predict_charge_states_multiple_basic_residues(self):
        """
        Test with a peptide containing multiple basic residues (K, R, H).
        The primary charge should be 4.
        """
        peptide = "GAVKIMFWPSTHRYNQE" # K, H, R = 3 basic
        distribution = predict_charge_states(peptide)
        # Expects charges around 4, e.g., {3: 0.22, 4: 1.0, 5: 0.22}
        self.assertIn(4, distribution)
        self.assertAlmostEqual(distribution[4], 1.0, places=2)
        self.assertIn(3, distribution)
        self.assertIn(5, distribution)
        self.assertLess(distribution[3], distribution[4])
        self.assertLess(distribution[5], distribution[4])

    def test_predict_charge_states_empty_sequence(self):
        """
        Test with an empty peptide sequence.
        Should return a default value.
        """
        peptide = ""
        distribution = predict_charge_states(peptide)
        self.assertEqual(distribution, {2: 1.0})

    def test_predict_charge_states_short_sequence(self):
        """
        Test with a very short peptide sequence.
        Should return a default value.
        """
        peptide = "GK"
        distribution = predict_charge_states(peptide)
        self.assertEqual(distribution, {2: 1.0})

if __name__ == '__main__':
    unittest.main()
