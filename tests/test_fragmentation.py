import unittest
import numpy as np
from spec_generator.logic.fragmentation import generate_fragment_ions
from spec_generator.core.constants import AMINO_ACID_MASSES, PROTON_MASS, H2O_MASS

class TestFragmentation(unittest.TestCase):
    def test_generate_fragment_ions(self):
        """
        Test generation of fragment ions for a simple peptide.
        """
        sequence = "PEPTIDE"
        ion_types = ['b', 'y']
        charges = [1, 2]

        # Expected masses for PEPTIDE
        # P = 97.052764
        # E = 129.042593
        # P = 97.052764
        # T = 101.047679
        # I = 113.084064
        # D = 115.026943
        # E = 129.042593

        # Neutral masses of b ions
        b_ions_neutral = [
            97.052764,  # b1
            97.052764 + 129.042593,  # b2
            97.052764 + 129.042593 + 97.052764,  # b3
            97.052764 + 129.042593 + 97.052764 + 101.047679,  # b4
            97.052764 + 129.042593 + 97.052764 + 101.047679 + 113.084064,  # b5
            97.052764 + 129.042593 + 97.052764 + 101.047679 + 113.084064 + 115.026943,  # b6
        ]

        # Neutral masses of y ions
        y_ions_neutral = [
            129.042593 + H2O_MASS,  # y1
            129.042593 + 115.026943 + H2O_MASS,  # y2
            129.042593 + 115.026943 + 113.084064 + H2O_MASS,  # y3
            129.042593 + 115.026943 + 113.084064 + 101.047679 + H2O_MASS,  # y4
            129.042593 + 115.026943 + 113.084064 + 101.047679 + 97.052764 + H2O_MASS,  # y5
            129.042593 + 115.026943 + 113.084064 + 101.047679 + 97.052764 + 129.042593 + H2O_MASS,  # y6
        ]

        expected_mzs = []
        for mass in b_ions_neutral + y_ions_neutral:
            for charge in charges:
                expected_mzs.append((mass + charge * PROTON_MASS) / charge)

        result_mzs = generate_fragment_ions(sequence, ion_types, charges)

        np.testing.assert_allclose(sorted(result_mzs), sorted(expected_mzs), rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
