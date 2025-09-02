import unittest
import numpy as np
from spec_generator.logic.fragmentation import generate_fragment_ions
from spec_generator.core.constants import (
    AMINO_ACID_MASSES,
    PROTON_MASS,
    H2O_MASS,
    CO_MASS,
    NH3_MASS,
)


class TestFragmentation(unittest.TestCase):
    def test_generate_fragment_ions_base(self):
        """
        Test generation of basic b and y ions for a simple peptide.
        This test checks that the core logic remains correct after refactoring.
        """
        sequence = "PEPTIDE"
        ion_types = ["b", "y"]
        charges = [1, 2]

        # Use constants to avoid magic numbers
        P, E, T, I, D = (
            AMINO_ACID_MASSES['P'], AMINO_ACID_MASSES['E'], AMINO_ACID_MASSES['T'],
            AMINO_ACID_MASSES['I'], AMINO_ACID_MASSES['D']
        )

        b_ions_neutral = [
            P,
            P + E,
            P + E + P,
            P + E + P + T,
            P + E + P + T + I,
            P + E + P + T + I + D,
        ]
        y_ions_neutral = [
            E + H2O_MASS,
            D + E + H2O_MASS,
            I + D + E + H2O_MASS,
            T + I + D + E + H2O_MASS,
            P + T + I + D + E + H2O_MASS,
            E + P + T + I + D + E + H2O_MASS,
        ]
        expected_mzs = []
        for mass in b_ions_neutral + y_ions_neutral:
            for charge in charges:
                expected_mzs.append((mass + charge * PROTON_MASS) / charge)

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)
        result_mzs = [item[0] for item in result_fragments]

        expected_set = {round(mz, 4) for mz in expected_mzs}
        result_set = {round(mz, 4) for mz in result_mzs}

        # Check that all expected base ions are present (neutral loss ions may also be present)
        self.assertTrue(expected_set.issubset(result_set),
                        f"Missing ions: {expected_set - result_set}")

    def test_intensity_model_proline(self):
        """Test that cleavage C-terminal to Proline is enhanced."""
        sequence = "TESTPEPTIDE"
        ion_types = ["b"]
        charges = [1]

        # b-ion series: T, TE, TES, TEST, TESTP, ...
        # The b5 ion is 'TESTP'. Cleavage is after P.
        # Its intensity should be higher than the b4 ion 'TEST'.
        P, E, S, T = (
            AMINO_ACID_MASSES['P'], AMINO_ACID_MASSES['E'],
            AMINO_ACID_MASSES['S'], AMINO_ACID_MASSES['T']
        )
        b4_mass = T + E + S + T
        b5_mass = T + E + S + T + P

        b4_mz = (b4_mass + PROTON_MASS) / 1
        b5_mz = (b5_mass + PROTON_MASS) / 1

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)
        result_dict = {round(item[0], 4): item[1] for item in result_fragments}

        b4_intensity = result_dict.get(round(b4_mz, 4), 0)
        b5_intensity = result_dict.get(round(b5_mz, 4), 0)

        self.assertGreater(b5_intensity, b4_intensity,
                         "Intensity of b-ion after Proline should be enhanced.")
        self.assertAlmostEqual(b5_intensity / b4_intensity, 5.0,
                             "Proline enhancement factor should be 5.0")

    def test_a_and_x_ions(self):
        """Test generation of a and x ions."""
        sequence = "GAV"
        ion_types = ["a", "x"]
        charges = [1]

        G, A, V = AMINO_ACID_MASSES['G'], AMINO_ACID_MASSES['A'], AMINO_ACID_MASSES['V']

        a_masses = [G - CO_MASS, G + A - CO_MASS]
        x_masses = [V + CO_MASS, A + V + CO_MASS]

        expected_mzs = []
        for mass in a_masses + x_masses:
            expected_mzs.append((mass + PROTON_MASS) / 1)

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)
        result_mzs = [item[0] for item in result_fragments]
        np.testing.assert_allclose(sorted(result_mzs), sorted(expected_mzs), rtol=1e-5)

    def test_neutral_loss(self):
        """Test generation of ions with neutral losses."""
        sequence = "TEST"
        ion_types = ["b"]
        charges = [1]

        T, E, S = AMINO_ACID_MASSES['T'], AMINO_ACID_MASSES['E'], AMINO_ACID_MASSES['S']

        b2_mass = T + E
        b3_mass = T + E + S

        # E, S, T can all lose H2O.
        # Expected ions: b2, b2-H2O, b3, b3-H2O
        expected_masses = [
            b2_mass,
            b2_mass - H2O_MASS,
            b3_mass,
            b3_mass - H2O_MASS,
        ]

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)
        result_mzs = [item[0] for item in result_fragments]

        expected_mzs = [(m + PROTON_MASS) / 1 for m in expected_masses]
        result_set = {round(mz, 4) for mz in result_mzs}
        expected_set = {round(mz, 4) for mz in expected_mzs}

        self.assertTrue(expected_set.issubset(result_set))

    def test_ptm_fragmentation(self):
        """Test that PTMs are correctly handled in fragment masses."""
        sequence = "PEPTIDE"
        ion_types = ["b", "y"]
        charges = [1]

        P, E, T, I, D = (
            AMINO_ACID_MASSES['P'], AMINO_ACID_MASSES['E'], AMINO_ACID_MASSES['T'],
            AMINO_ACID_MASSES['I'], AMINO_ACID_MASSES['D']
        )
        phos_mass = 79.966331
        ptms = {3: phos_mass}  # Phosphorylation of T at index 3

        b3_mass = P + E + P
        b4_mass_phos = P + E + P + T + phos_mass

        y3_mass = I + D + E + H2O_MASS
        y4_mass_phos = T + I + D + E + H2O_MASS + phos_mass

        expected_masses = [b3_mass, b4_mass_phos, y3_mass, y4_mass_phos]
        expected_mzs = [(m + PROTON_MASS) / 1 for m in expected_masses]

        result_fragments = generate_fragment_ions(sequence, ion_types, charges, ptms=ptms)
        result_mzs = [item[0] for item in result_fragments]

        result_set = {round(mz, 4) for mz in result_mzs}
        expected_set = {round(mz, 4) for mz in expected_mzs}

        self.assertTrue(expected_set.issubset(result_set),
                        f"Missing PTM ions: {expected_set - result_set}")


if __name__ == "__main__":
    unittest.main()
