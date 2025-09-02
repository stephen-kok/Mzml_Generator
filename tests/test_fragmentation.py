import unittest
import numpy as np
from spec_generator.logic.fragmentation import generate_fragment_ions, FragmentIon
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

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)

        # Expected masses
        P, E, T, I, D = (
            AMINO_ACID_MASSES['P'], AMINO_ACID_MASSES['E'], AMINO_ACID_MASSES['T'],
            AMINO_ACID_MASSES['I'], AMINO_ACID_MASSES['D']
        )
        # Note: b-ion neutral mass is just the sum of residues
        b_ions_neutral = [
            P, P + E, P + E + P, P + E + P + T, P + E + P + T + I, P + E + P + T + I + D
        ]
        # Note: y-ion neutral mass includes an extra H2O
        y_ions_neutral = [
            E + H2O_MASS, D + E + H2O_MASS, I + D + E + H2O_MASS,
            T + I + D + E + H2O_MASS, P + T + I + D + E + H2O_MASS,
            E + P + T + I + D + E + H2O_MASS,
        ]

        expected_mzs = set()
        for mass in b_ions_neutral + y_ions_neutral:
            for charge in charges:
                expected_mzs.add(round((mass + charge * PROTON_MASS) / charge, 4))

        # Calculate m/z from results
        result_mzs = set()
        for frag in result_fragments:
            self.assertIsInstance(frag, FragmentIon)
            mz = (frag.neutral_mass + frag.charge * PROTON_MASS) / frag.charge
            result_mzs.add(round(mz, 4))

        self.assertTrue(expected_mzs.issubset(result_mzs))

    def test_intensity_model_proline(self):
        """Test that cleavage C-terminal to Proline is enhanced."""
        sequence = "TESTPEPTIDE"
        ion_types = ["b"]
        charges = [1]

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)

        # Find the b4 ('TEST') and b5 ('TESTP') ions
        b4_ion = next((f for f in result_fragments if f.sequence == "TEST"), None)
        b5_ion = next((f for f in result_fragments if f.sequence == "TESTP"), None)

        self.assertIsNotNone(b4_ion, "b4 ion should be present")
        self.assertIsNotNone(b5_ion, "b5 ion should be present")

        self.assertGreater(b5_ion.intensity, b4_ion.intensity,
                         "Intensity of b-ion after Proline should be enhanced.")
        self.assertAlmostEqual(b5_ion.intensity / b4_ion.intensity, 5.0,
                             places=1, msg="Proline enhancement factor should be 5.0")

    def test_a_and_x_ions(self):
        """Test generation of a and x ions."""
        sequence = "GAV"
        ion_types = ["a", "x"]
        charges = [1]

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)

        G, A, V = AMINO_ACID_MASSES['G'], AMINO_ACID_MASSES['A'], AMINO_ACID_MASSES['V']

        # a-ion = b-ion - CO
        # x-ion = y-ion + CO - H2O
        expected_masses = {
            'a': [G - CO_MASS, G + A - CO_MASS],
            'x': [(V + H2O_MASS) + CO_MASS - H2O_MASS, (A + V + H2O_MASS) + CO_MASS - H2O_MASS]
        }

        for frag in result_fragments:
            self.assertIn(frag.ion_type, expected_masses)
            mass_list = expected_masses[frag.ion_type]
            self.assertTrue(any(np.isclose(frag.neutral_mass, m) for m in mass_list))

    def test_neutral_loss(self):
        """Test generation of ions with neutral losses."""
        sequence = "TEST"
        ion_types = ["b"]
        charges = [1]

        result_fragments = generate_fragment_ions(sequence, ion_types, charges)

        # We expect b-ions for "T", "TE", "TES", "TEST"
        # For "TES", which contains S and T, we expect a neutral loss variant.
        b3_ion = next((f for f in result_fragments if f.sequence == "TES" and f.neutral_loss is None), None)
        b3_loss_ion = next((f for f in result_fragments if f.sequence == "TES" and f.neutral_loss == "H2O"), None)

        self.assertIsNotNone(b3_ion, "b3 primary ion should be present")
        self.assertIsNotNone(b3_loss_ion, "b3 neutral loss ion should be present")

        # Neutral loss ion should be less intense
        self.assertLess(b3_loss_ion.intensity, b3_ion.intensity)
        self.assertAlmostEqual(b3_loss_ion.intensity / b3_ion.intensity, 0.2, places=1)

    def test_ptm_fragmentation(self):
        """Test that PTMs are correctly handled in fragment masses."""
        sequence = "PEPTIDE"
        ion_types = ["b", "y"]
        charges = [1]

        phos_mass = 79.966331
        ptms = {3: phos_mass}  # Phosphorylation of T at index 3

        result_fragments = generate_fragment_ions(sequence, ion_types, charges, ptms=ptms)

        P, E, T, I, D = (
            AMINO_ACID_MASSES['P'], AMINO_ACID_MASSES['E'], AMINO_ACID_MASSES['T'],
            AMINO_ACID_MASSES['I'], AMINO_ACID_MASSES['D']
        )

        b3_mass = P + E + P
        b4_mass_phos = P + E + P + T + phos_mass
        y3_mass = I + D + E + H2O_MASS
        y4_mass_phos = T + I + D + E + H2O_MASS + phos_mass

        expected_masses = {b3_mass, b4_mass_phos, y3_mass, y4_mass_phos}

        # Check if the key fragments have the correct masses
        found_masses = {round(f.neutral_mass, 4) for f in result_fragments if round(f.neutral_mass, 4) in {round(m, 4) for m in expected_masses}}

        self.assertTrue(len(found_masses) >= 4)


if __name__ == "__main__":
    unittest.main()
