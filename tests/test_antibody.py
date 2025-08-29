import unittest
from dataclasses import asdict

from spec_generator.logic.antibody import (
    generate_assembly_combinations,
    calculate_assembly_masses,
)
from spec_generator.config import Chain
from spec_generator.core.constants import DISULFIDE_MASS_LOSS

class TestAntibodyLogic(unittest.TestCase):
    def setUp(self):
        self.chains = [
            Chain(type='HC', name='H1', seq='PEPTIDE'),
            Chain(type='LC', name='L1', seq='SEQUENCE'),
        ]
        self.chains_with_cys = [
            Chain(type='HC', name='H_cys', seq='PEPTIDEC'),
            Chain(type='LC', name='L_cys', seq='SEQUENCEC'),
        ]

    def test_generate_assembly_combinations(self):
        chains_as_dicts = [asdict(c) for c in self.chains]
        assemblies = generate_assembly_combinations(chains_as_dicts)
        self.assertIsInstance(assemblies, list)
        self.assertGreater(len(assemblies), 0)

        # Check for a known assembly (order of components is sorted)
        full_antibody = {'name': 'H1H1L1L1', 'components': ['H1', 'H1', 'L1', 'L1']}
        # Find the assembly in the results list
        found = any(a['name'] == full_antibody['name'] and sorted(a['components']) == sorted(full_antibody['components']) for a in assemblies)
        self.assertTrue(found, "Standard H2L2 antibody not found in assemblies.")

    def test_calculate_assembly_masses(self):
        chains_as_dicts = [asdict(c) for c in self.chains]
        assemblies = [{'name': 'H1L1', 'components': ['H1', 'L1']}]
        assemblies_with_mass = calculate_assembly_masses(chains_as_dicts, assemblies)

        self.assertIn('mass', assemblies_with_mass[0])
        self.assertIn('bonds', assemblies_with_mass[0])
        self.assertGreater(assemblies_with_mass[0]['mass'], 0)
        self.assertEqual(assemblies_with_mass[0]['bonds'], 0)

    def test_disulfide_bond_mass_loss(self):
        chains_as_dicts = [asdict(c) for c in self.chains_with_cys]
        assemblies = [{'name': 'H_cysL_cys', 'components': ['H_cys', 'L_cys']}]
        assemblies_with_mass = calculate_assembly_masses(chains_as_dicts, assemblies)

        # 1 disulfide bond = 2 cysteines
        self.assertEqual(assemblies_with_mass[0]['bonds'], 1)

        # Calculate expected mass to verify the loss
        from pyteomics import mass
        mass_hc = mass.calculate_mass(sequence='PEPTIDEC', average=True)
        mass_lc = mass.calculate_mass(sequence='SEQUENCEC', average=True)
        expected_mass = mass_hc + mass_lc - DISULFIDE_MASS_LOSS

        self.assertAlmostEqual(assemblies_with_mass[0]['mass'], expected_mass, places=2)

    def test_invalid_sequence_raises_error(self):
        chains_invalid = [asdict(Chain(type='HC', name='H_invalid', seq='INVALID_SEQUENCE'))]
        assemblies = [{'name': 'H_invalid', 'components': ['H_invalid']}]
        with self.assertRaisesRegex(ValueError, "Could not process chain H_invalid"):
            calculate_assembly_masses(chains_invalid, assemblies)

if __name__ == '__main__':
    unittest.main()
