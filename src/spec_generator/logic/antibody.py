"""
This module contains the logic for simulating antibody spectra, including handling
of different chain types (heavy, light) and generating combinatorial assemblies
for standard and bispecific antibodies.
"""
from itertools import combinations_with_replacement
from pyteomics import mass
from pyteomics.mass import Composition
import math
import queue
from dataclasses import asdict

from ..core.constants import DISULFIDE_MASS_LOSS
from .simulation import execute_simulation_and_write_mzml
from ..config import AntibodySimConfig, SpectrumGeneratorConfig
from .retention_time import KYTE_DOOLITTLE
from .ptm import calculate_ptm_mass_shift


def calculate_assembly_properties(chains: list[dict], assemblies: list[dict]) -> list[dict]:
    """
    Calculates the total mass and hydrophobicity for each antibody assembly.
    Mass calculation accounts for disulfide bonds based on cysteine content.
    Hydrophobicity is the sum of Kyte-Doolittle scores of component chains.

    Args:
        chains: List of dicts, each with 'name', 'type', and 'seq'.
        assemblies: List of assembly dicts from generate_assembly_combinations.

    Returns:
        The list of assembly dicts, with 'mass', 'bonds', and 'hydrophobicity' keys added.
    """
    chain_info = {}
    water_mass_loss = mass.calculate_mass(formula='H2O', average=True)

    for chain in chains:
        if chain['name'] not in chain_info:
            try:
                original_sequence = chain['seq'].upper()
                k_loss = chain.get('k_loss', False)
                pyro_glu = chain.get('pyro_glu', False)

                # Store the final sequence after modifications for hydrophobicity calculation
                final_sequence = original_sequence
                if k_loss and final_sequence.endswith('K'):
                    final_sequence = final_sequence[:-1]

                chain_mass = mass.calculate_mass(sequence=final_sequence, average=True)

                # Apply stochastic PTMs
                ptm_configs = chain.get('ptms', [])
                if ptm_configs:
                    ptm_mass_shift = calculate_ptm_mass_shift(final_sequence, ptm_configs)
                    chain_mass += ptm_mass_shift

                if pyro_glu and (original_sequence.startswith('E') or original_sequence.startswith('Q')):
                    chain_mass -= water_mass_loss

                hydrophobicity_score = sum(KYTE_DOOLITTLE.get(aa, 0) for aa in final_sequence)

                chain_info[chain['name']] = {
                    'mass': chain_mass,
                    'cys_count': final_sequence.count('C'),
                    'hydrophobicity': hydrophobicity_score,
                }
            except Exception as e:
                raise ValueError(f"Could not process chain {chain['name']}. Invalid sequence? Error: {e}")

    for assembly in assemblies:
        total_sequence_mass = 0
        total_cys_count = 0
        total_hydrophobicity = 0
        for component_name in assembly['components']:
            total_sequence_mass += chain_info[component_name]['mass']
            total_cys_count += chain_info[component_name]['cys_count']
            total_hydrophobicity += chain_info[component_name]['hydrophobicity']

        num_disulfide_bonds = math.floor(total_cys_count / 2)
        disulfide_correction = num_disulfide_bonds * DISULFIDE_MASS_LOSS

        assembly['mass'] = total_sequence_mass - disulfide_correction
        assembly['bonds'] = num_disulfide_bonds
        assembly['hydrophobicity'] = total_hydrophobicity

    return assemblies

def execute_antibody_simulation(
    config: AntibodySimConfig,
    final_filepath: str,
    progress_callback=None,
    return_data_only: bool = False
):
    """
    Orchestrates the full antibody simulation process by generating assemblies,
    calculating their properties, and feeding them into the core simulation engine.
    """
    try:
        chains_as_dicts = [asdict(c) for c in config.chains]

        assemblies = generate_assembly_combinations(chains_as_dicts)
        if not assemblies:
            raise ValueError("No valid antibody assemblies could be generated. Check chain definitions.")

        assemblies_with_properties = calculate_assembly_properties(chains_as_dicts, assemblies)

        ordered_assembly_names = [a['name'] for a in assemblies_with_properties]
        intensity_scalars = [config.assembly_abundances[name] for name in ordered_assembly_names]

        if len(intensity_scalars) != len(assemblies_with_properties):
            raise ValueError("The number of intensity scalars does not match the number of generated assemblies.")

        if progress_callback:
            num_assemblies = len(assemblies_with_properties)
            progress_callback('log', f"Generated {num_assemblies} unique species. Simulating combined spectrum...\n")
            for assembly in assemblies_with_properties[:15]:
                progress_callback('log', f"  - {assembly['name']} ({assembly['mass']:.2f} Da)\n")
            if num_assemblies > 15:
                progress_callback('log', f"  ... and {num_assemblies - 15} more species.\n")

        protein_masses = [a['mass'] for a in assemblies_with_properties]
        hydrophobicity_scores = [a['hydrophobicity'] for a in assemblies_with_properties]

        sim_config = SpectrumGeneratorConfig(
            common=config.common,
            lc=config.lc,
            protein_list_file=None,
            protein_masses=protein_masses,
            intensity_scalars=intensity_scalars,
            mass_inhomogeneity=0.0,
            hydrophobicity_scores=hydrophobicity_scores
        )

        result = execute_simulation_and_write_mzml(
            config=sim_config,
            final_filepath=final_filepath,
            progress_callback=progress_callback,
            return_data_only=return_data_only
        )

        return result

    except (ValueError, Exception) as e:
        if progress_callback:
            progress_callback('error', f"Antibody simulation failed: {e}")
        print(f"Antibody simulation failed: {e}")
        return False

def generate_assembly_combinations(chains: list[dict]) -> list[dict]:
    """
    Generates all plausible antibody assemblies from a list of input chains.
    """
    hcs = [c['name'] for c in chains if c['type'] == 'HC']
    lcs = [c['name'] for c in chains if c['type'] == 'LC']

    if not hcs or not lcs:
        return []

    assemblies = set()

    for hc in hcs:
        assemblies.add(tuple(sorted([hc])))
    for lc in lcs:
        assemblies.add(tuple(sorted([lc])))

    for hc in hcs:
        for lc in lcs:
            assemblies.add(tuple(sorted([hc, lc])))
    for h_pair in combinations_with_replacement(hcs, 2):
        assemblies.add(tuple(sorted(h_pair)))

    hh_dimers = list(combinations_with_replacement(hcs, 2))
    for hh in hh_dimers:
        for lc in lcs:
            assemblies.add(tuple(sorted(hh + (lc,))))

    ll_pairs = list(combinations_with_replacement(lcs, 2))
    for ll in ll_pairs:
        assemblies.add(tuple(sorted(ll)))

    for hh in hh_dimers:
        for ll in ll_pairs:
            assemblies.add(tuple(sorted(hh + ll)))

    result = []
    for assembly_tuple in sorted(list(assemblies), key=len):
        components = list(assembly_tuple)
        name = "".join(components)
        result.append({'name': name, 'components': components})

    return result
