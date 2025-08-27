"""
This module contains the logic for simulating antibody spectra, including handling
of different chain types (heavy, light) and generating combinatorial assemblies
for standard and bispecific antibodies.
"""
from itertools import combinations_with_replacement
from pyteomics import mass

import math
from ..core.constants import DISULFIDE_MASS_LOSS

def calculate_assembly_masses(chains: list[dict], assemblies: list[dict]) -> list[dict]:
    """
    Calculates the total mass for each antibody assembly, accounting for
    mass loss from all possible disulfide bonds based on cysteine content.

    Args:
        chains: List of dicts, each with 'name', 'type', and 'seq'.
        assemblies: List of assembly dicts from generate_assembly_combinations.

    Returns:
        The list of assembly dicts, with 'mass' and 'bonds' keys added.
    """
    # 1. Calculate and cache info for each unique chain
    chain_info = {}
    for chain in chains:
        if chain['name'] not in chain_info:
            try:
                sequence = chain['seq'].upper()
                chain_info[chain['name']] = {
                    'mass': mass.calculate_mass(sequence=sequence, average=True),
                    'cys_count': sequence.count('C')
                }
            except Exception as e:
                raise ValueError(f"Could not process chain {chain['name']}. Invalid sequence? Error: {e}")

    # 2. Calculate mass for each assembly
    for assembly in assemblies:
        total_sequence_mass = 0
        total_cys_count = 0
        for component_name in assembly['components']:
            total_sequence_mass += chain_info[component_name]['mass']
            total_cys_count += chain_info[component_name]['cys_count']

        num_disulfide_bonds = math.floor(total_cys_count / 2)
        disulfide_correction = num_disulfide_bonds * DISULFIDE_MASS_LOSS

        assembly['mass'] = total_sequence_mass - disulfide_correction
        assembly['bonds'] = num_disulfide_bonds

    return assemblies

import queue
from .simulation import execute_simulation_and_write_mzml

def execute_antibody_simulation(
    chains: list[dict],
    final_filepath: str,
    update_queue: queue.Queue | None,
    # Simulation parameters:
    intensity_scalars: list[float],
    mz_step_str: str,
    peak_sigma_mz_str: str,
    mz_range_start_str: str,
    mz_range_end_str: str,
    noise_option: str,
    seed: int,
    lc_simulation_enabled: bool,
    num_scans: int,
    scan_interval: float,
    gaussian_std_dev: float,
    lc_tailing_factor: float,
    isotopic_enabled: bool,
    resolution: float,
    mass_inhomogeneity: float,
    pink_noise_enabled: bool,
    return_data_only: bool = False
):
    """
    Orchestrates the full antibody simulation process by generating assemblies,
    calculating their masses, and feeding them into the core simulation engine.
    """
    try:
        # Step 1: Generate assembly combinations
        assemblies = generate_assembly_combinations(chains)
        if not assemblies:
            raise ValueError("No valid antibody assemblies could be generated. Check chain definitions.")

        # Step 2: Calculate masses for all assemblies
        assemblies_with_mass = calculate_assembly_masses(chains, assemblies)

        protein_masses_str = ", ".join([str(a['mass']) for a in assemblies_with_mass])

        if len(intensity_scalars) != len(assemblies_with_mass):
            raise ValueError("The number of intensity scalars does not match the number of generated assemblies.")

        if update_queue:
            num_assemblies = len(assemblies_with_mass)
            update_queue.put(('log', f"Generated {num_assemblies} unique species. Simulating combined spectrum...\n"))
            for assembly in assemblies_with_mass[:15]:
                update_queue.put(('log', f"  - {assembly['name']} ({assembly['mass']:.2f} Da)\n"))
            if num_assemblies > 15:
                update_queue.put(('log', f"  ... and {num_assemblies - 15} more species.\n"))

        # Step 3: Call the core simulation engine
        result = execute_simulation_and_write_mzml(
            protein_masses_str=protein_masses_str,
            intensity_scalars=intensity_scalars,
            final_filepath=final_filepath,
            update_queue=update_queue,
            mz_step_str=mz_step_str,
            peak_sigma_mz_str=peak_sigma_mz_str,
            mz_range_start_str=mz_range_start_str,
            mz_range_end_str=mz_range_end_str,
            noise_option=noise_option,
            seed=seed,
            lc_simulation_enabled=lc_simulation_enabled,
            num_scans=num_scans,
            scan_interval=scan_interval,
            gaussian_std_dev=gaussian_std_dev,
            lc_tailing_factor=lc_tailing_factor,
            isotopic_enabled=isotopic_enabled,
            resolution=resolution,
            mass_inhomogeneity=mass_inhomogeneity,
            pink_noise_enabled=pink_noise_enabled,
            return_data_only=return_data_only
        )

        return result

    except (ValueError, Exception) as e:
        if update_queue:
            update_queue.put(('error', f"Antibody simulation failed: {e}"))
        print(f"Antibody simulation failed: {e}")
        return False

def generate_assembly_combinations(chains: list[dict]) -> list[dict]:
    """
    Generates all plausible antibody assemblies from a list of input chains.

    This function creates combinations for the most common species observed in
    antibody production, including free chains, half-antibodies, heavy-chain
    dimers, HHL impurities, and full H2L2 antibodies (including bispecifics).

    Args:
        chains: A list of dictionaries, where each dictionary represents a chain
                and has 'type' ('HC' or 'LC') and 'name' (e.g., 'H1') keys.

    Returns:
        A list of dictionaries, where each dictionary represents a unique
        assembly and has 'name' (e.g., 'H1H1L1L1') and 'components'
        (a list of chain names) keys.
    """
    hcs = [c['name'] for c in chains if c['type'] == 'HC']
    lcs = [c['name'] for c in chains if c['type'] == 'LC']

    if not hcs or not lcs:
        # Not a valid antibody if there's no heavy or light chain
        return []

    assemblies = set()

    # Use sorted tuples to represent assemblies, as this preserves duplicates
    # while still being hashable for storage in a set.

    # 1. Monomers (Free Chains)
    for hc in hcs:
        assemblies.add(tuple(sorted([hc])))
    for lc in lcs:
        assemblies.add(tuple(sorted([lc])))

    # 2. Dimers (Half-Antibodies and Heavy-Chain Dimers)
    # HL (Half-Antibody)
    for hc in hcs:
        for lc in lcs:
            assemblies.add(tuple(sorted([hc, lc])))
    # HH (Heavy-Chain Dimer)
    for h_pair in combinations_with_replacement(hcs, 2):
        assemblies.add(tuple(sorted(h_pair)))

    # 3. Trimers (HHL impurities)
    hh_dimers = list(combinations_with_replacement(hcs, 2))
    for hh in hh_dimers:
        for lc in lcs:
            assemblies.add(tuple(sorted(hh + (lc,))))

    # 4. Tetramers (Full Antibodies, H2L2) and LC-LC Dimers
    ll_pairs = list(combinations_with_replacement(lcs, 2))
    # Add LC-LC dimers to the assemblies
    for ll in ll_pairs:
        assemblies.add(tuple(sorted(ll)))

    for hh in hh_dimers:
        for ll in ll_pairs:
            assemblies.add(tuple(sorted(hh + ll)))

    # Format the output
    result = []
    for assembly_tuple in sorted(list(assemblies), key=len):
        components = list(assembly_tuple)
        name = "".join(components)
        result.append({'name': name, 'components': components})

    return result
