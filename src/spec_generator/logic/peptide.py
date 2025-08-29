"""
This module provides functions for handling peptide-specific logic,
starting with in-silico digestion of protein sequences.
"""
def digest_sequence(sequence: str, enzyme: str = 'trypsin', missed_cleavages: int = 0) -> list[str]:
    """
    Performs in-silico digestion of a protein sequence using a specified enzyme
    and allows for a certain number of missed cleavages.

    Args:
        sequence: The amino acid sequence of the protein.
        enzyme: The enzyme to use for digestion (currently, only 'trypsin' is supported).
        missed_cleavages: The maximum number of missed cleavages to allow.

    Returns:
        A list of unique peptide sequences generated from the digestion.
    """
    if not sequence:
        return []

    if enzyme != 'trypsin':
        raise NotImplementedError(f"Enzyme '{enzyme}' is not supported.")

    cleavage_sites = [0]
    for i in range(len(sequence) - 1):
        if sequence[i] in 'KR' and sequence[i+1] != 'P':
            cleavage_sites.append(i + 1)
    cleavage_sites.append(len(sequence))

    # Remove duplicates
    cleavage_sites = sorted(list(set(cleavage_sites)))

    fragments = []
    for i in range(len(cleavage_sites) - 1):
        fragments.append(sequence[cleavage_sites[i]:cleavage_sites[i+1]])

    peptides = set()
    for i in range(len(fragments)):
        for j in range(missed_cleavages + 1):
            if i + j < len(fragments):
                peptides.add("".join(fragments[i:i + j + 1]))

    return sorted(list(peptides))
