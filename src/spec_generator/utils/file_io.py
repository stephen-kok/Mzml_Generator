import os
import csv
import re


def create_unique_filename(filepath: str) -> str:
    """
    Checks if a file exists and appends a counter if it does, ensuring a
    unique filename.
    """
    if not os.path.exists(filepath):
        return filepath

    base, ext = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{ext}"
        counter += 1
    return filepath


def format_filename(template: str, values: dict) -> str:
    """
    Formats a filename template string using a dictionary of values and
    sanitizes the result to be a valid filename.
    """
    # Find all placeholder keys in the template
    keys_in_template = re.findall(r'\{(.*?)\}', template)

    # Ensure all keys found in the template have a value to avoid KeyErrors
    for key in keys_in_template:
        values.setdefault(key, "") # Default to empty string if placeholder has no value

    formatted_name = template.format(**values)

    # Sanitize the filename to remove characters invalid for most file systems
    return re.sub(r'[\\/*?:"<>|]', "_", formatted_name)


def read_protein_list_file(filepath: str) -> list[tuple[float, float]]:
    """
    Reads a tab-delimited file containing protein masses and intensity scalars.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Protein list not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        # Use csv.reader for robust parsing of tab-delimited files
        reader = csv.reader(f, delimiter='\t')

        # Read and normalize header
        header = [h.strip().lower() for h in next(reader)]

        try:
            mass_idx = header.index("protein")
        except ValueError:
            raise ValueError("File must contain a 'Protein' column header for masses.")

        try:
            intensity_idx = header.index("intensity")
        except ValueError:
            raise ValueError("File must contain an 'Intensity' column header for scalars.")

        proteins = []
        # Start line count from 2 for user-friendly error reporting
        for i, row in enumerate(reader, 2):
            if not row:  # Skip empty lines
                continue
            try:
                mass = float(row[mass_idx])
                scalar = float(row[intensity_idx])
                if mass <= 0 or scalar < 0:
                    raise ValueError("Mass must be > 0, and Intensity scalar must be >= 0.")
                proteins.append((mass, scalar))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid data on line {i}: {e}")

    if not proteins:
        raise ValueError("Protein list file is empty or contains no valid data rows.")

    return proteins


def read_compound_list_file(filepath: str) -> list[tuple[str, float]]:
    """
    Reads a tab-delimited file containing compound names and mass deltas.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Compound list not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter='\t')
        header = [h.strip().lower() for h in next(reader)]

        try:
            name_idx = header.index("name")
        except ValueError:
            raise ValueError("File must contain a 'Name' column header for compound names.")

        try:
            delta_idx = header.index("delta")
        except ValueError:
            raise ValueError("File must contain a 'Delta' column header for mass differences.")

        compounds = []
        for i, row in enumerate(reader, 2):
            if not row:
                continue
            try:
                name = row[name_idx].strip()
                mass = float(row[delta_idx])
                if not name:
                    raise ValueError("Compound name cannot be empty.")
                compounds.append((name, mass))
            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid data on line {i}: {e}")

    if not compounds:
        raise ValueError("Compound list is empty or contains no valid data rows.")

    return compounds
