try:
    from pyteomics.mass import mass as mass_mod
    print("Successfully imported pyteomics.mass.mass")

    # Now, let's see if the function is there
    if hasattr(mass_mod, 'isotopic_distribution'):
        print("\nSUCCESS: Found 'isotopic_distribution' in pyteomics.mass.mass")
        iso_dist_func = getattr(mass_mod, 'isotopic_distribution')
        print(iso_dist_func)
    else:
        print("\nFAILURE: 'isotopic_distribution' NOT FOUND in pyteomics.mass.mass")
        print("\nAvailable functions/attributes in pyteomics.mass.mass:")
        # Print all attributes to see what's available
        for attr in dir(mass_mod):
            if not attr.startswith('_'):
                print(attr)

except ImportError as e:
    print(f"Failed to import pyteomics.mass.mass: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
