import unittest
import os
import tempfile
import shutil

from spec_generator.utils.file_io import (
    create_unique_filename,
    format_filename,
    read_protein_list_file,
    read_compound_list_file,
)

class TestFileIO(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_unique_filename_new_file(self):
        filepath = os.path.join(self.test_dir, "test.txt")
        self.assertEqual(create_unique_filename(filepath), filepath)

    def test_create_unique_filename_existing_file(self):
        filepath = os.path.join(self.test_dir, "test.txt")
        open(filepath, "w").close()
        self.assertEqual(create_unique_filename(filepath), os.path.join(self.test_dir, "test_1.txt"))

    def test_create_unique_filename_multiple_existing(self):
        filepath = os.path.join(self.test_dir, "test.txt")
        open(filepath, "w").close()
        open(os.path.join(self.test_dir, "test_1.txt"), "w").close()
        self.assertEqual(create_unique_filename(filepath), os.path.join(self.test_dir, "test_2.txt"))

    def test_format_filename_basic(self):
        template = "{date}_{protein_mass}.mzML"
        values = {"date": "2023-10-27", "protein_mass": 12345}
        self.assertEqual(format_filename(template, values), "2023-10-27_12345.mzML")

    def test_format_filename_sanitization(self):
        template = "{name}|{date}.txt"
        values = {"name": "invalid<name>", "date": "2023/10/27"}
        self.assertEqual(format_filename(template, values), "invalid_name__2023_10_27.txt")

    def test_format_filename_missing_key(self):
        template = "{date}_{time}.log"
        values = {"date": "2023-10-27"}
        self.assertEqual(format_filename(template, values), "2023-10-27_.log")

    def test_read_protein_list_file_valid(self):
        content = "Protein\tIntensity\n25000.0\t1.0\n50000.0\t0.5"
        filepath = os.path.join(self.test_dir, "proteins.txt")
        with open(filepath, "w") as f:
            f.write(content)
        result = read_protein_list_file(filepath)
        self.assertEqual(result, [(25000.0, 1.0), (50000.0, 0.5)])

    def test_read_protein_list_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_protein_list_file("non_existent_file.txt")

    def test_read_protein_list_file_bad_header(self):
        content = "Mass\tIntensity\n25000.0\t1.0"
        filepath = os.path.join(self.test_dir, "proteins.txt")
        with open(filepath, "w") as f:
            f.write(content)
        with self.assertRaisesRegex(ValueError, "must contain a 'Protein' column"):
            read_protein_list_file(filepath)

    def test_read_protein_list_file_bad_data(self):
        content = "Protein\tIntensity\n25000.0\tnot_a_float"
        filepath = os.path.join(self.test_dir, "proteins.txt")
        with open(filepath, "w") as f:
            f.write(content)
        with self.assertRaisesRegex(ValueError, "Invalid data on line 2"):
            read_protein_list_file(filepath)

    def test_read_compound_list_file_valid(self):
        content = "Name\tDelta\nCompA\t128.1\nCompB\t256.2"
        filepath = os.path.join(self.test_dir, "compounds.txt")
        with open(filepath, "w") as f:
            f.write(content)
        result = read_compound_list_file(filepath)
        self.assertEqual(result, [("CompA", 128.1), ("CompB", 256.2)])

    def test_read_compound_list_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_compound_list_file("non_existent_file.txt")

    def test_read_compound_list_file_bad_header(self):
        content = "Compound\tMass\nCompA\t128.1"
        filepath = os.path.join(self.test_dir, "compounds.txt")
        with open(filepath, "w") as f:
            f.write(content)
        with self.assertRaisesRegex(ValueError, "must contain a 'Name' column"):
            read_compound_list_file(filepath)

if __name__ == '__main__':
    unittest.main()
