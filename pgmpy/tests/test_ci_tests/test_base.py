import unittest

from skbase.lookup import all_objects

from pgmpy.ci_tests import _BaseCITest


class TestCIRegistry(unittest.TestCase):
    def test_ci_registry(self):
        all_tests = [
            ci_test.get_class_tag("name")
            for ci_test in all_objects(
                object_types=_BaseCITest,
                package_name="pgmpy.ci_tests",
                return_names=False,
            )
        ]

        self.assertIn("chi_square", all_tests)
        self.assertIn("g_sq", all_tests)
        self.assertIn("log_likelihood", all_tests)
        self.assertIn("modified_log_likelihood", all_tests)
        self.assertIn("pearsonr", all_tests)
        self.assertIn("pillai", all_tests)
        self.assertIn("gcm", all_tests)
