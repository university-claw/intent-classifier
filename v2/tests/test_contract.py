import unittest

from serving.contract import (
    ABUSE_LABEL,
    NON_ABUSE_LABEL,
    clamp_top_k,
    final_from_p_abuse,
    validate_label_map,
)


class SafetyContractTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertEqual(final_from_p_abuse(0.25, 0.25), ABUSE_LABEL)

    def test_probability_below_threshold_is_non_abuse(self):
        self.assertEqual(final_from_p_abuse(0.249, 0.25), NON_ABUSE_LABEL)

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            final_from_p_abuse(1.01, 0.25)

    def test_label_map_requires_binary_safety_labels(self):
        validate_label_map({"non_abuse": 0, "abuse": 1})
        with self.assertRaises(ValueError):
            validate_label_map({"chat": 0, "abuse": 1})

    def test_top_k_is_clamped_to_label_count(self):
        self.assertEqual(clamp_top_k(3, 2), 2)


if __name__ == "__main__":
    unittest.main()
