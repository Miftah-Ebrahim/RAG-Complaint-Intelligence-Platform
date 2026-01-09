import unittest
import pandas as pd
from src.data_processing import stratified_sample


class TestSampling(unittest.TestCase):
    def test_stratified_sample(self):
        # Create dummy dataframe
        data = {"Product": ["A"] * 10 + ["B"] * 5, "Value": range(15)}
        df = pd.DataFrame(data)

        # Test 1: Sample size < class size
        result = stratified_sample(df, n_per_class=3)
        counts = result["Product"].value_counts()
        self.assertEqual(counts["A"], 3)
        self.assertEqual(counts["B"], 3)

        # Test 2: Sample size > class size (should take all)
        result = stratified_sample(df, n_per_class=8)
        counts = result["Product"].value_counts()
        self.assertEqual(counts["A"], 8)
        self.assertEqual(counts["B"], 5)  # Should be capped at 5


if __name__ == "__main__":
    unittest.main()
