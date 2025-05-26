import unittest
import pandas as pd
from models.landmark.dataset.prepare_data import encode_label, train_test_split


class TestLabelEncodingAndSplit(unittest.TestCase):
    def setUp(self):
        # Sample dataframe to use in tests
        self.df = pd.DataFrame(
            {
                "label": [
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "cat",
                    "dog",
                    "dog",
                    "dog",
                    "dog",
                    "dog",
                    "dog",
                    "bird",
                    "bird",
                    "bird",
                    "bird",
                    "bird",
                    "bird",
                ],
                "data_source": [
                    "video1",
                    "video2",
                    "video3",
                    "video4",
                    "video5",
                    "video6",
                    "video1",
                    "video2",
                    "video3",
                    "video4",
                    "video5",
                    "video6",
                    "video1",
                    "video2",
                    "video3",
                    "video4",
                    "video5",
                    "video6",
                ],
            }
        )

    def test_encode_label(self):
        df_encoded = encode_label(self.df.copy())

        # Check if 'label_encoded' column exists
        self.assertIn("label_encoded", df_encoded.columns)

        # Check if the mapping is correct
        unique_labels = set(self.df["label"])
        unique_encoded = set(df_encoded["label_encoded"])
        self.assertEqual(len(unique_labels), len(unique_encoded))

        # Check if same labels have same encoded value
        cat_encoding = df_encoded[df_encoded["label"] == "cat"]["label_encoded"].iloc[0]
        self.assertTrue(
            (
                df_encoded[df_encoded["label"] == "cat"]["label_encoded"]
                == cat_encoding
            ).all()
        )

    def test_train_test_split(self):
        df_encoded = encode_label(self.df.copy())
        df_split = train_test_split(df_encoded)

        # Check if 'dataset_split' column exists
        self.assertIn("dataset_split", df_split.columns)

        # Check if all rows got assigned to a split
        self.assertFalse(df_split["dataset_split"].isnull().any())

        # Check that there's exactly 1 "test" sample per label
        for label in df_split["label"].unique():
            label_group = df_split[df_split["label"] == label]
            test_samples = label_group[label_group["dataset_split"] == "test"]
            self.assertEqual(len(test_samples), 1)

        # Optional: check that no group assignment messed up the total length
        self.assertEqual(len(df_split), len(self.df))
