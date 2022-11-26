import pandas as pd

import feature_transforms


def test_normalized_name():
    """Tests the feature_transforms.normalized_name function."""
    input_series = pd.Series(['Foo, Sir.', 'Matilde, the Countess.', 'Devin, Monsignor.'])
    expected = pd.Series(['Sir', 'the Countess', 'Monsignor'])
    actual = feature_transforms.normalized_name(input_series)
    pd.testing.assert_series_equal(actual, expected)
