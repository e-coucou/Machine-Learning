import numpy as np
import pandas as pd
import pytest

from nop_analysis.utils import (
    ensure_columns,
    ensure_datetime_index,
    rolling_sum,
    to_business_unit,
    zscore,
)


class TestEnsureDatetimeIndex:
    def test_passes_with_datetime_index(self):
        df = pd.DataFrame({"a": [1, 2]}, index=pd.date_range("2026-01-01", periods=2))
        ensure_datetime_index(df)  # ne doit pas lever

    def test_raises_with_range_index(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(TypeError):
            ensure_datetime_index(df)


class TestEnsureColumns:
    def test_passes_when_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        ensure_columns(df, ["a", "b"])  # ne doit pas lever

    def test_raises_when_column_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError):
            ensure_columns(df, ["a", "b"])


class TestRollingSum:
    def test_matches_manual_convolution(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_sum(values, 2)
        np.testing.assert_allclose(result, [3.0, 5.0, 7.0, 9.0])

    def test_returns_empty_when_window_too_large(self):
        values = np.array([1.0, 2.0])
        result = rolling_sum(values, 5)
        assert result.size == 0

    def test_raises_on_non_positive_window(self):
        with pytest.raises(ValueError):
            rolling_sum(np.array([1.0, 2.0]), 0)


class TestZscore:
    def test_mean_zero_std_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zscore(values)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10

    def test_constant_array_returns_zeros(self):
        values = np.array([5.0, 5.0, 5.0])
        z = zscore(values)
        np.testing.assert_array_equal(z, [0.0, 0.0, 0.0])


class TestToBusinessUnit:
    def test_multiplies_by_factor(self):
        assert to_business_unit(10, 2.71) == pytest.approx(27.1)
