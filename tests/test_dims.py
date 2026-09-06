"""Normalization of flat, nested, and inferred subsystem dimensions."""

import pytest

from quchip.backend._dims import normalize_dims_from_list


@pytest.mark.parametrize(
    "dims,fallback,expected",
    [
        (None, 3, (3,)),
        ([[2, 3], [2, 3]], None, (2, 3)),
        ([[2, 3]], None, (2, 3)),
        ([2, 3, 4], None, (2, 3, 4)),
    ],
)
def test_normalize_supported_dimension_forms(dims, fallback, expected):
    """Flat and nested layouts retain subsystem order; missing dims use the fallback."""
    assert normalize_dims_from_list(dims, fallback=fallback) == expected


def test_missing_dims_without_fallback_raises():
    """Dimensions cannot be inferred without a fallback."""
    with pytest.raises(ValueError):
        normalize_dims_from_list(None, fallback=None)
