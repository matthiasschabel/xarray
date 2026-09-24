import warnings

import pytest

import xarray as xr
from xarray.core.coordinates import Coordinates
from xarray.structure.alignment import AlignmentError
from xarray.structure.merge import MergeError
from xarray.tests.indexes import ScalarIndex


@pytest.fixture(params=[None, "warn", "raise"])
def check_mode(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv("XARRAY_PROTOTYPE_MERGE_CHECK", raising=False)
    else:
        monkeypatch.setenv("XARRAY_PROTOTYPE_MERGE_CHECK", request.param)
    return request.param


def assert_indexed_value_wins(
    operation, check_mode, coordinate, value, error_type=MergeError
):
    message = f"conflicting values on objects to be combined for coordinate '{coordinate}'"
    if check_mode == "raise":
        with pytest.raises(error_type, match=message):
            operation()
    elif check_mode == "warn":
        with pytest.warns(FutureWarning, match=message):
            result = operation()
        assert result[coordinate].values.tolist() == value
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = operation()
        assert result[coordinate].values.tolist() == value


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("operation_name", ["add", "merge", "align"])
def test_scalar_index_conflict(check_mode, reverse, operation_name):
    indexed = xr.DataArray([1, 2], dims="x", coords={"tag": 10}, name="a").set_xindex(
        "tag", ScalarIndex
    )
    plain = xr.DataArray([3, 4], dims="x", coords={"tag": 20}, name="b")
    left, right = (plain, indexed) if reverse else (indexed, plain)

    def operation():
        if operation_name == "merge":
            return xr.merge([left, right])
        if operation_name == "align":
            return xr.align(left, right, join="exact")[1]
        return left + right

    assert_indexed_value_wins(
        operation, check_mode, "tag", 10, error_type=AlignmentError
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_pandas_dimension_index_conflict(check_mode, reverse):
    indexed = xr.Dataset({"a": ("x", [1, 2])}, coords={"x": [10, 11]})
    plain = xr.Dataset(
        {"b": ("x", [3, 4])},
        coords=Coordinates({"x": ("x", [20, 21])}, indexes={}),
    )
    left, right = (plain, indexed) if reverse else (indexed, plain)

    assert_indexed_value_wins(
        lambda: xr.merge([left, right], join="exact"), check_mode, "x", [10, 11]
    )


@pytest.mark.parametrize("merge", [False, True])
def test_equal_scalar_coordinate(check_mode, merge):
    indexed = xr.DataArray([1, 2], dims="x", coords={"tag": 10}, name="a").set_xindex(
        "tag", ScalarIndex
    )
    plain = xr.DataArray([3, 4], dims="x", coords={"tag": 10}, name="b")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = xr.merge([indexed, plain]) if merge else indexed + plain

    assert result["tag"].item() == 10


def test_plain_scalar_coordinate_conflict(check_mode):
    left = xr.Dataset(coords={"tag": 10})
    right = xr.Dataset(coords={"tag": 20})

    with pytest.raises(MergeError, match="conflicting values for variable 'tag'"):
        xr.merge([left, right])


def test_override_skips_check(check_mode):
    indexed = xr.Dataset(coords={"tag": 10}).set_xindex("tag", ScalarIndex)
    plain = xr.Dataset(coords={"tag": 20})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = xr.merge([indexed, plain], compat="override")

    assert result["tag"].item() == 10
