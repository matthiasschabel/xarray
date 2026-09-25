"""Regression tests for optional custom-index operation hooks."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.core.indexing import IndexSelResult
from xarray.core.indexes import PandasIndex
from xarray.core.types import JoinOptions


class TwoCoordinateIndex(xr.Index):
    def __init__(
        self,
        axes: Mapping[Hashable, PandasIndex],
        scalars: Mapping[Hashable, xr.Variable] | None = None,
    ) -> None:
        self.axes = dict(axes)
        self.scalars = dict(scalars or {})

    @classmethod
    def from_variables(
        cls, variables: Mapping[Hashable, xr.Variable], *, options: Mapping[str, Any]
    ) -> TwoCoordinateIndex:
        return cls(
            {
                name: PandasIndex.from_variables({name: var}, options={})
                for name, var in variables.items()
            }
        )

    def create_variables(
        self, variables: Mapping[Hashable, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        result = dict(self.scalars)
        for name, axis in self.axes.items():
            source = {name: variables[name]} if variables and name in variables else None
            result.update(axis.create_variables(source))
        return result

    def isel(self, indexers: Mapping[Hashable, Any]) -> TwoCoordinateIndex:
        axes = {}
        scalars = dict(self.scalars)
        for name, axis in self.axes.items():
            if axis.dim in indexers:
                reduced = axis.isel({axis.dim: indexers[axis.dim]})
                if reduced is None:
                    scalars[name] = xr.Variable((), axis.index[indexers[axis.dim]])
                else:
                    axes[name] = reduced
            else:
                axes[name] = axis
        return type(self)(axes, scalars)

    def sel(self, labels: dict[Hashable, Any]) -> IndexSelResult:
        indexers = {}
        for name, label in labels.items():
            indexers.update(self.axes[name].sel({name: label}).dim_indexers)
        return IndexSelResult(indexers)

    def equals(self, other: xr.Index, **kwargs: Any) -> bool:
        return (
            isinstance(other, TwoCoordinateIndex)
            and self.axes.keys() == other.axes.keys()
            and self.scalars.keys() == other.scalars.keys()
            and all(self.axes[name].equals(other.axes[name]) for name in self.axes)
            and all(self.scalars[name].equals(other.scalars[name]) for name in self.scalars)
        )

    def join(self, other: TwoCoordinateIndex, how: str = "inner") -> TwoCoordinateIndex:
        return type(self)(
            {name: axis.join(other.axes[name], how=how) for name, axis in self.axes.items()},
            self.scalars,
        )

    def reindex_like(self, other: TwoCoordinateIndex) -> dict[Hashable, Any]:
        result = {}
        for name, axis in self.axes.items():
            result.update(axis.reindex_like(other.axes[name]))
        return result

    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        axes = dict(self.axes)
        targets = {}
        for name, other in other_indexes.items():
            assert isinstance(other, PandasIndex)
            if how == "exact":
                target = axes[name]
            elif how == "left":
                target = axes[name]
            elif how == "right":
                target = other
            else:
                target = axes[name].join(other, how=how)
            axes[name] = target
            targets[name] = target
        return type(self)(axes, self.scalars), targets


class GuardedIndex(TwoCoordinateIndex):
    def check_unindexed_coord_conflicts(
        self, variables: Mapping[Hashable, xr.Variable]
    ) -> None:
        for name, var in variables.items():
            if not var.equals(self.create_variables()[name]):
                raise ValueError(f"conflicting unindexed coordinate {name!r}")

    def check_override(self, other: xr.Index) -> None:
        if not self.equals(other):
            raise ValueError("index cannot be overridden")


class RefusingIndex(TwoCoordinateIndex):
    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions,
    ) -> tuple[TwoCoordinateIndex, Mapping[Hashable, xr.Index]]:
        raise ValueError("overlapping labels are unsupported")


class NoOverlapHookIndex(TwoCoordinateIndex):
    join_overlapping = xr.Index.join_overlapping


class AttributeCheckingIndex(TwoCoordinateIndex):
    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        for name, variable in other_variables.items():
            if variable.attrs.get("units") != "meters":
                raise ValueError(f"incompatible units on {name!r}")
        return super().join_overlapping(
            other_indexes, other_variables=other_variables, how=how
        )


def make_array(
    index_cls: type[TwoCoordinateIndex] = TwoCoordinateIndex,
    x_values: list[int] | None = None,
) -> xr.DataArray:
    array = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": x_values or [10, 20, 30]},
    )
    return array.drop_indexes(["y", "x"]).set_xindex(["y", "x"], index_cls)


def make_plain(x_values: list[int]) -> xr.DataArray:
    return xr.DataArray(
        np.arange(6).reshape(2, 3), dims=("y", "x"), coords={"x": x_values}
    )


def test_broadcast_with_two_dimension_index() -> None:
    array = make_array()
    other = xr.DataArray([1, 2], dims="channel")
    result, _ = xr.broadcast(array, other)
    assert result.dims == ("y", "x", "channel")
    assert result.shape == (2, 3, 2)
    np.testing.assert_array_equal(result.x, [10, 20, 30])
    assert result.xindexes["x"] is result.xindexes["y"]
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


@pytest.mark.parametrize("as_dataset", [False, True])
def test_broadcast_preserves_common_indexes(as_dataset: bool) -> None:
    joint: xr.DataArray | xr.Dataset = make_array()
    if as_dataset:
        joint = joint.to_dataset(name="data")
    other = xr.DataArray([1, 2], dims="channel")
    first, second = xr.broadcast(joint, other)
    for result in (first, second):
        assert result.xindexes["x"] is result.xindexes["y"]
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
    result = other.broadcast_like(joint)
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_equal_labels_replace_subset_index(reverse: bool) -> None:
    joint = make_array()
    plain = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 20, 30]},
    )
    operands = (plain, joint) if reverse else (joint, plain)
    for result in xr.align(*operands, copy=False):
        assert result.xindexes["x"] is result.xindexes["y"]
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


@pytest.mark.parametrize("operation", ["arithmetic", "ufunc", "where"])
def test_overlapping_equal_labels_keep_index_in_operations(operation: str) -> None:
    joint = make_array()
    plain = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": [0, 1], "x": [10, 20, 30]},
    )
    if operation == "arithmetic":
        result = plain + joint
    elif operation == "ufunc":
        result = xr.apply_ufunc(lambda a, b: a + b, plain, joint)
    else:
        result = xr.where(plain > 0, plain, joint)
    assert result.xindexes["x"] is result.xindexes["y"]
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


def test_overlapping_right_join_replaces_subset_index() -> None:
    plain = make_plain([20, 30, 40])
    joint = make_array()
    for result in xr.align(plain, joint, join="right"):
        assert result.xindexes["x"] is result.xindexes["y"]
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    ("join", "expected_x"),
    [
        ("inner", [20, 30]),
        ("outer", [10, 20, 30, 40]),
        ("left", [10, 20, 30]),
        ("right", [20, 30, 40]),
    ],
)
def test_overlapping_index_join(
    join: JoinOptions, expected_x: list[int], reverse: bool
) -> None:
    joint = make_array()
    subset = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]}
    )
    operands = (subset, joint) if reverse else (joint, subset)
    if reverse and join in ("left", "right"):
        expected_x = [20, 30, 40] if join == "left" else [10, 20, 30]
    aligned = xr.align(*operands, join=join)
    for result in aligned:
        np.testing.assert_array_equal(result.x, expected_x)
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
        assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_index_exact_rejects_shifted_labels(reverse: bool) -> None:
    joint = make_array()
    subset = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]}
    )
    operands = (subset, joint) if reverse else (joint, subset)
    with pytest.raises(ValueError, match="join='exact'"):
        xr.align(*operands, join="exact")


def test_overlapping_index_refusal_propagates() -> None:
    joint = make_array(RefusingIndex)
    subset = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": [10, 20, 30]}
    )
    with pytest.raises(ValueError, match="overlapping labels are unsupported"):
        xr.align(joint, subset)


def test_overlapping_join_receives_coordinate_attributes() -> None:
    joint = make_array(AttributeCheckingIndex)
    plain = make_plain([10, 20, 30])
    plain.coords["x"].attrs["units"] = "meters"
    assert isinstance(xr.align(joint, plain)[1].xindexes["x"], AttributeCheckingIndex)
    plain.coords["x"].attrs["units"] = "seconds"
    with pytest.raises(ValueError, match="incompatible units on 'x'"):
        xr.align(joint, plain)


def test_overlapping_index_without_hook_keeps_existing_failure() -> None:
    joint = make_array(NoOverlapHookIndex)
    subset = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]}
    )
    with pytest.raises(ValueError, match="conflicting indexes"):
        xr.align(joint, subset)


@pytest.mark.parametrize("reverse", [False, True])
def test_arithmetic_keeps_joint_index_in_both_orders(reverse: bool) -> None:
    joint = make_array()
    subset = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": [10, 20, 30]}
    )
    result = subset + joint if reverse else joint + subset
    assert result.xindexes["x"] is result.xindexes["y"]
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


def test_overlapping_index_joins_two_subset_coordinates() -> None:
    joint = make_array()
    subset = xr.DataArray(
        np.ones((2, 3)),
        dims=("y", "x"),
        coords={"y": [1, 2], "x": [20, 30, 40]},
    )
    first, second = xr.align(joint, subset, join="outer")
    for result in (first, second):
        np.testing.assert_array_equal(result.y, [0, 1, 2])
        np.testing.assert_array_equal(result.x, [10, 20, 30, 40])
        assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize("superset_position", [0, 1, 2])
@pytest.mark.parametrize(
    ("join", "expected_x"),
    [
        ("inner", [30]),
        ("outer", [10, 20, 30, 40, 50]),
        ("left", [20, 30, 40]),
        ("right", [30, 40, 50]),
    ],
)
def test_overlapping_index_join_three_objects(
    join: JoinOptions, expected_x: list[int], superset_position: int
) -> None:
    plain = make_plain([20, 30, 40])
    plain2 = make_plain([30, 40, 50])
    operands = [plain, plain2]
    operands.insert(superset_position, make_array())
    if superset_position == 0 and join == "left":
        expected_x = [10, 20, 30]
    if superset_position == 2 and join == "right":
        expected_x = [10, 20, 30]

    for result in xr.align(*operands, join=join):
        np.testing.assert_array_equal(result.x, expected_x)
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
        assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize("superset_position", [1, 2])
def test_overlapping_index_exact_three_objects(superset_position: int) -> None:
    plain = make_plain([20, 30, 40])
    operands = [plain, plain.copy()]
    operands.insert(superset_position, make_array())
    with pytest.raises(ValueError, match="join='exact'"):
        xr.align(*operands, join="exact")


@pytest.mark.parametrize("superset_position", [1, 2])
def test_overlapping_index_exact_equal_labels_keeps_superset(
    superset_position: int,
) -> None:
    plain = make_plain([10, 20, 30])
    operands = [plain, plain.copy()]
    operands.insert(superset_position, make_array())
    for result in xr.align(*operands, join="exact", copy=False):
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
        assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize(
    ("join", "expected_x"),
    [
        ("inner", [30]),
        ("outer", [10, 20, 30, 40, 50]),
        ("left", [10, 20, 30]),
        ("right", [20, 30, 40]),
    ],
)
def test_overlapping_index_joins_matching_superset_indexes_first(
    join: JoinOptions, expected_x: list[int]
) -> None:
    first = make_array()
    second = make_array(x_values=[20, 30, 40])
    plain = make_plain([30, 40, 50])
    for result in xr.align(first, plain, second, join=join):
        np.testing.assert_array_equal(result.x, expected_x)
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


@pytest.mark.parametrize("reverse", [False, True])
def test_where_three_operands_keep_overlapping_index(reverse: bool) -> None:
    plain = make_plain([10, 20, 30])
    joint = make_array()
    result = xr.where(plain > 2, joint, plain) if reverse else xr.where(plain > 2, plain, joint)
    np.testing.assert_array_equal(result.x, [10, 20, 30])
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
    assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize("superset_position", [1, 2])
def test_apply_ufunc_three_inputs_keep_overlapping_index(
    superset_position: int,
) -> None:
    plain = make_plain([20, 30, 40])
    operands = [plain, plain.copy()]
    operands.insert(superset_position, make_array())
    result = xr.apply_ufunc(lambda a, b, c: a + b + c, *operands, join="inner")
    np.testing.assert_array_equal(result.x, [20, 30])
    assert isinstance(result.xindexes["x"], TwoCoordinateIndex)


def test_alignment_checks_unindexed_coordinate_before_replacement() -> None:
    array = make_array(GuardedIndex)
    other = xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"x": ("x", [11, 21, 31])}
    ).drop_indexes("x")
    with pytest.raises(ValueError, match="conflicting unindexed coordinate 'x'"):
        xr.align(array, other)


def test_merge_checks_unindexed_coordinate_before_replacement() -> None:
    array = make_array(GuardedIndex).to_dataset(name="a")
    other = xr.Dataset({"b": (("y", "x"), np.ones((2, 3)))}, coords={"x": [11, 21, 31]})
    other = other.drop_indexes("x")
    with pytest.raises(ValueError, match="conflicting unindexed coordinate 'x'"):
        xr.merge([array, other])


def test_override_can_be_refused_by_replaced_index() -> None:
    first = make_array(GuardedIndex)
    second = first.isel(x=slice(None, None, -1))
    with pytest.raises(ValueError, match="index cannot be overridden"):
        xr.align(first, second, join="override")


@pytest.mark.parametrize("reverse", [False, True])
def test_override_consults_overlapping_superset_in_both_orders(reverse: bool) -> None:
    joint = make_array(GuardedIndex)
    plain = make_plain([10, 20, 30])
    operands = (plain, joint) if reverse else (joint, plain)
    with pytest.raises(ValueError, match="index cannot be overridden"):
        xr.align(*operands, join="override")


def test_override_consults_first_superset_index() -> None:
    first = make_array(GuardedIndex)
    second = make_array(x_values=[20, 30, 40])
    with pytest.raises(ValueError, match="index cannot be overridden"):
        xr.align(first, second, join="override")


@pytest.mark.parametrize("method", ["isel", "sel"])
@pytest.mark.parametrize("as_dataset", [False, True])
def test_drop_scalar_coordinate_rejects_partial_index_drop(
    method: str, as_dataset: bool
) -> None:
    array = make_array()
    obj = array.to_dataset(name="data") if as_dataset else array
    with pytest.raises(ValueError, match="would corrupt the following index"):
        getattr(obj, method)(x=0 if method == "isel" else 10, drop=True)


def test_drop_all_scalar_index_coordinates() -> None:
    result = make_array().isel(x=0, y=0, drop=True)
    assert not result.coords
    assert not result.xindexes
