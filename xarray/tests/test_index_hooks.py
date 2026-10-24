from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from types import EllipsisType
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.core import missing
from xarray.core.indexes import PandasIndex
from xarray.core.indexing import IndexSelResult
from xarray.core.types import CompatOptions, JoinOptions
from xarray.tests import assert_allclose, assert_identical, requires_scipy


class TwoCoordinateIndex(xr.Index):
    def __init__(self, variables: Mapping[Hashable, xr.Variable]) -> None:
        self.variables = dict(variables)

    @classmethod
    def from_variables(
        cls, variables: Mapping[Hashable, xr.Variable], *, options: Mapping[str, Any]
    ) -> TwoCoordinateIndex:
        return cls(variables)

    def create_variables(
        self, variables: Mapping[Hashable, xr.Variable] | None = None
    ) -> dict[Hashable, xr.Variable]:
        return dict(self.variables)

    def equals(self, other: xr.Index, **kwargs: Any) -> bool:
        return (
            isinstance(other, TwoCoordinateIndex)
            and self.variables.keys() == other.variables.keys()
            and all(
                var.equals(other.variables[name])
                for name, var in self.variables.items()
            )
        )


@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("operation", ["broadcast", "broadcast_like"])
def test_broadcast_preserves_common_indexes(as_dataset: bool, operation: str) -> None:
    joint: xr.DataArray | xr.Dataset = (
        xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [10, 20, 30]},
            name="data",
        )
        .drop_indexes(["x", "y"])
        .set_xindex(["x", "y"], TwoCoordinateIndex)
    )
    other: xr.DataArray | xr.Dataset = xr.DataArray(
        [1, 2], dims="channel", name="other"
    )
    if as_dataset:
        joint = joint.to_dataset()
        other = other.to_dataset()

    if operation == "broadcast":
        first, second = xr.broadcast(joint, other)
    else:
        first = joint.broadcast_like(other)
        second = other.broadcast_like(joint)

    for result in (first, second):
        assert result.xindexes["x"] is result.xindexes["y"]
        assert isinstance(result.xindexes["x"], TwoCoordinateIndex)
        for name in ("x", "y"):
            assert_identical(result.coords[name].variable, joint.coords[name].variable)

    first_data = first["data"] if as_dataset else first
    second_data = second["other"] if as_dataset else second
    np.testing.assert_array_equal(
        first_data.transpose("y", "x", "channel"),
        np.broadcast_to(np.arange(6).reshape(2, 3, 1), (2, 3, 2)),
    )
    np.testing.assert_array_equal(
        second_data.transpose("y", "x", "channel"),
        np.broadcast_to([1, 2], (2, 3, 2)),
    )


@pytest.mark.parametrize("as_dataset", [False, True])
@pytest.mark.parametrize("operation", ["broadcast", "broadcast_like"])
def test_broadcast_excludes_joint_index(as_dataset: bool, operation: str) -> None:
    joint: xr.DataArray | xr.Dataset = (
        xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("y", "x"),
            coords={"y": [0, 1], "x": [10, 20, 30]},
            name="data",
        )
        .drop_indexes(["x", "y"])
        .set_xindex(["x", "y"], TwoCoordinateIndex)
    )
    other: xr.DataArray | xr.Dataset = xr.DataArray(
        [1, 2], dims="channel", name="other"
    )
    if as_dataset:
        joint = joint.to_dataset()
        other = other.to_dataset()

    if operation == "broadcast":
        first, second = xr.broadcast(joint, other, exclude={"x"})
    else:
        first = joint.broadcast_like(other, exclude={"x"})
        second = other.broadcast_like(joint, exclude={"x"})

    assert first.xindexes["x"] is first.xindexes["y"]
    assert first.xindexes["x"].equals(joint.xindexes["x"])
    for name in ("x", "y"):
        assert_identical(first.coords[name].variable, joint.coords[name].variable)
    assert set(second.dims) == {"channel", "y"}
    assert not second.coords
    assert not second.xindexes
    first_data = first["data"] if as_dataset else first
    second_data = second["other"] if as_dataset else second
    np.testing.assert_array_equal(
        first_data.transpose("y", "x", "channel"),
        np.broadcast_to(np.arange(6).reshape(2, 3, 1), (2, 3, 2)),
    )
    np.testing.assert_array_equal(
        second_data.transpose("y", "channel"), np.broadcast_to([1, 2], (2, 2))
    )


class DropCheckingIndex(TwoCoordinateIndex):
    def __init__(self, variables: Mapping[Hashable, xr.Variable]) -> None:
        super().__init__(variables)
        self.calls: list[tuple[str, tuple[Hashable, ...]]] = []
        self.refuse = False

    def check_drop(self, operation: str, dims: Sequence[Hashable]) -> None:
        self.calls.append((operation, tuple(dims)))
        if self.refuse:
            raise ValueError(f"index refuses {operation}")


@pytest.fixture(params=[False, True], ids=["dataarray", "dataset"])
def check_drop_obj(request: pytest.FixtureRequest) -> xr.DataArray | xr.Dataset:
    array = (
        xr.DataArray(
            np.arange(64).reshape(4, 4, 4),
            dims=("time", "x", "y"),
            coords={"time": range(4), "a": ("x", range(4)), "b": ("y", range(4))},
            name="data",
        )
        .drop_indexes("time")
        .set_xindex(["a", "b"], DropCheckingIndex)
        .set_xindex("time", DropCheckingIndex)
    )
    return array.to_dataset() if request.param else array


@pytest.mark.parametrize(
    "operation",
    [
        "stack",
        "pad",
        "coarsen",
        pytest.param("interp", marks=requires_scipy),
        pytest.param("interp_like", marks=requires_scipy),
    ],
)
@pytest.mark.parametrize("dims", [("x",), ("y",), ("x", "y"), ("x", "time"), ("time",)])
def test_check_drop_payload_once_and_untouched(
    check_drop_obj: xr.DataArray | xr.Dataset, operation: str, dims: tuple[str, ...]
) -> None:
    obj = check_drop_obj
    joint = obj.xindexes["a"]
    time = obj.xindexes["time"]
    assert isinstance(joint, DropCheckingIndex)
    assert isinstance(time, DropCheckingIndex)
    assert obj.xindexes["b"] is joint
    joint.refuse = not {"x", "y"}.intersection(dims)
    time.refuse = "time" not in dims

    if operation == "stack":
        result = obj.stack(z=dims, create_index=False)
    elif operation == "pad":
        result = obj.pad(dict.fromkeys(dims, 0), mode="edge")
    elif operation == "interp":
        result = obj.interp({dim: [0.5, 1.5] for dim in dims})
    elif operation == "interp_like":
        result = obj.interp_like(xr.Dataset(coords={dim: [0.5, 1.5] for dim in dims}))
    else:
        coarsened = obj.coarsen(dict.fromkeys(dims, 2))
        assert joint.calls == ([] if joint.refuse else [(operation, dims)])
        assert time.calls == ([] if time.refuse else [(operation, dims)])
        result = coarsened.mean()
        coarsened.construct({dim: (dim, f"{dim}_window") for dim in dims})

    for index, index_dims, names in (
        (joint, ("x", "y"), ("a", "b")),
        (time, ("time",), ("time",)),
    ):
        if index.refuse:
            assert index.calls == []
            for name in names:
                if operation in ("interp", "interp_like"):
                    assert result.xindexes[name].equals(index)
                else:
                    assert result.xindexes[name] is index
        else:
            expected_dims = (
                tuple(dim for dim in dims if dim in index_dims)
                if operation in ("pad", "interp", "interp_like")
                else dims
            )
            expected_operation = "interp" if operation == "interp_like" else operation
            assert index.calls == [(expected_operation, expected_dims)]
            assert all(other is not index for other in result.xindexes.values())


@pytest.mark.parametrize("create_index", [True, False, None])
@pytest.mark.parametrize("dims", [("x",), ("y",), ("x", "time"), (..., "x")])
def test_check_drop_stack_refuses_before_transform(
    check_drop_obj: xr.DataArray | xr.Dataset,
    create_index: bool | None,
    dims: tuple[Hashable | EllipsisType, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = check_drop_obj.xindexes["a"]
    assert isinstance(index, DropCheckingIndex)
    index.refuse = True

    def unexpected_stack(*args: Any, **kwargs: Any) -> None:
        pytest.fail("variables stacked before index preflight")

    monkeypatch.setattr(xr.Variable, "stack", unexpected_stack)
    with pytest.raises(ValueError, match="index refuses stack"):
        check_drop_obj.stack(z=dims, create_index=create_index)
    expected_dims = ("time", "y", "x") if ... in dims else dims
    assert index.calls == [("stack", expected_dims)]


@pytest.mark.parametrize("width", [0, (0, 0), (1, 2)])
@pytest.mark.parametrize("dim", ["x", "y"])
def test_check_drop_pad_refuses_before_transform(
    check_drop_obj: xr.DataArray | xr.Dataset,
    width: int | tuple[int, int],
    dim: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = check_drop_obj.xindexes["a"]
    assert isinstance(index, DropCheckingIndex)
    index.refuse = True

    def unexpected_pad(*args: Any, **kwargs: Any) -> None:
        pytest.fail("variables padded before index preflight")

    monkeypatch.setattr(xr.Variable, "pad", unexpected_pad)
    with pytest.raises(ValueError, match="index refuses pad"):
        check_drop_obj.pad({dim: width, "time": 1})
    assert index.calls == [("pad", (dim,))]


@pytest.mark.parametrize("dim", ["x", "y"])
def test_check_drop_coarsen_refuses_at_construction(
    check_drop_obj: xr.DataArray | xr.Dataset, dim: str
) -> None:
    index = check_drop_obj.xindexes["a"]
    assert isinstance(index, DropCheckingIndex)
    index.refuse = True

    with pytest.raises(ValueError, match="index refuses coarsen"):
        check_drop_obj.coarsen({dim: 2, "time": 2})
    assert index.calls == [("coarsen", (dim, "time"))]


@pytest.mark.parametrize("operation", ["interp", "interp_like"])
@pytest.mark.parametrize("assume_sorted", [False, True])
def test_check_drop_interp_refuses_before_computation(
    check_drop_obj: xr.DataArray | xr.Dataset,
    operation: str,
    assume_sorted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = check_drop_obj.xindexes["a"]
    assert isinstance(index, DropCheckingIndex)
    index.refuse = True

    def unexpected_computation(*args: Any, **kwargs: Any) -> None:
        pytest.fail("interpolation computed before index preflight")

    monkeypatch.setattr(xr.Dataset, "_sort_for_interp", unexpected_computation)
    monkeypatch.setattr(missing, "_localize", unexpected_computation)
    monkeypatch.setattr(missing, "interp", unexpected_computation)
    coords = {"x": [0.5, 1.5], "time": [0.5, 1.5]}
    with pytest.raises(ValueError, match="index refuses interp"):
        if operation == "interp":
            check_drop_obj.interp(coords, assume_sorted=assume_sorted)
        else:
            check_drop_obj.interp_like(
                xr.Dataset(coords=coords), assume_sorted=assume_sorted
            )
    assert index.calls == [("interp", ("x",))]


def test_check_drop_coarsen_validates_dimensions_first(
    check_drop_obj: xr.DataArray | xr.Dataset,
) -> None:
    with pytest.raises(ValueError, match=r"Window dimensions .*missing.* not found"):
        check_drop_obj.coarsen(x=2, missing=2)
    for index in check_drop_obj.xindexes.values():
        assert isinstance(index, DropCheckingIndex)
        assert index.calls == []


@pytest.mark.parametrize(
    "operation",
    [
        "stack",
        "pad",
        "coarsen",
        pytest.param("interp", marks=requires_scipy),
        pytest.param("interp_like", marks=requires_scipy),
    ],
)
def test_check_drop_default_allows_drop(
    check_drop_obj: xr.DataArray | xr.Dataset, operation: str
) -> None:
    plain = check_drop_obj.drop_indexes(["a", "b", "time"])
    obj = plain.set_xindex(["a", "b"], TwoCoordinateIndex)
    index = obj.xindexes["a"]

    if operation == "stack":
        result = obj.stack(z=(...,), create_index=False)
        expected = plain.stack(z=(...,), create_index=False)
    elif operation == "pad":
        result = obj.pad(x=1, mode="edge")
        expected = plain.pad(x=1, mode="edge")
    elif operation == "interp":
        result = obj.interp(x=[0.5, 1.5])
        expected = plain.interp(x=[0.5, 1.5])
    elif operation == "interp_like":
        target = xr.Dataset(coords={"x": [0.5, 1.5]})
        result = obj.interp_like(target)
        expected = plain.interp_like(target)
    else:
        result = obj.coarsen(x=2).mean()
        expected = plain.coarsen(x=2).mean()

    assert all(other is not index for other in result.xindexes.values())
    assert_allclose(result, expected, check_default_indexes=False)


class OverlappingIndex(xr.Index):
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
    ) -> OverlappingIndex:
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
            source = (
                {name: variables[name]} if variables and name in variables else None
            )
            result.update(axis.create_variables(source))
        return result

    def isel(self, indexers: Mapping[Hashable, Any]) -> OverlappingIndex:
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
            isinstance(other, OverlappingIndex)
            and self.axes.keys() == other.axes.keys()
            and self.scalars.keys() == other.scalars.keys()
            and all(self.axes[name].equals(other.axes[name]) for name in self.axes)
            and all(
                self.scalars[name].equals(other.scalars[name]) for name in self.scalars
            )
        )

    def join(self, other: OverlappingIndex, how: str = "inner") -> OverlappingIndex:
        return type(self)(
            {
                name: axis.join(other.axes[name], how=how)
                for name, axis in self.axes.items()
            },
            self.scalars,
        )

    def reindex_like(self, other: OverlappingIndex) -> dict[Hashable, Any]:
        result = {}
        for name, axis in self.axes.items():
            result.update(axis.reindex_like(other.axes[name]))
        return result

    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        if how is None or how == "override":
            return None
        axes = dict(self.axes)
        targets = {}
        for name, other in other_indexes.items():
            assert isinstance(other, PandasIndex)
            if how in ("exact", "left"):
                target = axes[name]
            elif how == "right":
                target = other
            else:
                target = axes[name].join(other, how=how)
            axes[name] = target
            targets[name] = target
        return type(self)(axes, self.scalars), targets


class GuardedIndex(OverlappingIndex):
    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        if how == "override":
            if any(not self.equals(other) for other in other_indexes.values()):
                raise ValueError("index cannot be overridden")
        elif how is None:
            assert not other_indexes
            for name, var in other_variables.items():
                if not var.equals(self.create_variables()[name]):
                    raise ValueError(f"conflicting unindexed coordinate {name!r}")
        return super().join_overlapping(
            other_indexes, other_variables=other_variables, how=how
        )


class RefusingIndex(OverlappingIndex):
    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[OverlappingIndex, Mapping[Hashable, xr.Index]]:
        raise ValueError("overlapping labels are unsupported")


class NoOverlapHookIndex(OverlappingIndex):
    join_overlapping = xr.Index.join_overlapping


class AttributeCheckingIndex(OverlappingIndex):
    def join_overlapping(
        self,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        for name, variable in other_variables.items():
            if variable.attrs.get("units") != "meters":
                raise ValueError(f"incompatible units on {name!r}")
        return super().join_overlapping(
            other_indexes, other_variables=other_variables, how=how
        )


def make_array(
    index_cls: type[OverlappingIndex] = OverlappingIndex,
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
        assert isinstance(result.xindexes["x"], OverlappingIndex)


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
    assert isinstance(result.xindexes["x"], OverlappingIndex)


def test_overlapping_right_join_replaces_subset_index() -> None:
    plain = make_plain([20, 30, 40])
    joint = make_array()
    for result in xr.align(plain, joint, join="right"):
        assert result.xindexes["x"] is result.xindexes["y"]
        assert isinstance(result.xindexes["x"], OverlappingIndex)


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
    subset = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]})
    operands = (subset, joint) if reverse else (joint, subset)
    if reverse and join in ("left", "right"):
        expected_x = [20, 30, 40] if join == "left" else [10, 20, 30]
    aligned = xr.align(*operands, join=join)
    for result in aligned:
        np.testing.assert_array_equal(result.x, expected_x)
        assert isinstance(result.xindexes["x"], OverlappingIndex)
        assert result.xindexes["x"] is result.xindexes["y"]


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_index_exact_rejects_shifted_labels(reverse: bool) -> None:
    joint = make_array()
    subset = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]})
    operands = (subset, joint) if reverse else (joint, subset)
    with pytest.raises(ValueError, match="join='exact'"):
        xr.align(*operands, join="exact")


def test_overlapping_index_refusal_propagates() -> None:
    joint = make_array(RefusingIndex)
    subset = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"x": [10, 20, 30]})
    with pytest.raises(ValueError, match="overlapping labels are unsupported"):
        xr.align(joint, subset)


def test_overlapping_join_accepts_compatible_coordinate_attributes() -> None:
    joint = make_array(AttributeCheckingIndex)
    plain = make_plain([10, 20, 30])
    plain.coords["x"].attrs["units"] = "meters"
    assert isinstance(xr.align(joint, plain)[1].xindexes["x"], AttributeCheckingIndex)


def test_overlapping_join_refuses_incompatible_coordinate_attributes() -> None:
    joint = make_array(AttributeCheckingIndex)
    plain = make_plain([10, 20, 30])
    plain.coords["x"].attrs["units"] = "seconds"
    with pytest.raises(ValueError, match="incompatible units on 'x'"):
        xr.align(joint, plain)


def test_overlapping_index_without_hook_keeps_existing_failure() -> None:
    joint = make_array(NoOverlapHookIndex)
    subset = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"x": [20, 30, 40]})
    with pytest.raises(ValueError, match="conflicting indexes"):
        xr.align(joint, subset)


@pytest.mark.parametrize("reverse", [False, True])
def test_arithmetic_keeps_joint_index_in_both_orders(reverse: bool) -> None:
    joint = make_array()
    subset = xr.DataArray(np.ones((2, 3)), dims=("y", "x"), coords={"x": [10, 20, 30]})
    result = subset + joint if reverse else joint + subset
    assert result.xindexes["x"] is result.xindexes["y"]
    assert isinstance(result.xindexes["x"], OverlappingIndex)


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
        assert isinstance(result.xindexes["x"], OverlappingIndex)
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
        assert isinstance(result.xindexes["x"], OverlappingIndex)
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
        assert isinstance(result.xindexes["x"], OverlappingIndex)


@pytest.mark.parametrize("reverse", [False, True])
def test_where_three_operands_keep_overlapping_index(reverse: bool) -> None:
    plain = make_plain([10, 20, 30])
    joint = make_array()
    result = (
        xr.where(plain > 2, joint, plain)
        if reverse
        else xr.where(plain > 2, plain, joint)
    )
    np.testing.assert_array_equal(result.x, [10, 20, 30])
    assert isinstance(result.xindexes["x"], OverlappingIndex)
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
    assert isinstance(result.xindexes["x"], OverlappingIndex)


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


def make_labelled(x_values: list[int]) -> xr.DataArray:
    return xr.DataArray(
        np.ones((2, 3)), dims=("y", "x"), coords={"y": [0, 1], "x": x_values}
    )


def update_with_array(ds: xr.Dataset, array: xr.DataArray, method: str) -> None:
    if method == "setitem":
        ds["v"] = array
    elif method == "update_mapping":
        ds.update({"v": array})
    else:
        ds.update(array.to_dataset(name="v"))


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_keeps_index_accepting_equal_labels(method: str) -> None:
    ds = make_array().to_dataset(name="a")
    original_index = ds.xindexes["x"]
    labelled = make_labelled([10, 20, 30])
    update_with_array(ds, labelled, method)

    assert isinstance(ds.xindexes["x"], OverlappingIndex)
    assert ds.xindexes["x"] is original_index
    assert ds.xindexes["x"] is ds.xindexes["y"]
    np.testing.assert_array_equal(ds["v"], labelled)
    assert isinstance(labelled.xindexes["x"], PandasIndex)


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_with_shifted_labels_raises(method: str) -> None:
    ds = make_array(RefusingIndex).to_dataset(name="a")

    with pytest.raises(xr.AlignmentError, match="drop_indexes"):
        update_with_array(ds, make_labelled([11, 21, 31]), method)


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_without_overlap_hook_raises(method: str) -> None:
    ds = make_array(NoOverlapHookIndex).to_dataset(name="a")

    with pytest.raises(xr.AlignmentError, match="drop_indexes"):
        update_with_array(ds, make_labelled([10, 20, 30]), method)


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_overlap_refusal_propagates(method: str) -> None:
    ds = make_array(RefusingIndex).to_dataset(name="a")

    with pytest.raises(ValueError, match="overlapping labels are unsupported"):
        update_with_array(ds, make_labelled([10, 20, 30]), method)


@pytest.mark.parametrize(
    "compat",
    ["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("conflict", ["values", "units"])
def test_coordinates_merge_validates_unindexed_conflicts(
    compat: CompatOptions, reverse: bool, conflict: str
) -> None:
    index_cls = GuardedIndex if conflict == "values" else AttributeCheckingIndex
    joint = make_array(index_cls).coords
    other = xr.Coordinates(
        {"x": ("x", [11, 21, 31] if conflict == "values" else [10, 20, 30])},
        indexes={},
    )
    other["x"].attrs["units"] = "seconds"
    first, second = (other, joint) if reverse else (joint, other)
    message = (
        "conflicting unindexed coordinate"
        if conflict == "values"
        else "incompatible units"
    )
    with pytest.raises(ValueError, match=message):
        first.merge(second, compat=compat)


@pytest.mark.parametrize(
    "compat",
    ["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"],
)
@pytest.mark.parametrize("index_cls", [GuardedIndex, AttributeCheckingIndex])
def test_coordinates_merge_accepts_compatible_unindexed_coordinates(
    compat: CompatOptions, index_cls: type[OverlappingIndex]
) -> None:
    joint = make_array(index_cls).coords
    other = xr.Coordinates({"x": ("x", [10, 20, 30], {"units": "meters"})}, indexes={})
    result = joint.merge(other, compat=compat)
    assert result.xindexes["x"] is joint.xindexes["x"]
    assert_identical(result.x.variable, joint["x"].variable)


@pytest.mark.parametrize(
    "compat",
    ["identical", "equals", "broadcast_equals", "no_conflicts", "override", "minimal"],
)
def test_coordinates_merge_default_allows_unindexed_conflicts(
    compat: CompatOptions,
) -> None:
    joint = make_array(NoOverlapHookIndex).coords
    other = xr.Coordinates({"x": ("x", [11, 21, 31], {"units": "seconds"})}, indexes={})
    result = joint.merge(other, compat=compat)
    assert result.xindexes["x"] is joint.xindexes["x"]
    assert_identical(result.x.variable, joint["x"].variable)


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right", "exact"])
@pytest.mark.parametrize("conflict", ["values", "units"])
def test_alignment_validates_unindexed_coordinates_independent_of_join(
    join: JoinOptions, conflict: str
) -> None:
    index_cls = GuardedIndex if conflict == "values" else AttributeCheckingIndex
    joint = make_array(index_cls)
    values = [11, 21, 31] if conflict == "values" else [10, 20, 30]
    other = make_plain(values).drop_indexes("x")
    other.x.attrs["units"] = "seconds"
    message = (
        "conflicting unindexed coordinate"
        if conflict == "values"
        else "incompatible units"
    )
    with pytest.raises(ValueError, match=message):
        xr.align(joint, other, join=join, copy=True)


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right", "exact"])
def test_alignment_default_allows_unindexed_conflicts(join: JoinOptions) -> None:
    joint = make_array(NoOverlapHookIndex)
    other = make_plain([11, 21, 31]).drop_indexes("x")
    _, result = xr.align(joint, other, join=join)
    assert_identical(result.x.variable, joint.x.variable)
    assert isinstance(result.xindexes["x"], NoOverlapHookIndex)


@pytest.mark.parametrize("operation", ["align", "merge"])
def test_unindexed_validation_ignores_returned_targets(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Mapping[Hashable, xr.Variable]] = []

    def validate(
        self: OverlappingIndex,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]]:
        assert how is None
        assert not other_indexes
        calls.append(other_variables)
        return xr.Index(), {}

    monkeypatch.setattr(OverlappingIndex, "join_overlapping", validate)
    joint = make_array()
    other = make_plain([10, 20, 30]).drop_indexes("x")
    if operation == "align":
        _, result = xr.align(joint, other, copy=False)
    else:
        result = joint.coords.merge(other.coords, compat="override")
    assert len(calls) == 1
    assert set(calls[0]) == {"x"}
    assert_identical(calls[0]["x"], other.x.variable)
    assert result.xindexes["x"] is joint.xindexes["x"]


@pytest.mark.parametrize("reverse", [False, True])
def test_override_refuses_partial_intersection(reverse: bool) -> None:
    joint = make_array(GuardedIndex)
    other = (
        make_array()
        .drop_indexes(["x", "y"])
        .rename(y="z")
        .set_xindex(["x", "z"], OverlappingIndex)
    )
    operands = (other, joint) if reverse else (joint, other)
    with pytest.raises(ValueError, match="index cannot be overridden"):
        xr.align(*operands, join="override")


@pytest.mark.parametrize("partial", [False, True])
def test_override_checks_peer_indexes_and_variables_in_both_directions(
    partial: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[
        tuple[xr.Index, Mapping[Hashable, xr.Index], Mapping[Hashable, xr.Variable]]
    ] = []

    def validate(
        self: OverlappingIndex,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]]:
        assert how == "override"
        calls.append((self, other_indexes, other_variables))
        return xr.Index(), {}

    monkeypatch.setattr(OverlappingIndex, "join_overlapping", validate)
    first = make_array()
    second = make_array(x_values=[20, 30, 40])
    first.x.attrs["units"] = "meters"
    second.x.attrs["units"] = "seconds"
    if partial:
        second = (
            second.drop_indexes(["x", "y"])
            .rename(y="z")
            .set_xindex(["x", "z"], OverlappingIndex)
        )
        # Partial groups still conflict after both veto-only calls have allowed them.
        with pytest.raises(xr.AlignmentError, match="conflicting indexes"):
            xr.align(first, second, join="override")
    else:
        left, right = xr.align(first, second, join="override")
        assert left is first
        assert right.xindexes["x"] is first.xindexes["x"]
        assert_identical(right.x.variable, first.x.variable)
        np.testing.assert_array_equal(right.data, second.data)

    assert len(calls) == 2
    for (receiver, indexes, variables), own, peer in zip(
        calls, (first, second), (second, first), strict=True
    ):
        assert receiver is own.xindexes["x"]
        assert set(indexes) == set(variables) == ({"x"} if partial else {"x", "y"})
        for name in indexes:
            assert indexes[name] is peer.xindexes[name]
            assert_identical(variables[name], peer.coords[name].variable)


def test_override_default_keeps_first_matching_index() -> None:
    first = make_array(NoOverlapHookIndex)
    second = make_array(NoOverlapHookIndex, x_values=[20, 30, 40])
    _, result = xr.align(first, second, join="override")
    assert result.xindexes["x"] is first.xindexes["x"]
    assert_identical(result.x.variable, first.x.variable)
    np.testing.assert_array_equal(result.data, second.data)


def test_overlapping_join_is_excluded_with_explicit_indexes() -> None:
    joint = make_array(RefusingIndex)
    other = make_plain([10, 20, 30])
    first, second = xr.align(joint, other, indexes={"x": other.x}, copy=False)
    assert isinstance(first.xindexes["x"], RefusingIndex)
    assert isinstance(second.xindexes["x"], PandasIndex)
    assert first.xindexes["x"].equals(joint.xindexes["x"])
    assert second.xindexes["x"].equals(other.xindexes["x"])


def test_explicit_indexes_keep_conflicting_index_failure() -> None:
    joint = make_array(RefusingIndex)
    other = make_plain([20, 30, 40])
    with pytest.raises(xr.AlignmentError, match="conflicting indexes"):
        xr.align(joint, other, indexes={"x": other.x})


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("coverage", "target indexes for every overlapping coordinate"),
        ("group", "one target index per subset index group"),
    ],
)
def test_overlapping_join_rejects_invalid_targets(
    refresh: bool, invalid: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def join(
        self: OverlappingIndex,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]]:
        nonlocal calls
        calls += 1
        targets = dict(other_indexes)
        if not refresh or calls == 3:
            if refresh:
                assert how == "left"
            if invalid == "coverage":
                del targets["x"]
            else:
                targets["x"] = targets["x"].copy()
        return self, targets

    monkeypatch.setattr(OverlappingIndex, "join_overlapping", join)
    plain = make_labelled([10, 20, 30]).assign_coords(label=("x", ["a", "b", "c"]))
    joint = plain.drop_indexes(["x", "y"]).set_xindex(
        ["x", "y", "label"], OverlappingIndex
    )
    subset = (
        plain.drop_vars("y")
        .drop_indexes("x")
        .set_xindex(["x", "label"], OverlappingIndex)
    )
    operands = (joint, subset, subset.copy()) if refresh else (joint, subset)
    with pytest.raises(ValueError, match=message):
        xr.align(*operands, join="outer")


def test_overlapping_join_cannot_decline_target_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = OverlappingIndex.join_overlapping
    calls = 0

    def join(
        self: OverlappingIndex,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]] | None:
        nonlocal calls
        calls += 1
        if calls == 3:
            assert how == "left"
            return None
        return original(self, other_indexes, other_variables=other_variables, how=how)

    monkeypatch.setattr(OverlappingIndex, "join_overlapping", join)
    with pytest.raises(ValueError, match="must return targets for the joined index"):
        xr.align(make_array(), make_plain([20, 30, 40]), make_plain([30, 40, 50]))


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_uses_left_validation_and_ignores_returned_targets(
    method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_array().to_dataset(name="a")
    original_index = ds.xindexes["x"]
    incoming = make_labelled([10, 20, 30])
    incoming.x.attrs["units"] = "meters"
    calls = 0

    def accept(
        self: OverlappingIndex,
        other_indexes: Mapping[Hashable, xr.Index],
        *,
        other_variables: Mapping[Hashable, xr.Variable],
        how: JoinOptions | None,
    ) -> tuple[xr.Index, Mapping[Hashable, xr.Index]]:
        nonlocal calls
        # merge_collected may also perform validation after incoming indexes are dropped.
        if how is not None:
            calls += 1
            assert how == "left"
            assert self is original_index
            assert set(other_indexes) == {"x", "y"}
            for name, index in other_indexes.items():
                assert index is incoming.xindexes[name]
                assert_identical(other_variables[name], incoming.coords[name].variable)
        return xr.Index(), {}

    monkeypatch.setattr(OverlappingIndex, "join_overlapping", accept)
    update_with_array(ds, incoming, method)
    assert calls == 1
    assert ds.xindexes["x"] is ds.xindexes["y"] is original_index
    np.testing.assert_array_equal(ds["v"], incoming)


@pytest.mark.parametrize("method", ["setitem", "update_mapping", "update_dataset"])
def test_update_rejects_incoming_index_group_outside_receiver(method: str) -> None:
    ds = make_array(RefusingIndex).to_dataset(name="a")
    incoming = (
        make_labelled([10, 20, 30])
        .assign_coords(z=("x", [1, 2, 3]))
        .drop_indexes("x")
        .set_xindex(["x", "z"], OverlappingIndex)
    )
    with pytest.raises(xr.AlignmentError, match="drop_indexes"):
        update_with_array(ds, incoming, method)
