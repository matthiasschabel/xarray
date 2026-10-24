from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from types import EllipsisType
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.core import missing
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
