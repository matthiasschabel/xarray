from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.tests import assert_identical


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
