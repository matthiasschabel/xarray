.. currentmodule:: xarray

Indexes
=======


.. seealso::
    See the Xarray gallery on `custom indexes <https://xarray-indexes.readthedocs.io/>`_ for more examples.


Creating indexes
----------------
.. autosummary::
   :toctree: ../generated/

   cftime_range
   date_range
   date_range_like
   indexes.RangeIndex.arange
   indexes.RangeIndex.linspace


Built-in Indexes
----------------

Default, pandas-backed indexes built-in to Xarray:

.. autosummary::
   :toctree: ../generated/

   indexes.PandasIndex
   indexes.PandasMultiIndex


More complex indexes built-in to Xarray:

.. autosummary::
   :toctree: ../generated/

   CFTimeIndex
   indexes.RangeIndex
   indexes.NDPointIndex
   indexes.CoordinateTransformIndex


Building custom indexes
-----------------------

These classes are building blocks for more complex Indexes:

.. autosummary::
   :toctree: ../generated/

   indexes.CoordinateTransform
   indexes.CoordinateTransformIndex
   indexes.NDPointIndex
   indexes.TreeAdapter

The Index base class for building custom indexes:

.. autosummary::
   :toctree: ../generated/

   Index
   Index.from_variables
   Index.concat
   Index.stack
   Index.unstack
   Index.check_drop
   Index.join_overlapping
   Index.create_variables
   Index.should_add_coord_to_array
   Index.to_pandas_index
   Index.isel
   Index.sel
   Index.join
   Index.reindex_like
   Index.equals
   Index.roll
   Index.swap_dims
   Index.rename
   Index.copy

``Index.check_drop`` is optional and allows the operation by default. Override
it to refuse stacking, padding, interpolation (including ``interp_like``), or
coarsening before an index is dropped or replaced.

``Index.join_overlapping`` optionally joins indexed subsets and validates overlapping
coordinates. Its default returns ``None``, declining joins while allowing validation.
With ``how="override"`` it may veto overlapping indexes; with ``how=None`` it may veto
replacement of same-name unindexed coordinates. Both validation modes ignore the return
value. Raising refuses the operation in every mode.


The following are useful when building custom Indexes

.. autosummary::
   :toctree: ../generated/

   IndexSelResult
