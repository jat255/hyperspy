
Tips for writing methods that work on lazy signals
==================================================

With the addition of the ``LazySignal`` class and its derivatives, adding
methods that operate on the data becomes slightly more complicated. However, we
have attempted to streamline it as much as possible. ``LazySignals`` use
``dask.array.Array`` for the ``data`` field instead of the usual
``numpy.ndarray``. The full documentation is available
`here <https://dask.readthedocs.io/en/latest/array.html>`_. While interfaces of
the two arrays are indeed almost identical, the most important differences are
(``da`` being ``dask.array.Array`` in the examples):

 - **Dask arrays are immutable**: ``da[3] = 2`` does not work. ``da += 2``
   does, but it's actually a new object -- you might as well use ``da = da + 2``
   for a better distinction.
 - **Unknown shapes are problematic**: ``res = da[da>0.3]`` works, but the
   shape of the result depends on the values and cannot be inferred without
   execution. Hence, few operations can be run on ``res`` lazily, and it should
   be avoided if possible.

The easiest way to add new methods that work both with arbitrary navigation
dimensions and ``LazySignals`` is by using the ``map`` (or, for more control,
``_map_all`` or ``_map_iterate``) method to map your function ``func`` across
all "navigation pixels" (e.g. spectra in a spectrum-image). ``map`` methods
will run the function on all pixels efficiently and put the results back in the
correct order. ``func`` is not constrained by ``dask`` and can use whatever
code (assignment, etc.) you wish.

If the new method cannot be coerced into a shape suitable for ``map``, separate
cases for lazy signals will have to be written. If a function operates on
arbitrary-sized arrays and the shape of the output can be known before calling,
``da.map_blocks`` and ``da.map_overlap`` are efficient and flexible.

Finally, in addition to ``_iterate_signal`` that is available to all HyperSpy
signals, lazy counterparts also have the ``_block_iterator`` method that 
supports signal and navigation masking and yields (returns on subsequent calls)
the underlying dask blocks as numpy arrays. It is important to note that
stacking all (flat) blocks and reshaping the result into the initial data shape
will not result in identical arrays. For illustration it is best to see the
`dask documentation <https://dask.readthedocs.io/en/latest/array.html>`_.
