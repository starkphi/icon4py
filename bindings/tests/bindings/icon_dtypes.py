# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""
Hand the bindings their float arrays in the dtype ICON passes.

The tests build their arrays from savepoints in icon4py's compute dtypes, while the bindings
declare (and check) the dtype ICON passes, which `ICON4PY_BINDINGS_ICON_PRECISION` fixes. Where the
two differ -- e.g. icon4py `remixed` (wp=vp=f32 inside) under ICON mixed (wp=f64) -- a call would
fail the bindings' dtype check. Where they agree (double/double, single/single, mixed/mixed) all
of this is a no-op, so those tests see exactly the arrays they built.
"""

import numpy as np
import pytest

from icon4py.bindings import diffusion_wrapper, dycore_wrapper, grid_wrapper
from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import test_utils

from .in_guard import in_field_names


_FLOAT_DTYPES = {py2fgen.FLOAT32: np.dtype(np.float32), py2fgen.FLOAT64: np.dtype(np.float64)}

#: the exported functions the tests call
WRAPPED = (
    (grid_wrapper, "grid_init"),
    (dycore_wrapper, "solve_nh_init"),
    (dycore_wrapper, "solve_nh_run"),
    (dycore_wrapper, "solve_nh_substeps_run"),
    (diffusion_wrapper, "diffusion_init"),
    (diffusion_wrapper, "diffusion_run"),
)


#: the exported functions themselves, taken at import, before any test wraps them
_EXPORTED = {fun_name: getattr(module, fun_name) for module, fun_name in WRAPPED}


def icon_dtype(fun_name: str, name: str) -> np.dtype | None:
    """The dtype ICON passes for float array argument `name` of exported `fun_name`, else None."""
    descriptor = _EXPORTED[fun_name].param_descriptors.get(name)
    if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
        return _FLOAT_DTYPES.get(descriptor.dtype)
    return None


def as_icon_arrays(fun_name: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """`arrays` with every float array in the dtype ICON passes to `fun_name` (same object if equal)."""
    result = {}
    for name, array in arrays.items():
        dtype = icon_dtype(fun_name, name)
        cast = dtype is not None and array is not None and array.dtype != dtype
        result[name] = array.astype(dtype, order="F") if cast else array
    return result


def _cast_arguments(monkeypatch, module, fun_name: str) -> None:
    fun = getattr(module, fun_name)
    in_names = in_field_names(fun)

    def call(*args, **kwargs):  # tests pass `ffi` by position or by keyword
        ffi = args[0] if args else kwargs["ffi"]
        written_back = []
        for name, value in kwargs.items():
            dtype = icon_dtype(fun_name, name)
            if dtype is None or value is None:
                continue
            array = py2fgen.as_array(ffi, value)
            if array is None or array.dtype == dtype:
                continue
            icon_array = array.astype(dtype, order="F")
            kwargs[name] = test_utils.array_to_array_info(icon_array, ffi=ffi)
            if name not in in_names:
                written_back.append((array, icon_array))
        result = fun(*args, **kwargs)
        for array, icon_array in written_back:
            array[...] = icon_array  # back into the test's array, rounded to its dtype
        return result

    # what in_guard reads from an exported function
    call.param_descriptors = fun.param_descriptors  # type: ignore[attr-defined]
    call.__wrapped__ = fun.__wrapped__  # type: ignore[attr-defined]
    monkeypatch.setattr(module, fun_name, call)


@pytest.fixture(autouse=True)
def arguments_in_icon_dtypes(monkeypatch):
    """
    Cast every float array argument whose dtype differs from what ICON passes, for every call.

    Arguments declared IN are not written back. Others are copied back after the call, rounded to
    the test array's dtype: a test that needs ICON-dtype results across calls (e.g. comparing the
    substep loop with per-call substeps bitwise) must build its arrays with `as_icon_arrays`.
    Applied before `in_arguments_unchanged`, which then watches the test's own arrays.
    """
    for module, fun_name in WRAPPED:
        _cast_arguments(monkeypatch, module, fun_name)
