# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import os
import sys
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any

import cffi

from icon4py.tools.py2fgen import _conversion, _definitions, _runtime


# TODO(egparedes): possibly use `TypeForm` for the annotation parameter,
# once https://peps.python.org/pep-0747/ is approved.
def _from_annotated(annotation: Any) -> _definitions.ParamDescriptor | None:
    if isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    if hasattr(annotation, "__metadata__"):
        for perf_counters in annotation.__metadata__:
            if isinstance(perf_counters, _definitions.ParamDescriptor):
                return perf_counters
    return None


type AnnotationDescriptorHook = Callable[[Any], _definitions.ParamDescriptor | None]


def param_descriptor_from_annotation(
    annotation: Any,
    annotation_descriptor_hook: AnnotationDescriptorHook | None,
) -> _definitions.ParamDescriptor:
    descriptor = None
    if annotation_descriptor_hook is not None:
        descriptor = annotation_descriptor_hook(annotation)
    if descriptor is None:
        descriptor = _from_annotated(annotation)
    if descriptor is None:
        raise ValueError(f"Could not determine descriptor for type annotation {annotation}.")
    return descriptor


def get_param_descriptors(
    type_hints: dict[str, Any],
    param_descriptors: _definitions.ParamDescriptors | None,
    annotation_descriptor_hook: AnnotationDescriptorHook | None,
) -> _definitions.ParamDescriptors:
    if param_descriptors is not None:
        if annotation_descriptor_hook is not None:
            raise ValueError(
                "Cannot pass both 'param_descriptors' and 'annotation_descriptor_hook'."
            )
        return param_descriptors

    return {
        name: param_descriptor_from_annotation(annotation, annotation_descriptor_hook)
        for name, annotation in type_hints.items()
    }


type AnnotationMappingHook = Callable[
    [Any, _definitions.ParamDescriptor], _definitions.MapperType | None
]


def get_param_mappings(
    type_hints: dict[str, Any],
    annotation_mapping_hook: AnnotationMappingHook | None,
    param_descriptors: _definitions.ParamDescriptors,
) -> Mapping[str, _definitions.MapperType]:
    mappings: dict[str, _definitions.MapperType] = {}
    for name, annotation in type_hints.items():
        if annotation_mapping_hook is not None:
            mapping = annotation_mapping_hook(annotation, param_descriptors[name])
            if mapping is None:
                mapping = _conversion.default_mapping(annotation, param_descriptors[name])
            if mapping is not None:
                mappings[name] = mapping
    return mappings


# Note that the mapping functions are performance relevant and should be cached (not the hook itself)
@dataclasses.dataclass
class _DecoratedFunction:
    """
    Wraps a function to make it exportable with 'py2fgen'.

    A function is exportable if it provides the attribute 'param_descriptors'.

    See ``export()`` for details.
    """

    _fun: Callable
    annotation_descriptor_hook: AnnotationDescriptorHook | None
    annotation_mapping_hook: AnnotationMappingHook | None
    param_descriptors: _definitions.ParamDescriptors | None
    _mapping: Mapping[str, Callable] = dataclasses.field(init=False)
    _writeback: Mapping[str, Callable] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        type_hints = typing.get_type_hints(self._fun, include_extras=True)
        if "return" in type_hints:
            if type_hints["return"] is types.NoneType:
                type_hints.pop("return")
            else:
                raise ValueError(
                    "Exported functions must not have a return type. Use 'None' instead."
                )

        self.param_descriptors = get_param_descriptors(
            type_hints, self.param_descriptors, self.annotation_descriptor_hook
        )

        self._mapping = get_param_mappings(
            type_hints, self.annotation_mapping_hook, self.param_descriptors
        )
        self._writeback = {
            name: mapping.writeback
            for name, mapping in self._mapping.items()
            if hasattr(mapping, "writeback")
        }

    def __call__(self, ffi: cffi.FFI, perf_counters: dict | None, **kwargs: Any) -> Any:
        if _TIMING:
            return _timed_call(self, ffi, perf_counters, **kwargs)
        # Notes: For performance reasons we could switch to positional-only arguments
        # (this is purely internal between the generated Python code and this function).
        # However, an experiment showed that this had small impact.
        # Re-evaluate if we need to tweak performance.
        if __debug__ and perf_counters is not None:
            # If `perf_counters` is injected, we store the conversion timers,
            # the `perf_counters` result is evaluated in the generated python code.
            perf_counters["convert_start_time"] = _runtime.perf_counter()
        mapped = {}
        for k, v in kwargs.items():
            if k not in self._mapping:
                mapped[k] = v
                continue
            try:
                mapped[k] = self._mapping[k](v, ffi=ffi)
            except Exception as e:
                e.add_note(f"while converting argument '{k}' of '{self._fun.__name__}'")
                raise
        if __debug__ and perf_counters is not None:
            perf_counters["convert_end_time"] = _runtime.perf_counter()
        result = self._fun(**mapped)
        for k, writeback in self._writeback.items():
            try:
                writeback(kwargs[k], mapped[k], ffi=ffi)
            except Exception as e:
                e.add_note(f"while writing back argument '{k}' of '{self._fun.__name__}'")
                raise
        return result


_TIMING = os.environ.get("ICON4PY_TIMING", "0") == "1"
_TIMING_CALLS: dict[str, int] = {}


def _device_sync(on_gpu: bool) -> None:
    if on_gpu:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()


def _address(obj: Any) -> int | None:
    arr = getattr(obj, "ndarray", None)
    if arr is None:
        return None
    if hasattr(arr, "data") and hasattr(arr.data, "ptr"):
        return int(arr.data.ptr)
    if hasattr(arr, "ctypes"):
        return int(arr.ctypes.data)
    return None


def _timed_call(self, ffi: cffi.FFI, perf_counters: dict | None, **kwargs: Any) -> Any:
    """Measurement only (branch measure/boundary-timing): phase wall times with a device sync at
    each boundary, plus the bytes the conversion layer copies. One line per call to stderr."""
    on_gpu = any(isinstance(v, tuple) and len(v) == 4 and v[2] for v in kwargs.values())
    _device_sync(on_gpu)
    t0 = _runtime.perf_counter()
    mapped = {}
    for k, v in kwargs.items():
        mapped[k] = self._mapping[k](v, ffi=ffi) if k in self._mapping else v
    _device_sync(on_gpu)
    t1 = _runtime.perf_counter()
    result = self._fun(**mapped)
    _device_sync(on_gpu)
    t2 = _runtime.perf_counter()
    for k, writeback in self._writeback.items():
        writeback(kwargs[k], mapped[k], ffi=ffi)
    _device_sync(on_gpu)
    t3 = _runtime.perf_counter()
    name = self._fun.__name__
    n = _TIMING_CALLS[name] = _TIMING_CALLS.get(name, 0) + 1
    copied_in = copied_out = 0
    n_in = n_out = 0
    for k, v in kwargs.items():
        if not (isinstance(v, tuple) and len(v) == 4) or v[0] is None or k not in self._mapping:
            continue
        addr = _address(mapped[k])
        if addr is None or addr == int(ffi.cast("uintptr_t", v[0])):
            continue
        nbytes = mapped[k].ndarray.nbytes
        copied_in += nbytes
        n_in += 1
        if k in self._writeback:
            copied_out += nbytes
            n_out += 1
        if n == 1:
            print(f"TIMING_ARG {name} {k} shape={tuple(v[1])} icon4py_bytes={nbytes} "
                  f"writeback={k in self._writeback}", file=sys.stderr, flush=True)
    print(f"TIMING {name} call={n} convert={t1 - t0:.6f} granule={t2 - t1:.6f} "
          f"writeback={t3 - t2:.6f} copied_in={copied_in} n_in={n_in} copied_out={copied_out} "
          f"n_out={n_out}", file=sys.stderr, flush=True)
    return result


def export(
    annotation_descriptor_hook: AnnotationDescriptorHook | None = None,
    annotation_mapping_hook: AnnotationMappingHook | None = None,
    param_descriptors: _definitions.ParamDescriptors | None = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator to mark a function as exportable.

    The standard mechanism for exporting a function is to decorate the function with
    '@py2fgen.export(param_descriptors=...)'. Where 'ParamDescriptors' is a dictionary
    that provides a ``ParamDescriptor`` for each parameter of the function.

    Additionally, the user can provide a hook to fill 'param_descriptors' from the parameters
    type annotations.

    For runtime processing of a scalar parameter or an 'ArrayInfo', the user can provide a hook
    which provides a mapping function to translate the parameter to a different type, e.g.
    to translate 'ArrayInfo' to a NumPy array.
    Note: The mapping function (not the hook) is called at every invocation of the function,
    therefore it is recommended to use a cache for the mapping function.

    A default mapping is provided, see ``_conversion.default_mapping()``.

    A mapping function may carry a ``writeback(value, mapped_value, *, ffi)`` attribute. It is
    called with the raw argument and the mapped value after the function returns, e.g. to copy
    an array that was converted on the way in back into the caller's memory. It is not called if
    the function raises: the caller then gets an error code, not results.
    """

    # precise typing is impossible since we are manipulating the args (e.g. ArrayInfo to the Python runtime objects)
    def impl(fun: Callable) -> Callable:
        return functools.update_wrapper(
            _DecoratedFunction(
                fun,
                annotation_descriptor_hook=annotation_descriptor_hook,
                annotation_mapping_hook=annotation_mapping_hook,
                param_descriptors=param_descriptors,
            ),
            fun,
        )

    return impl
