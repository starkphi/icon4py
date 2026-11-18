# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Structured (I, J, Kolor) backend hooks of the driver (structured_global branch only).

Everything the driver needs for the structured backend lives here, so that driver.py differs from icon4py
main only by three calls (install_if_enabled, time_loop_start, time_loop_end); removing the structured mode = delete
this file and those calls. Contents: wrapping the granule programs in GenericStructuredWrapper, the shadow mode
(STRUCTURED_SHADOW=1, with STRUCTURED_SHADOW_DIAG dumps), the input mirror cache switch for the time loop and the
prognostic .npy dump for the structured-vs-unstructured comparison (STRUCTURED_DUMP_PROGNOSTICS=<dir>).
"""

import functools
import logging
import os
import pathlib
from collections.abc import Callable

import gt4py.next as gtx

from icon4py.model.common.grid.icon import IconGrid


log = logging.getLogger(__name__)


def _backend_label(backend: object) -> str:
    """main's driver hands over a resolved gtx Backend (custom gtfn or DaCe backend); older callers a backend_like dict."""
    if isinstance(backend, dict):
        factory = backend.get("backend_factory")
        return str(getattr(factory, "__name__", backend)).lower()
    return str(getattr(backend, "name", backend)).lower()


def _program_param_names(program_callable: object) -> tuple[str, ...]:
    past_stage = getattr(program_callable, "past_stage", None)
    past_node = getattr(past_stage, "past_node", None)
    params = getattr(past_node, "params", ())
    names: list[str] = []
    for param in params:
        param_id = getattr(param, "id", None)
        if param_id is not None:
            names.append(str(param_id))
    return tuple(names)


def _make_structured_program_adapter(
    original_callable: functools.partial,
    wrapper: Callable[..., None],
    static_kwargs: dict[str, object],
    param_names: tuple[str, ...],
    shadow: Callable[[Callable[..., None], dict], None] | None = None,
) -> Callable[..., None]:
    def _call(*args, **kwargs):
        if args and not param_names:
            return original_callable(*args, **kwargs)

        merged_kwargs = dict(static_kwargs)
        arg_idx = 0
        for param_name in param_names:
            if param_name in merged_kwargs:
                continue
            if arg_idx >= len(args):
                break
            merged_kwargs[param_name] = args[arg_idx]
            arg_idx += 1

        if arg_idx != len(args):
            raise TypeError(
                f"Unable to map positional args for structured call with params={param_names}."
            )

        merged_kwargs.update(kwargs)
        if shadow is None:
            return wrapper(**merged_kwargs)
        return shadow(wrapper, merged_kwargs)

    return _call


def _make_shadow_runner(  # noqa: PLR0915 [too-many-statements]  # debug aid, kept in one closure
    operator: object, backend: object, connectivities: dict, name: str
) -> Callable[[Callable[..., None], dict], None]:
    """STRUCTURED_SHADOW=1: run the structured program, then the unstructured one on copies of the same inputs,
    log every field that differs, and keep the unstructured result (so later programs see identical inputs and
    differences do not propagate). Debug aid to find which program makes structured differ from unstructured."""
    import numpy as np

    from icon4py.model.common import model_options

    # The structured wrapper builds its toolchain with cached_translation=False, so the reference build cannot be handed
    # the structured C++ from the translation cache (that was a problem with the factory-boy backends before the merge).
    unstructured = operator.with_backend(model_options.customize_backend(operator, backend))
    tol = float(os.environ.get("STRUCTURED_SHADOW_RTOL", "1e-10"))
    counter = {"n": 0}

    def _is_field(v: object) -> bool:
        return hasattr(v, "ndarray") and hasattr(v, "domain")

    # STRUCTURED_SHADOW_DIAG=N (default 1): for the first N calls of this program that differ, log the bad levels and
    # sample values, rerun the structured program on an exact copy of the same inputs (deterministic bug or not?)
    # and save inputs + both results to $STRUCTURED_SHADOW_DUMP_DIR (default: cwd) as an .npz for an offline replay.
    diag_left = {"n": int(os.environ.get("STRUCTURED_SHADOW_DIAG", "1"))}
    dump_dir = os.environ.get("STRUCTURED_SHADOW_DUMP_DIR", ".")

    def _compare(s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, float, float]:
        if s.dtype == bool or not np.issubdtype(s.dtype, np.number):
            bad = s != u
            return bad, 1.0, float(bad.sum())
        d = np.abs(s.astype(float) - u.astype(float))
        scale = max(float(np.nanmax(np.abs(u))) if u.size else 0.0, 1e-300)
        bad = ~np.isclose(s, u, rtol=tol, atol=tol * scale, equal_nan=True)
        return bad, scale, float(np.nanmax(d)) if d.size else 0.0

    def _diagnose(wrapper, kwargs, pre, results, bad_names) -> None:
        n = counter["n"]
        for k in bad_names:
            s, u, bad = results[k]
            idx = np.nonzero(bad)
            if bad.ndim > 1:
                lv, cnt = np.unique(idx[1], return_counts=True)
                log.warning(
                    "SHADOW-DIAG %s#%d %s: bad per level %s",
                    name,
                    n,
                    k,
                    dict(zip(lv.tolist(), cnt.tolist())),
                )
            p0 = np.asarray(gtx.as_field(kwargs[k].domain, pre[id(kwargs[k])]).asnumpy())
            for j in range(min(5, len(idx[0]))):
                at = tuple(int(a[j]) for a in idx)
                log.warning(
                    "SHADOW-DIAG %s#%d %s%s: struct=%r unstruct=%r before_call=%r",
                    name,
                    n,
                    k,
                    list(at),
                    float(s[at]),
                    float(u[at]),
                    float(p0[at]),
                )
        # rerun the structured program on a fresh copy of the same inputs
        kw2 = {
            k: gtx.as_field(v.domain, pre[id(v)].copy()) if _is_field(v) else v
            for k, v in kwargs.items()
        }
        wrapper(**kw2)
        for k in bad_names:
            s2 = np.asarray(kw2[k].asnumpy())
            s, u, _ = results[k]
            bad_u, _, _ = _compare(s2, u)
            bad_s, _, _ = _compare(s2, s)
            log.warning(
                "SHADOW-DIAG %s#%d %s rerun on same inputs: %d differ from unstructured, %d differ from 1st structured run",
                name,
                n,
                k,
                int(bad_u.sum()),
                int(bad_s.sum()),
            )
        path = pathlib.Path(dump_dir) / f"shadow_dump_{name}_{n}.npz"
        arrays = {
            f"in__{k}": np.asarray(gtx.as_field(v.domain, pre[id(v)]).asnumpy())
            for k, v in kwargs.items()
            if _is_field(v)
        }
        arrays.update({f"struct__{k}": results[k][0] for k in bad_names})
        arrays.update({f"unstruct__{k}": results[k][1] for k in bad_names})
        scalars = {k: repr(v) for k, v in kwargs.items() if not _is_field(v)}
        np.savez_compressed(path, scalars=np.array(repr(scalars)), **arrays)
        log.warning("SHADOW-DIAG %s#%d inputs + results saved to %s", name, n, path)

    def _run(wrapper: Callable[..., None], kwargs: dict) -> None:
        counter["n"] += 1
        copies: dict[int, object] = {}
        for v in kwargs.values():
            if _is_field(v) and id(v) not in copies:
                copies[id(v)] = gtx.as_field(v.domain, v.ndarray.copy())
        pre = {i: c.ndarray.copy() for i, c in copies.items()} if diag_left["n"] > 0 else None
        wrapper(**kwargs)
        ukwargs = {k: copies.get(id(v), v) if _is_field(v) else v for k, v in kwargs.items()}
        ukwargs["offset_provider"] = connectivities
        # gt4py reads USE_STRUCTURED_BACKEND at compile time (pass_manager, gtfn_module, type inference):
        # switch it off for the reference build, else the "unstructured" program gets the structured passes.
        saved = os.environ.get("USE_STRUCTURED_BACKEND")
        os.environ["USE_STRUCTURED_BACKEND"] = "0"
        try:
            unstructured(**ukwargs)
        finally:
            os.environ["USE_STRUCTURED_BACKEND"] = saved if saved is not None else "0"
        results: dict[str, tuple] = {}
        bad_names: list[str] = []
        for k, v in kwargs.items():
            if not _is_field(v):
                continue
            # copy: on numpy asnumpy() is a view, and the field is overwritten with the unstructured result below
            s, u = np.array(v.asnumpy(), copy=True), np.asarray(copies[id(v)].asnumpy())
            bad, scale, dmax = _compare(s, u)
            if bad.any():
                rows = np.unique(np.nonzero(bad)[0])
                log.warning(
                    "SHADOW %s#%d %s: %d/%d differ, max_abs=%.3e (field max %.3e), first rows %s (of %d)",
                    name,
                    counter["n"],
                    k,
                    int(bad.sum()),
                    bad.size,
                    dmax,
                    scale,
                    rows[:8].tolist(),
                    rows.size,
                )
                results[k] = (s, u, bad)
                bad_names.append(k)
            v.ndarray[...] = copies[id(v)].ndarray
        if bad_names and pre is not None:
            diag_left["n"] -= 1
            try:
                _diagnose(wrapper, kwargs, pre, results, bad_names)
            except Exception as e:  # diagnostics must not kill the run
                log.warning(
                    "SHADOW-DIAG %s#%d failed: %s: %s", name, counter["n"], type(e).__name__, e
                )
            for v in (
                kwargs.values()
            ):  # the rerun wrote into copies only, but keep the unstructured state anyway
                if _is_field(v):
                    v.ndarray[...] = copies[id(v)].ndarray
        log.info("SHADOW %s#%d checked", name, counter["n"])

    return _run


def _split_programs_for(attr_name: str, wrapper: object) -> tuple | None:
    """Programs that replace a fused granule program in structured mode (structured_programs.SPLIT_PROGRAMS), or None.
    Only for programs that setup_program already wrapped (their wrapper carries sizes and static params);
    STRUCTURED_SPLIT_PROGRAMS=0 keeps the fused programs (to reproduce the halo-frame error)."""
    from gt4py.next.modules.cartesian_interceptor import GenericStructuredWrapper

    from icon4py.model.driver import structured_programs

    if os.environ.get("STRUCTURED_SPLIT_PROGRAMS", "1") == "0" or not isinstance(
        wrapper, GenericStructuredWrapper
    ):
        return None
    return structured_programs.SPLIT_PROGRAMS.get(attr_name)


def _make_split_adapter(
    programs: tuple,
    fused_wrapper: object,
    original_callable: functools.partial,
    backend: object,
    connectivities: dict,
    name: str,
) -> Callable[..., None]:
    """Run `programs` in order instead of the fused program, each in its own structured wrapper (so the output
    unpack + input pack in between refreshes the halo). Every part takes its own parameters from the fused call's
    bound and call-time arguments; sizes and static params are inherited from the fused program's wrapper."""
    import numpy as np
    from gt4py.next.modules.cartesian_interceptor import GenericStructuredWrapper

    bound = dict(original_callable.keywords or {})
    fused_static = frozenset(getattr(fused_wrapper, "_static_params", frozenset()))
    shadow_on = os.environ.get("STRUCTURED_SHADOW", "0") == "1"
    parts = []
    for program in programs:
        params = frozenset(_program_param_names(program))
        sizes = {
            k: v
            for k, v in bound.items()
            if k in params
            and k not in fused_static
            and isinstance(v, (int, np.integer))
            and not isinstance(v, (bool, np.bool_))
        }
        sub = GenericStructuredWrapper(
            operator=program,
            backend_factory=fused_wrapper._backend_factory,
            index_map=fused_wrapper.index_map,
            remap_sizes=fused_wrapper.remap_sizes,
            allocator=fused_wrapper.allocator,
            offset_provider=bound.get("offset_provider", connectivities),
            symbolic_domain_sizes=sizes,
            static_params=fused_static & params,
        )
        sub_bound = {k: v for k, v in bound.items() if k in params or k == "offset_provider"}
        part_name = f"{name}[{getattr(program, '__name__', None) or program.definition_stage.definition.__name__}]"
        shadow = (
            _make_shadow_runner(program, backend, connectivities, part_name) if shadow_on else None
        )
        parts.append((sub, sub_bound, params, shadow))
    log.info("structured mode: %s split into %d programs", name, len(parts))

    def _call(*args, **kwargs):
        if args:
            raise TypeError(f"{name}: the structured split adapter takes keyword arguments only")
        for sub, sub_bound, params, shadow in parts:
            kw = {**sub_bound, **{k: v for k, v in kwargs.items() if k in params}}
            if shadow is None:
                sub(**kw)
            else:
                shadow(sub, kw)

    return _call


def _wrap_granule_programs_for_structured_backend(
    granules: tuple[object, ...],
    grid: IconGrid,
    allocator: gtx.typing.Allocator,
    backend: object = None,
) -> None:
    from gt4py.next.modules.cartesian_interceptor import (
        GenericStructuredWrapper,
        get_global_grid_mapping,
    )
    from gt4py.next.modules.translator import build_cell_ijk_maps
    from gt4py.next.program_processors.runners import gtfn as gtfn_runner

    if "dace" in _backend_label(backend):
        from gt4py.next.program_processors.runners.dace.workflow.backend import DaCeBackendFactory

        _default_factory = DaCeBackendFactory
    else:
        # upstream replaced the factory-boy GTFNBackendFactory by toolchain builders (#2808); the wrapper ignores the
        # factory on the gtfn path and builds its own toolchain (same as model_options.setup_program on this branch).
        _default_factory = gtfn_runner.make_gtfn_toolchain

    e2v_conn = grid.connectivities.get("E2V")
    e2v_array = e2v_conn.asnumpy() if e2v_conn is not None else None
    index_map, remap_sizes = get_global_grid_mapping(e2v_override=e2v_array)

    wrapped_count = 0
    for granule in granules:
        for attr_name, attr_value in vars(granule).items():
            if not isinstance(attr_value, functools.partial):
                continue

            operator = attr_value.func
            static_kwargs = dict(attr_value.keywords or {})

            # model_options.setup_program may already have wrapped this program in a GenericStructuredWrapper
            # (a partial of the wrapper): reuse it instead of wrapping twice.
            if isinstance(operator, GenericStructuredWrapper):
                wrapper = operator
                # If setup_program created the wrapper without connectivity (e.g. copy_field),
                # inject cell mapping now using the full grid connectivities.
                if wrapper.cell_to_ijk is None and "C2V" in grid.connectivities:
                    import numpy as np

                    c2v_np = grid.connectivities["C2V"].asnumpy()
                    wrapper.cell_to_ijk, wrapper.ijk_to_cell = build_cell_ijk_maps(
                        c2v_np, wrapper.index_map
                    )
                    # Also inject cell_to_ijk into the symbolic_domain_sizes_base so
                    # the compiler knows about cell structure.
                    if hasattr(wrapper, "_symbolic_domain_sizes_base"):
                        wrapper._symbolic_domain_sizes_base["cell_to_ijk"] = [
                            (int(i), int(j), int(k)) for i, j, k in np.asarray(wrapper.cell_to_ijk)
                        ]
                param_names = _program_param_names(wrapper._operator)
            else:
                wrapper = GenericStructuredWrapper(
                    operator=operator,
                    backend_factory=_default_factory,
                    index_map=index_map,
                    remap_sizes=remap_sizes,
                    allocator=allocator,
                    offset_provider=grid.connectivities,
                )
                param_names = _program_param_names(operator)

            split = _split_programs_for(attr_name, wrapper)
            if split is not None:
                setattr(
                    granule,
                    attr_name,
                    _make_split_adapter(
                        split,
                        wrapper,
                        attr_value,
                        backend,
                        grid.connectivities,
                        f"{type(granule).__name__}.{attr_name}",
                    ),
                )
                wrapped_count += 1
                continue

            adapter = _make_structured_program_adapter(
                original_callable=attr_value,
                wrapper=wrapper,
                static_kwargs=static_kwargs,
                param_names=param_names,
                shadow=_make_shadow_runner(
                    wrapper._operator,
                    backend,
                    grid.connectivities,
                    f"{type(granule).__name__}.{attr_name}",
                )
                if os.environ.get("STRUCTURED_SHADOW", "0") == "1"
                else None,
            )
            setattr(granule, attr_name, adapter)
            wrapped_count += 1

    log.info(
        "Structured backend enabled in the driver: wrapped %s program callables",
        wrapped_count,
    )


def install_if_enabled(
    granules: tuple[object, ...],
    grid: IconGrid,
    allocator: gtx.typing.Allocator,
    backend: object,
) -> None:
    """USE_STRUCTURED_BACKEND=1: route the granules' programs through GenericStructuredWrapper."""
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
        return
    backend_label = _backend_label(backend)
    if "gtfn" not in backend_label and "dace" not in backend_label:
        log.warning(
            "USE_STRUCTURED_BACKEND=1 requested, but backend '%s' is not GTFN or DaCe. Skipping structured wrapper injection.",
            backend_label,
        )
        return
    _wrap_granule_programs_for_structured_backend(
        granules=tuple(g for g in granules if g is not None),
        grid=grid,
        allocator=allocator,
        backend=backend,
    )


def _mirror_cache_switch() -> Callable[[bool], None] | None:
    if (
        os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1"
        or os.environ.get("STRUCTURED_SHADOW", "0")
        == "1"  # the shadow runner writes the unstructured fields itself
        or os.environ.get("GT4PY_GLOBAL_MIRRORS", "1") == "0"
    ):
        return None
    try:
        from gt4py.next.modules.cartesian_interceptor import enable_mirror_cache
    except ImportError:
        return None
    return enable_mirror_cache


def time_loop_start() -> None:
    """Structured global mode: reuse packed input mirrors across program calls during the time loop (write-through,
    gt4py cartesian_interceptor). Off in shadow mode and with GT4PY_GLOBAL_MIRRORS=0."""
    switch = _mirror_cache_switch()
    if switch is not None:
        switch(True)
        log.info("structured input mirror cache enabled for the time loop")


def time_loop_end(prognostic_state: object | None = None) -> None:
    switch = _mirror_cache_switch()
    if switch is not None:
        switch(False)
    _dump_prognostics(prognostic_state)


def _dump_prognostics(prognostic_state: object | None) -> None:
    """STRUCTURED_DUMP_PROGNOSTICS=<dir>: save the prognostic fields after the time loop as <dir>/<name>.npy
    (structured-vs-unstructured comparison with icon4py_kb/scripts/compare_driver_output.py; works in both modes)."""
    out = os.environ.get("STRUCTURED_DUMP_PROGNOSTICS")
    if not out or prognostic_state is None:
        return
    import dataclasses

    import numpy as np

    path = pathlib.Path(out)
    path.mkdir(parents=True, exist_ok=True)
    for f in dataclasses.fields(prognostic_state):
        v = getattr(prognostic_state, f.name)
        if hasattr(v, "asnumpy"):
            np.save(path / f"{f.name}.npy", v.asnumpy())
    log.info("prognostic fields saved to %s", path)
