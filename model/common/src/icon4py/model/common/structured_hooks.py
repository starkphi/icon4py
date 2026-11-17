# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Structured (I, J, Kolor) backend hooks for direct program calls (structured_global branch only).

Programs that are not set up via `model_options.setup_program` (e.g. in the initial conditions) are called as
`program.with_backend(backend)(...)`. In structured mode (USE_STRUCTURED_BACKEND=1) they must run through
GenericStructuredWrapper instead, which packs the unstructured fields, compiles the structured program and
unpacks the outputs. Call sites use `structured_hooks.with_backend(program, backend, connectivities)` in place of
`program.with_backend(backend)`; removing the structured mode = delete this file and restore those expressions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from icon4py.model.common import model_backends


def with_backend(program: Any, backend: Any, connectivities: Mapping[str, Any]) -> Callable[..., Any]:
    """`program.with_backend(backend)`, or its GenericStructuredWrapper when USE_STRUCTURED_BACKEND=1."""
    if os.environ.get("USE_STRUCTURED_BACKEND", "0") != "1":
        return program.with_backend(backend)
    from gt4py.next.modules.cartesian_interceptor import (
        GenericStructuredWrapper,
        get_global_grid_mapping,
    )
    from gt4py.next.program_processors.runners import gtfn as gtfn_runner

    e2v = connectivities.get("E2V")
    index_map, remap_sizes = get_global_grid_mapping(
        e2v_override=e2v.asnumpy() if e2v is not None else None
    )
    return GenericStructuredWrapper(
        operator=program,
        backend_factory=gtfn_runner.make_gtfn_toolchain,  # the wrapper builds its own gtfn toolchain
        index_map=index_map,
        remap_sizes=remap_sizes,
        allocator=model_backends.get_allocator(backend),
        offset_provider=dict(connectivities),
    )
