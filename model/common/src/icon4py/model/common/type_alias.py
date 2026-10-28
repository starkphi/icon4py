# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import os
from typing import Literal, TypeAlias

import gt4py.next as gtx


DEFAULT_PRECISION = "double"

# wp: working precision, vp: variable precision
# wp_icon: ICON's wp, for fields that cross the ICON boundary; differs from wpfloat only in remixed
wpfloat: TypeAlias = gtx.float64  # noqa: UP040
vpfloat: TypeAlias = gtx.float64  # noqa: UP040
wp_icon: TypeAlias = gtx.float64  # noqa: UP040
type anyfloat = gtx.float32 | gtx.float64

precision = os.environ.get("ICON4PY_FLOAT_PRECISION", DEFAULT_PRECISION).lower()


def set_precision(new_precision: Literal["double", "mixed", "single", "remixed"]) -> None:
    global precision, vpfloat, wpfloat, wp_icon  # noqa: PLW0603 [global-statement]

    precision = new_precision.lower()
    match precision:
        case "double":
            wpfloat = gtx.float64
            vpfloat = gtx.float64
            wp_icon = wpfloat
        case "mixed":
            wpfloat = gtx.float64
            vpfloat = gtx.float32
            wp_icon = wpfloat
        case "single":
            vpfloat = gtx.float32
            wpfloat = gtx.float32
            wp_icon = wpfloat
        case "remixed":
            wpfloat = gtx.float32
            vpfloat = gtx.float32
            wp_icon = gtx.float64
        case _:
            raise ValueError(
                "Only 'double', 'mixed', 'single' and 'remixed' precision are supported."
            )


set_precision(precision)


def dataclass_float_to_wp(self, attributes: list[str] | None = None):
    """Cast float attributes of a dataclass instance to `wpfloat` in place.

    Meant as a helper function to call from `__post_init__`.

    Args:
        self:       The dataclass instance to convert.
        attributes: Names of the attributes to convert.
                    Defaults to all fields whose type annotation contains "float".
    """
    if not dataclasses.is_dataclass(self):
        raise ValueError("This function is meant for dataclasses")
    if attributes is None:
        attributes = [
            field.name
            for field in self.__dataclass_fields__.values()
            if "float" in repr(field.type)
        ]
    for name in attributes or []:
        if not isinstance(v := object.__getattribute__(self, name), wpfloat):
            object.__setattr__(self, name, wpfloat(v))
