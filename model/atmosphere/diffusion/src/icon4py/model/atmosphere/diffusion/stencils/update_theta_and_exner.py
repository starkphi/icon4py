# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wp_icon, wpfloat


@gtx.field_operator
def _update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wp_icon],
    exner: fa.CellKField[wp_icon],
    rd_o_cvd: vpfloat,
) -> tuple[fa.CellKField[wp_icon], fa.CellKField[wp_icon]]:
    z_temp_wp = astype(z_temp, wpfloat)
    # exner in wp_icon: theta_v / z_theta - 1 cancels
    rd_o_cvd_wp_icon = astype(rd_o_cvd, wp_icon)

    z_theta = theta_v
    theta_v = theta_v + astype(area * z_temp_wp, wp_icon)
    exner = exner * (wp_icon("1.0") + rd_o_cvd_wp_icon * (theta_v / z_theta - wp_icon("1.0")))
    return theta_v, exner


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wp_icon],
    exner: fa.CellKField[wp_icon],
    rd_o_cvd: vpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_theta_and_exner(
        z_temp=z_temp,
        area=area,
        theta_v=theta_v,
        exner=exner,
        rd_o_cvd=rd_o_cvd,
        out=(theta_v, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
