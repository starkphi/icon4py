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
def _add_analysis_increments_to_vn(
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wp_icon],
    iau_wgt_dyn: wpfloat,
) -> fa.EdgeKField[wp_icon]:
    """Formerly known as _mo_solve_nonhydro_stencil_28."""
    vn_incr_wp = astype(vn_incr, wpfloat)

    vn_wp = vn + astype(iau_wgt_dyn * vn_incr_wp, wp_icon)
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_analysis_increments_to_vn(
    vn_incr: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wp_icon],
    iau_wgt_dyn: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_analysis_increments_to_vn(
        vn_incr=vn_incr,
        vn=vn,
        iau_wgt_dyn=iau_wgt_dyn,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
