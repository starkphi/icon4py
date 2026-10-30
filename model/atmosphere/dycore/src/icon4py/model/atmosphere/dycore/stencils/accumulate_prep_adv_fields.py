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
from icon4py.model.common.type_alias import wp_icon, wpfloat


@gtx.field_operator
def _accumulate_prep_adv_fields(
    z_vn_avg: fa.EdgeKField[wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    vn_traj: fa.EdgeKField[wp_icon],
    mass_flx_me: fa.EdgeKField[wp_icon],
    r_nsubsteps: wpfloat,
) -> tuple[fa.EdgeKField[wp_icon], fa.EdgeKField[wp_icon]]:
    """Formerly kown as _mo_solve_nonhydro_stencil_34."""
    vn_traj_wp = vn_traj + astype(r_nsubsteps * z_vn_avg, wp_icon)
    mass_flx_me_wp = mass_flx_me + astype(r_nsubsteps * mass_fl_e, wp_icon)
    return vn_traj_wp, mass_flx_me_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def accumulate_prep_adv_fields(
    z_vn_avg: fa.EdgeKField[wpfloat],
    mass_fl_e: fa.EdgeKField[wpfloat],
    vn_traj: fa.EdgeKField[wp_icon],
    mass_flx_me: fa.EdgeKField[wp_icon],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _accumulate_prep_adv_fields(
        z_vn_avg=z_vn_avg,
        mass_fl_e=mass_fl_e,
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        r_nsubsteps=r_nsubsteps,
        out=(vn_traj, mass_flx_me),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
