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
def _compute_explicit_part_for_rho_and_exner(
    rho_nnow: fa.CellKField[wp_icon],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKHalfField[wpfloat],
    exner_pr: fa.CellKField[wp_icon],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKHalfField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[wp_icon], fa.CellKField[wp_icon]]:
    """Formerly known as _mo_solve_nonhydro_stencil_48 or _mo_solve_nonhydro_stencil_49."""
    inv_ddqz_z_full_wp, z_flxdiv_mass_wp, z_beta_wp, z_flxdiv_theta_wp, ddt_exner_phy_wp = astype(
        (inv_ddqz_z_full, z_flxdiv_mass, z_beta, z_flxdiv_theta, ddt_exner_phy), wpfloat
    )

    z_rho_expl_wp = rho_nnow - astype(
        dtime
        * inv_ddqz_z_full_wp
        * (z_flxdiv_mass_wp + z_contr_w_fl_l(dims.KDim - 0.5) - z_contr_w_fl_l(dims.KDim + 0.5)),
        wp_icon,
    )

    z_exner_expl_wp = (
        exner_pr
        - astype(
            z_beta_wp
            * (
                z_flxdiv_theta_wp
                + theta_v_ic(dims.KDim - 0.5) * z_contr_w_fl_l(dims.KDim - 0.5)
                - theta_v_ic(dims.KDim + 0.5) * z_contr_w_fl_l(dims.KDim + 0.5)
            ),
            wp_icon,
        )
        + astype(dtime * ddt_exner_phy_wp, wp_icon)
    )
    return z_rho_expl_wp, z_exner_expl_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_part_for_rho_and_exner(
    z_rho_expl: fa.CellKField[wp_icon],
    z_exner_expl: fa.CellKField[wp_icon],
    rho_nnow: fa.CellKField[wp_icon],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_flxdiv_mass: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKHalfField[wpfloat],
    exner_pr: fa.CellKField[wp_icon],
    z_beta: fa.CellKField[vpfloat],
    z_flxdiv_theta: fa.CellKField[vpfloat],
    theta_v_ic: fa.CellKHalfField[wpfloat],
    ddt_exner_phy: fa.CellKField[vpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_explicit_part_for_rho_and_exner(
        rho_nnow=rho_nnow,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=z_flxdiv_mass,
        z_contr_w_fl_l=z_contr_w_fl_l,
        exner_pr=exner_pr,
        z_beta=z_beta,
        z_flxdiv_theta=z_flxdiv_theta,
        theta_v_ic=theta_v_ic,
        ddt_exner_phy=ddt_exner_phy,
        dtime=dtime,
        out=(z_rho_expl, z_exner_expl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
