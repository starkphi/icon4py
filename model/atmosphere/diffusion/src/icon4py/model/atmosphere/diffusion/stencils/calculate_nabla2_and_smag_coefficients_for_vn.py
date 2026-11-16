# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2V, E2C2VDim
from icon4py.model.common.math.differential_operators import laplacian_on_edge
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[wpfloat],
]:
    diff_multfac_smag_wp, u_vert_wp, v_vert_wp = astype(
        (diff_multfac_smag, u_vert, v_vert), wpfloat
    )

    # Use explicit per-slot E2C2V access (matching calculate_nabla4 pattern) to
    # avoid a map_(neighbors(E2C2V)) IR node that generates invalid C++ kolor
    # stride lookups for 1-kolor vertex fields in the structured backend.
    v_n_0 = u_vert_wp(E2C2V[0]) * primal_normal_vert_x[E2C2VDim(0)] + v_vert_wp(E2C2V[0]) * primal_normal_vert_y[E2C2VDim(0)]
    v_n_1 = u_vert_wp(E2C2V[1]) * primal_normal_vert_x[E2C2VDim(1)] + v_vert_wp(E2C2V[1]) * primal_normal_vert_y[E2C2VDim(1)]
    v_n_2 = u_vert_wp(E2C2V[2]) * primal_normal_vert_x[E2C2VDim(2)] + v_vert_wp(E2C2V[2]) * primal_normal_vert_y[E2C2VDim(2)]
    v_n_3 = u_vert_wp(E2C2V[3]) * primal_normal_vert_x[E2C2VDim(3)] + v_vert_wp(E2C2V[3]) * primal_normal_vert_y[E2C2VDim(3)]

    nabla2_of_vn = (v_n_0 + v_n_1 - wpfloat("2.0") * vn) * (
        inv_primal_edge_length * inv_primal_edge_length
    ) + (v_n_2 + v_n_3 - wpfloat("2.0") * vn) * (
        inv_vert_vert_length * inv_vert_vert_length
    )
    # The factor of 4 comes from the lengths in the denominator being twice those needed
    # for the diffusion stencil (https://doi.org/10.1002%2Fqj.2378). (merge: upstream moved this into
    # `laplacian_on_edge`; the structured per-slot E2C2V form is kept here)
    nabla2_of_vn = wpfloat("4.0") * nabla2_of_vn

    v_t_0 = u_vert_wp(E2C2V[0]) * dual_normal_vert_x[E2C2VDim(0)] + v_vert_wp(E2C2V[0]) * dual_normal_vert_y[E2C2VDim(0)]
    v_t_1 = u_vert_wp(E2C2V[1]) * dual_normal_vert_x[E2C2VDim(1)] + v_vert_wp(E2C2V[1]) * dual_normal_vert_y[E2C2VDim(1)]
    v_t_2 = u_vert_wp(E2C2V[2]) * dual_normal_vert_x[E2C2VDim(2)] + v_vert_wp(E2C2V[2]) * dual_normal_vert_y[E2C2VDim(2)]
    v_t_3 = u_vert_wp(E2C2V[3]) * dual_normal_vert_x[E2C2VDim(3)] + v_vert_wp(E2C2V[3]) * dual_normal_vert_y[E2C2VDim(3)]
    l_p = tangent_orientation * inv_primal_edge_length
    l_vv = inv_vert_vert_length

    kh_smag_part1 = (v_n_3 - v_n_2) * l_vv - (v_t_1 - v_t_0) * l_p
    kh_smag_part2 = (v_n_1 - v_n_0) * l_p + (v_t_3 - v_t_2) * l_vv

    kh_smag_wp = diff_multfac_smag_wp * sqrt(
        kh_smag_part1 * kh_smag_part1 + kh_smag_part2 * kh_smag_part2
    )

    return astype(kh_smag_wp, vpfloat), nabla2_of_vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_and_smag_coefficients_for_vn(
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    tangent_orientation: fa.EdgeField[wpfloat],
    inv_primal_edge_length: fa.EdgeField[wpfloat],
    inv_vert_vert_length: fa.EdgeField[wpfloat],
    u_vert: fa.VertexKField[vpfloat],
    v_vert: fa.VertexKField[vpfloat],
    primal_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    primal_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_x: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    dual_normal_vert_y: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2VDim], wpfloat],
    vn: fa.EdgeKField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla2_and_smag_coefficients_for_vn(
        diff_multfac_smag=diff_multfac_smag,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        inv_vert_vert_length=inv_vert_vert_length,
        u_vert=u_vert,
        v_vert=v_vert,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        vn=vn,
        out=(kh_smag_e, z_nabla2_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
