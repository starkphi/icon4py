# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Structured-backend replacements for granule programs whose fused neighbour chains exceed the structured halo frame
(structured_global only; installed by structured_hooks.py, removable together with it).

compute_velocity_advection_in_predictor_step (main #1460) computes vt (E2C2E) and the kinetic energy in the same program
as the advection, which reads the kinetic energy through E2C∘C2E: chain E2C∘C2E∘E2C2E on vn. At depth 2 the halo edges of
the (n+2)² frame miss E2C2E neighbours, so vt / kinetic energy are wrong there and the rhombus seam edges read them
(icon4py_kb questions_log 2026-11-18). Split into
  A = diagnostics from vn + contravariant correction to cells (chains up to C2E∘E2C2E, all inside the frame),
  B = advection in the vertical and horizontal momentum equations (= the corrector program plus the skip flag),
with the wrapper's output unpack / input pack in between refreshing the halo from the owned values.
Same outputs and domains as the fused program.
"""

import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.velocity_advection_predictor import (
    _compute_diagnostics_from_normal_wind,
    _interpolate_contravariant_correction_to_cells_on_half_levels,
)
from icon4py.model.atmosphere.dycore.stencils.velocity_advection_terms import (
    _compute_advection_in_horizontal_momentum,
    _compute_advection_in_vertical_momentum,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _predictor_diagnostics_from_normal_wind(
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    nlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
]:
    (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    ) = _compute_diagnostics_from_normal_wind(
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nlev=nlev,
    )
    contravariant_correction_at_cells_on_half_levels = (
        _interpolate_contravariant_correction_to_cells_on_half_levels(
            contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
            e_bln_c_s=e_bln_c_s,
            wgtfac_c=wgtfac_c,
            nflatlev=nflatlev,
        )
    )
    return (
        tangential_wind,
        tangential_wind_on_half_levels,
        vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
        contravariant_correction_at_cells_on_half_levels,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_diagnostics_from_normal_wind(
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    wgtfac_e: fa.EdgeKHalfField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    skip_compute_predictor_vertical_advection: bool,
    nflatlev: gtx.int32,
    start_edge_lateral_boundary_level_5: gtx.int32,
    end_edge_halo_level_2: gtx.int32,
    start_cell_lateral_boundary_level_4: gtx.int32,
    end_cell_halo: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Part A of compute_velocity_advection_in_predictor_step (outputs 1-6, same domains)."""
    _predictor_diagnostics_from_normal_wind(
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        nflatlev=nflatlev,
        nlev=vertical_end,
        out=(
            tangential_wind,
            tangential_wind_on_half_levels,
            vn_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
            contravariant_correction_at_cells_on_half_levels,
        ),
        domain=(
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KHalfDim: (vertical_start, vertical_end + 1),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_lateral_boundary_level_5, end_edge_halo_level_2),
                dims.KDim: (nflatlev, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
        ),
    )


@gtx.field_operator
def _predictor_advection(
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    apply_extra_diffusion_on_vn: bool,
    nlev: gtx.int32,
    end_index_of_damping_layer: gtx.int32,
) -> tuple[
    fa.CellKHalfField[ta.vpfloat],
    fa.CellKHalfField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    (
        maybe_vertical_wind_advective_tendency,
        contravariant_corrected_w_at_cells_on_model_levels,
        vertical_cfl,
    ) = _compute_advection_in_vertical_momentum(
        w=w,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        dtime=dtime,
        skip_vertical_wind_advective_tendency=skip_compute_predictor_vertical_advection,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )
    # as in the fused predictor: forward the previous value if skip_compute_predictor_vertical_advection
    vertical_wind_advective_tendency = (
        maybe_vertical_wind_advective_tendency
        if not skip_compute_predictor_vertical_advection
        else vertical_wind_advective_tendency
    )
    normal_wind_advective_tendency = _compute_advection_in_horizontal_momentum(
        vn=vn,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        tangential_wind=tangential_wind,
        coriolis_frequency=coriolis_frequency,
        contravariant_corrected_w_at_cells_on_model_levels=contravariant_corrected_w_at_cells_on_model_levels,
        vn_on_half_levels=vn_on_half_levels,
        e_bln_c_s=e_bln_c_s,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        geofac_grdiv=geofac_grdiv,
        dtime=dtime,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=nlev,
        end_index_of_damping_layer=end_index_of_damping_layer,
    )
    return vertical_wind_advective_tendency, vertical_cfl, normal_wind_advective_tendency


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def predictor_advection(
    vertical_wind_advective_tendency: fa.CellKHalfField[ta.vpfloat],
    vertical_cfl: fa.CellKHalfField[ta.vpfloat],
    normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    w: fa.CellKHalfField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    tangential_wind_on_half_levels: fa.EdgeKHalfField[ta.wpfloat],
    vn_on_half_levels: fa.EdgeKHalfField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKHalfField[ta.vpfloat],
    coeff1_dwdz: fa.CellKField[ta.vpfloat],
    coeff2_dwdz: fa.CellKField[ta.vpfloat],
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    ddqz_z_half: fa.CellKHalfField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    owner_mask: fa.CellField[bool],
    coriolis_frequency: fa.EdgeField[ta.wpfloat],
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat],
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    dtime: ta.wpfloat,
    skip_compute_predictor_vertical_advection: bool,
    apply_extra_diffusion_on_vn: bool,
    end_index_of_damping_layer: gtx.int32,
    start_cell_lateral_boundary_level_4: gtx.int32,
    end_cell_halo: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_local: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """Part B of compute_velocity_advection_in_predictor_step (outputs 7-9, same domains)."""
    _predictor_advection(
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        vn=vn,
        w=w,
        tangential_wind=tangential_wind,
        tangential_wind_on_half_levels=tangential_wind_on_half_levels,
        vn_on_half_levels=vn_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels=horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        c_intp=c_intp,
        inv_dual_edge_length=inv_dual_edge_length,
        inv_primal_edge_length=inv_primal_edge_length,
        tangent_orientation=tangent_orientation,
        e_bln_c_s=e_bln_c_s,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        owner_mask=owner_mask,
        coriolis_frequency=coriolis_frequency,
        geofac_rot=geofac_rot,
        coeff_gradekin=coeff_gradekin,
        c_lin_e=c_lin_e,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        geofac_grdiv=geofac_grdiv,
        dtime=dtime,
        skip_compute_predictor_vertical_advection=skip_compute_predictor_vertical_advection,
        apply_extra_diffusion_on_vn=apply_extra_diffusion_on_vn,
        nlev=vertical_end,
        end_index_of_damping_layer=end_index_of_damping_layer,
        out=(vertical_wind_advective_tendency, vertical_cfl, normal_wind_advective_tendency),
        domain=(
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start + 1, vertical_end),
            },
            {
                dims.CellDim: (start_cell_lateral_boundary_level_4, end_cell_halo),
                dims.KHalfDim: (vertical_start, vertical_end),
            },
            {
                dims.EdgeDim: (start_edge_nudging_level_2, end_edge_local),
                dims.KDim: (vertical_start, vertical_end),
            },
        ),
    )


# granule attribute -> programs that replace it in structured mode, run in this order with the same arguments
SPLIT_PROGRAMS = {
    "_compute_velocity_advection_in_predictor_step": (
        predictor_diagnostics_from_normal_wind,
        predictor_advection,
    ),
}
