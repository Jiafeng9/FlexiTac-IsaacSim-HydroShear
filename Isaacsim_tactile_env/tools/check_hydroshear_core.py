from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactile.contact import SurfacePointContactState  # noqa: E402
from tactile.backend import HydroShearTactileBackend, HydroShearTactileBackendCfg  # noqa: E402
from tactile.elastomer import FlatPatchElastomerSdf, FlatPatchElastomerSdfCfg, MeshPatchElastomerSdf  # noqa: E402
from tactile.hydroshear import (  # noqa: E402
    SurfacePointHydroShearCfg,
    SurfacePointHydroShearTracker,
    contact_segment_fraction,
)
from tactile.readout import (  # noqa: E402
    HydroShearMarkerProjector,
    HydroShearMarkerReadoutCfg,
    ProjectedSurfacePointTracker,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjector,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
    create_taxel_grid_points,
)
from tactile.surface import ObjectSurfaceSamples, ObjectSurfaceSampler, ObjectSurfaceSamplerCfg, signed_distance_to_mesh  # noqa: E402
from aloha.tactile.backend import (  # noqa: E402
    HydroShearSensorState,
    HydroShearTactileBackend as AlohaHydroShearTactileBackend,
    HydroShearTactileBackendCfg as AlohaHydroShearTactileBackendCfg,
)


def assert_close(actual, expected, *, atol=1.0e-6):
    actual_t = torch.as_tensor(actual)
    expected_t = torch.as_tensor(expected, dtype=actual_t.dtype, device=actual_t.device)
    if not torch.allclose(actual_t, expected_t, atol=atol, rtol=0.0):
        raise AssertionError(f"expected {expected_t}, got {actual_t}")


def test_surface_sampler_area():
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    samples = ObjectSurfaceSampler(ObjectSurfaceSamplerCfg(num_points=100, seed=2)).sample_arrays(vertices, faces)
    assert_close(samples.total_area, torch.tensor(1.0), atol=1.0e-6)
    assert_close(samples.area.mean(), torch.tensor(0.01), atol=1.0e-6)
    assert samples.vertices_o is not None
    assert samples.faces is not None


def cube_mesh():
    vertices = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.long,
    )
    return vertices, faces


def test_signed_distance_to_mesh_cube():
    vertices, faces = cube_mesh()
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sdf = signed_distance_to_mesh(points, vertices, faces, chunk_size=2).sdf
    assert sdf[0].item() < -0.99
    assert_close(sdf[1], torch.tensor(0.5), atol=1.0e-5)
    assert_close(sdf[2], torch.tensor(0.0), atol=1.0e-5)


def test_flat_sdf():
    sdf = FlatPatchElastomerSdf(FlatPatchElastomerSdfCfg(normal_axis=2))
    points = torch.tensor([[0.0, 0.0, -0.002], [0.0, 0.0, 0.003]], dtype=torch.float32)
    out = sdf.evaluate(points)
    assert_close(out.sdf, torch.tensor([-0.002, 0.003]))


def test_flat_sdf_can_be_bounded_to_patch_extent():
    sdf = FlatPatchElastomerSdf(
        FlatPatchElastomerSdfCfg(normal_axis=2, half_extent_u=0.01, half_extent_v=0.02, eps=1.0e-6)
    )
    points = torch.tensor(
        [
            [0.0, 0.0, -0.002],
            [0.02, 0.0, -0.002],
            [0.0, 0.03, -0.002],
        ],
        dtype=torch.float32,
    )
    out = sdf.evaluate(points)
    assert_close(out.sdf[0], torch.tensor(-0.002), atol=1.0e-6)
    assert out.sdf[1].item() > 0.0
    assert out.sdf[2].item() > 0.0


def test_mesh_elastomer_sdf_uses_real_geometry():
    vertices, faces = cube_mesh()
    sdf = MeshPatchElastomerSdf(vertices_p=vertices, faces=faces, chunk_size=2)
    out = sdf.evaluate(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    assert out.sdf[0].item() < -0.99
    assert_close(out.sdf[1], torch.tensor(0.5), atol=1.0e-5)


def test_alpha_piecewise():
    prev = torch.tensor([0.001, -0.001, -0.001, 0.001])
    curr = torch.tensor([-0.001, -0.002, 0.001, 0.002])
    alpha = contact_segment_fraction(prev, curr)
    assert_close(alpha, torch.tensor([0.5, 1.0, 0.5, 0.0]), atol=1.0e-5)


def make_contact(points, normals, sdf):
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32)
    return SurfacePointContactState(
        points_e=torch.as_tensor(points, dtype=torch.float32),
        normals_e=torch.as_tensor(normals, dtype=torch.float32),
        points_p=torch.as_tensor(points, dtype=torch.float32),
        sdf=sdf_t,
        contact_mask=sdf_t < 0.0,
        penetration=(-sdf_t).clamp_min(0.0),
    )


def test_hydroshear_update_and_reset():
    samples = ObjectSurfaceSamples(
        points_o=torch.zeros((1, 3), dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([0.5], dtype=torch.float32),
    )
    tracker = SurfacePointHydroShearTracker(
        SurfacePointHydroShearCfg(
            normal_stiffness=1000.0,
            shear_stiffness=100.0,
            friction_coefficient=0.5,
            area_mode="mesh_area",
        )
    )

    prev = make_contact([[0.0, 0.0, 0.001]], [[0.0, 0.0, -1.0]], [0.001])
    tracker.update(samples, prev)

    curr = make_contact([[0.0, 0.0, -0.001]], [[0.0, 0.0, -1.0]], [-0.001])
    out = tracker.update(samples, curr)
    assert_close(out.alpha, torch.tensor([0.5]), atol=1.0e-5)
    assert_close(out.normal_force, torch.tensor([0.5]), atol=1.0e-5)
    assert_close(out.shear_force_e, torch.zeros((1, 3)), atol=1.0e-6)

    tangential = make_contact([[0.010, 0.0, -0.001]], [[0.0, 0.0, -1.0]], [-0.001])
    out = tracker.update(samples, tangential)
    assert out.shear_force_e.norm(dim=-1).item() <= 0.5 * out.normal_force.item() + 1.0e-6

    outside = make_contact([[0.010, 0.0, 0.001]], [[0.0, 0.0, -1.0]], [0.001])
    out = tracker.update(samples, outside)
    assert_close(out.normal_force, torch.tensor([0.0]), atol=1.0e-6)
    assert_close(out.shear_force_e, torch.zeros((1, 3)), atol=1.0e-6)


def test_hydroshear_unit_area_displacement_state():
    samples = ObjectSurfaceSamples(
        points_o=torch.zeros((1, 3), dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([0.5], dtype=torch.float32),
    )
    tracker = SurfacePointHydroShearTracker(SurfacePointHydroShearCfg())

    tracker.update(samples, make_contact([[0.0, 0.0, 0.001]], [[0.0, 0.0, -1.0]], [0.001]))
    out = tracker.update(samples, make_contact([[0.0, 0.0, -0.001]], [[0.0, 0.0, -1.0]], [-0.001]))
    assert_close(out.normal_displacement, torch.tensor([0.001]), atol=1.0e-6)
    assert_close(out.displacement_e, torch.tensor([[0.0, 0.0, 0.001]], dtype=torch.float32), atol=1.0e-6)


def test_hydroshear_displacement_stabilizers():
    samples = ObjectSurfaceSamples(
        points_o=torch.zeros((1, 3), dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
    )

    deadband_tracker = SurfacePointHydroShearTracker(
        SurfacePointHydroShearCfg(
            normal_stiffness=1000.0,
            shear_stiffness=100.0,
            motion_deadband=0.01,
        )
    )
    deadband_tracker.update(samples, make_contact([[0.0, 0.0, -0.001]], [[0.0, 0.0, -1.0]], [-0.001]))
    out = deadband_tracker.update(samples, make_contact([[0.0, 0.0, -0.002]], [[0.0, 0.0, -1.0]], [-0.002]))
    assert_close(out.normal_force, torch.tensor([0.0]), atol=1.0e-6)

    clamped_tracker = SurfacePointHydroShearTracker(
        SurfacePointHydroShearCfg(
            normal_stiffness=1000.0,
            shear_stiffness=100.0,
            max_frame_displacement=0.002,
        )
    )
    clamped_tracker.update(samples, make_contact([[0.0, 0.0, -0.001]], [[0.0, 0.0, -1.0]], [-0.001]))
    out = clamped_tracker.update(samples, make_contact([[0.0, 0.0, -0.101]], [[0.0, 0.0, -1.0]], [-0.101]))
    assert_close(out.normal_force, torch.tensor([2.0]), atol=1.0e-5)


def test_taxel_grid_points_match_warp_sdf_layout():
    grid = create_taxel_grid_points(
        TaxelGridCfg(num_rows=2, num_cols=3, point_distance=0.01, normal_axis=2, normal_offset=0.003)
    )
    expected = torch.tensor(
        [
            [-0.005, -0.010, 0.003],
            [-0.005, 0.000, 0.003],
            [-0.005, 0.010, 0.003],
            [0.005, -0.010, 0.003],
            [0.005, 0.000, 0.003],
            [0.005, 0.010, 0.003],
        ],
        dtype=torch.float32,
    )
    assert_close(grid, expected, atol=1.0e-6)


def test_surface_force_projection_to_taxel_grid():
    projector = SurfacePointForceProjector(
        TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
        SurfacePointForceProjectorCfg(lambda_s=0.0),
    )
    out = projector.project(
        surface_points_p=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        penetration=torch.tensor([0.2], dtype=torch.float32),
        normal_force=torch.tensor([2.0], dtype=torch.float32),
        shear_force_e=torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32),
    )
    assert_close(out.normal_force, torch.tensor([[2.0]], dtype=torch.float32), atol=1.0e-6)
    assert_close(out.shear_force_uv, torch.tensor([[[0.6, 0.8]]], dtype=torch.float32), atol=1.0e-6)
    assert_close(out.tactile_force, torch.tensor([[[2.0, 0.6, 0.8]]], dtype=torch.float32), atol=1.0e-6)


def test_projection_separates_normal_and_shear_penetration_weights():
    projector = SurfacePointForceProjector(
        TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
        SurfacePointForceProjectorCfg(lambda_s=0.0),
    )
    out = projector.project(
        surface_points_p=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        penetration=torch.tensor([0.25], dtype=torch.float32),
        normal_force=torch.tensor([4.0], dtype=torch.float32),
        shear_force_e=torch.tensor([[8.0, 12.0, 0.0]], dtype=torch.float32),
    )
    assert_close(out.normal_force, torch.tensor([[4.0]], dtype=torch.float32), atol=1.0e-6)
    assert_close(out.shear_force_uv, torch.tensor([[[2.0, 3.0]]], dtype=torch.float32), atol=1.0e-6)


def test_projection_calibration_and_chunking():
    surface_points = torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=torch.float32)
    penetration = torch.tensor([0.2, 0.1], dtype=torch.float32)
    normal_force = torch.tensor([2.0, 1.0], dtype=torch.float32)
    shear_force = torch.tensor([[3.0, 4.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float32)
    grid_cfg = TaxelGridCfg(num_rows=2, num_cols=2, point_distance=0.01, normal_axis=2)

    full = SurfacePointForceProjector(
        grid_cfg,
        SurfacePointForceProjectorCfg(
            lambda_s=10.0,
            normal_scale=2.0,
            shear_scale=3.0,
            shear_axis_signs=(1.0, -1.0),
        ),
    ).project(
        surface_points_p=surface_points,
        penetration=penetration,
        normal_force=normal_force,
        shear_force_e=shear_force,
    )
    chunked = SurfacePointForceProjector(
        grid_cfg,
        SurfacePointForceProjectorCfg(
            lambda_s=10.0,
            normal_scale=2.0,
            shear_scale=3.0,
            shear_axis_signs=(1.0, -1.0),
            chunk_size=1,
        ),
    ).project(
        surface_points_p=surface_points,
        penetration=penetration,
        normal_force=normal_force,
        shear_force_e=shear_force,
    )
    assert_close(chunked.tactile_force, full.tactile_force, atol=1.0e-6)
    assert torch.all(full.shear_force_uv[..., 1] <= 0.0)


def test_projection_can_use_3d_distance():
    surface_points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    penetration = torch.ones(2, dtype=torch.float32)
    normal_force = torch.ones(2, dtype=torch.float32)
    shear_force = torch.zeros((2, 3), dtype=torch.float32)
    grid_cfg = TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2)

    uv_only = SurfacePointForceProjector(
        grid_cfg,
        SurfacePointForceProjectorCfg(lambda_s=1.0, weight_by_penetration=False, use_3d_distance=False),
    ).project(
        surface_points_p=surface_points,
        penetration=penetration,
        normal_force=normal_force,
        shear_force_e=shear_force,
    )
    full_3d = SurfacePointForceProjector(
        grid_cfg,
        SurfacePointForceProjectorCfg(lambda_s=1.0, weight_by_penetration=False, use_3d_distance=True),
    ).project(
        surface_points_p=surface_points,
        penetration=penetration,
        normal_force=normal_force,
        shear_force_e=shear_force,
    )
    assert_close(uv_only.normal_force, torch.tensor([[2.0]], dtype=torch.float32), atol=1.0e-6)
    expected_3d = torch.tensor([[1.0 + float(torch.exp(torch.tensor(-1.0)))]], dtype=torch.float32)
    assert_close(full_3d.normal_force, expected_3d, atol=1.0e-6)


def test_marker_projector_outputs_marker_field_channels():
    projector = HydroShearMarkerProjector(
        TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
        HydroShearMarkerReadoutCfg(
            lambda_s=0.0,
            lambda_d=0.0,
            shear_weight_by_penetration=False,
            dilation_weight_by_penetration=False,
            shear_scale=1.0,
            dilation_scale=1.0,
            shear_axis_signs=(1.0, -1.0),
        ),
    )
    out = projector.project(
        surface_points_p=torch.tensor([[0.0, 0.0, -0.001]], dtype=torch.float32),
        penetration=torch.tensor([0.001], dtype=torch.float32),
        displacement_e=torch.tensor([[0.002, 0.003, 0.004]], dtype=torch.float32),
    )
    assert out.marker_field.shape == (1, 1, 3)
    assert out.dilation_field.shape == (1, 1, 3)
    assert out.shear_field.shape == (1, 1, 3)
    assert_close(out.dilation_field, torch.tensor([[[0.001, 0.0, 0.0]]], dtype=torch.float32), atol=1.0e-6)
    assert_close(out.shear_field, torch.tensor([[[0.0, -8.0e-6, 1.2e-5]]], dtype=torch.float32), atol=1.0e-8)


def test_marker_projector_uses_object_sdf_for_official_dilation():
    projector = HydroShearMarkerProjector(
        TaxelGridCfg(num_rows=1, num_cols=2, point_distance=1.0, normal_axis=2),
        HydroShearMarkerReadoutCfg(
            lambda_s=0.0,
            lambda_d=0.0,
            shear_weight_by_penetration=False,
            dilation_weight_by_penetration=False,
            shear_scale=0.0,
            dilation_scale=1.0,
        ),
    )
    out = projector.project(
        surface_points_p=torch.zeros((1, 3), dtype=torch.float32),
        penetration=torch.zeros(1, dtype=torch.float32),
        displacement_e=torch.zeros((1, 3), dtype=torch.float32),
        marker_object_sdf=torch.tensor([-0.2, 0.0], dtype=torch.float32),
    )
    assert_close(out.dilation_field[0, 0], torch.tensor([0.0, 0.0, 0.0]), atol=1.0e-6)
    assert_close(out.dilation_field[0, 1], torch.tensor([0.0, 0.0, 0.2]), atol=1.0e-6)


def test_projected_surface_point_tracker():
    tracker = ProjectedSurfacePointTracker(
        normal_axis=2,
        cfg=ProjectedSurfacePointTrackerCfg(
            lambda_d=0.1,
            decay=0.5,
            max_displacement=0.25,
            include_normal_displacement=False,
        ),
    )
    points = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    shear = torch.tensor([[2.0, 0.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    contact = torch.tensor([True, False])

    out = tracker.update(surface_points_p=points, shear_force_e=shear, contact_mask=contact)
    assert_close(out.displacement_p[0], torch.tensor([0.2, 0.0, 0.0]), atol=1.0e-6)
    assert_close(out.displacement_p[1], torch.zeros(3), atol=1.0e-6)
    assert_close(out.projected_points_p[0], torch.tensor([1.2, 2.0, 3.0]), atol=1.0e-6)

    out = tracker.update(surface_points_p=points, shear_force_e=shear, contact_mask=contact)
    assert_close(out.displacement_p[0], torch.tensor([0.25, 0.0, 0.0]), atol=1.0e-6)

    tracker.reset()
    assert tracker.state is None

    full_force_tracker = ProjectedSurfacePointTracker(
        normal_axis=2,
        cfg=ProjectedSurfacePointTrackerCfg(lambda_d=0.1),
    )
    out = full_force_tracker.update(surface_points_p=points, force_e=shear, contact_mask=contact)
    assert_close(out.displacement_p[0], torch.tensor([0.2, 0.0, 0.3]), atol=1.0e-6)


def test_hydroshear_backend_update_observations():
    samples = ObjectSurfaceSamples(
        points_o=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
            projection=SurfacePointForceProjectorCfg(lambda_s=0.0),
            output_key="tactile",
        )
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    backend.update(
        samples,
        object_pos_e=torch.tensor([0.0, 0.0, 0.001], dtype=torch.float32),
        object_quat_e=quat_identity,
    )
    out = backend.update(
        samples,
        object_pos_e=torch.tensor([0.0, 0.0, -0.001], dtype=torch.float32),
        object_quat_e=quat_identity,
    )
    assert out.observations["tactile"].shape == (1, 1)
    assert out.observations["tactile_force"].shape == (1, 1, 3)
    assert out.observations["tactile_shear"].shape == (1, 1, 2)
    assert out.observations["tactile_marker"].shape == (1, 1, 3)
    assert out.observations["tactile"].item() > 0.0


def test_hydroshear_backend_marker_field_output_mode():
    samples = ObjectSurfaceSamples(
        points_o=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
            projection=SurfacePointForceProjectorCfg(lambda_s=0.0),
            marker_projection=HydroShearMarkerReadoutCfg(lambda_s=0.0, lambda_d=0.0),
            output_mode="marker_field",
            output_key="tactile",
        )
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, 0.001]), object_quat_e=quat_identity)
    out = backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, -0.001]), object_quat_e=quat_identity)
    assert out.observations["tactile"].shape == (1, 1, 3)
    assert_close(out.observations["tactile"], out.observations["tactile_marker"], atol=1.0e-6)


def test_axis_aligned_motion_only_produces_matching_marker_shear_axis():
    def run_case(delta_xy: tuple[float, float]) -> torch.Tensor:
        samples = ObjectSurfaceSamples(
            points_o=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
            area=torch.tensor([1.0], dtype=torch.float32),
        )
        backend = HydroShearTactileBackend(
            HydroShearTactileBackendCfg(
                grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
                hydroshear=SurfacePointHydroShearCfg(
                    normal_stiffness=1.0,
                    shear_stiffness=1.0,
                    friction_coefficient=10.0,
                    normal_axis=2,
                ),
                marker_projection=HydroShearMarkerReadoutCfg(
                    lambda_s=0.0,
                    lambda_d=0.0,
                    shear_scale=1.0,
                    dilation_scale=0.0,
                    shear_axis_signs=(1.0, 1.0),
                ),
                output_mode="marker_field",
                output_key="tactile",
            )
        )
        quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, 0.001]), object_quat_e=quat_identity)
        backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, -0.002]), object_quat_e=quat_identity)
        out = backend.update(
            samples,
            object_pos_e=torch.tensor([delta_xy[0], delta_xy[1], -0.002]),
            object_quat_e=quat_identity,
        )
        return out.observations["tactile_marker_shear"][0, 0]

    x_shear = run_case((0.001, 0.0))
    assert_close(x_shear[0], torch.tensor(0.0), atol=1.0e-10)
    assert abs(float(x_shear[1])) > 1.0e-9
    assert_close(x_shear[2], torch.tensor(0.0), atol=1.0e-10)

    y_shear = run_case((0.0, 0.001))
    assert_close(y_shear[0], torch.tensor(0.0), atol=1.0e-10)
    assert_close(y_shear[1], torch.tensor(0.0), atol=1.0e-10)
    assert abs(float(y_shear[2])) > 1.0e-9


def test_force_and_marker_shear_axes_for_all_normal_axes():
    def run_case(normal_axis: int, move_axis: int):
        normal = torch.zeros(3, dtype=torch.float32)
        normal[normal_axis] = -1.0
        samples = ObjectSurfaceSamples(
            points_o=torch.zeros((1, 3), dtype=torch.float32),
            normals_o=normal.view(1, 3),
            area=torch.ones(1, dtype=torch.float32),
        )
        backend = HydroShearTactileBackend(
            HydroShearTactileBackendCfg(
                grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=normal_axis),
                elastomer=FlatPatchElastomerSdfCfg(normal_axis=normal_axis),
                hydroshear=SurfacePointHydroShearCfg(
                    normal_stiffness=1.0,
                    shear_stiffness=1.0,
                    friction_coefficient=10.0,
                    normal_axis=normal_axis,
                ),
                projection=SurfacePointForceProjectorCfg(lambda_s=0.0, shear_axis_signs=(1.0, 1.0)),
                marker_projection=HydroShearMarkerReadoutCfg(
                    lambda_s=0.0,
                    lambda_d=0.0,
                    shear_scale=1.0,
                    dilation_scale=0.0,
                    shear_axis_signs=(1.0, 1.0),
                ),
                output_mode="force_grid",
                output_key="tactile",
            )
        )
        quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        previous = torch.zeros(3, dtype=torch.float32)
        previous[normal_axis] = 0.001
        current = torch.zeros(3, dtype=torch.float32)
        current[normal_axis] = -0.002
        moved = current.clone()
        moved[move_axis] = 0.001

        backend.update(samples, object_pos_e=previous, object_quat_e=quat_identity)
        backend.update(samples, object_pos_e=current, object_quat_e=quat_identity)
        out = backend.update(samples, object_pos_e=moved, object_quat_e=quat_identity)
        return out.observations["tactile_shear"][0, 0], out.observations["tactile_marker_shear"][0, 0]

    for normal_axis in (0, 1, 2):
        tangent_axes = [0, 1, 2]
        tangent_axes.remove(normal_axis)
        for channel, move_axis in enumerate(tangent_axes):
            force_shear, marker_shear = run_case(normal_axis, move_axis)
            other_channel = 1 - channel
            assert abs(float(force_shear[channel])) > 1.0e-9
            assert_close(force_shear[other_channel], torch.tensor(0.0), atol=1.0e-10)
            assert_close(marker_shear[0], torch.tensor(0.0), atol=1.0e-10)
            assert abs(float(marker_shear[channel + 1])) > 1.0e-9
            assert_close(marker_shear[other_channel + 1], torch.tensor(0.0), atol=1.0e-10)


def test_hydroshear_backend_queries_object_sdf_for_marker_dilation():
    vertices, faces = cube_mesh()
    samples = ObjectSurfaceSamples(
        points_o=torch.tensor([[0.0, 0.0, -0.1]], dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
        vertices_o=vertices,
        faces=faces,
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=TaxelGridCfg(num_rows=1, num_cols=2, point_distance=1.0, normal_axis=2),
            marker_projection=HydroShearMarkerReadoutCfg(lambda_s=0.0, lambda_d=0.0, shear_scale=0.0, dilation_scale=1.0),
            output_mode="marker_field",
            output_key="tactile",
        )
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    out = backend.update(samples, object_pos_e=torch.zeros(3), object_quat_e=quat_identity)
    dilation = out.observations["tactile_marker_dilation"]
    assert dilation[0, 0, 2].item() < -0.49
    assert dilation[0, 1, 2].item() > 0.49


def test_hydroshear_backend_readout_ema():
    samples = ObjectSurfaceSamples(
        points_o=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
            hydroshear=SurfacePointHydroShearCfg(normal_stiffness=1000.0),
            projection=SurfacePointForceProjectorCfg(lambda_s=0.0, weight_by_penetration=False),
            readout_ema_alpha=0.5,
        )
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, 0.001]), object_quat_e=quat_identity)
    out = backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, -0.001]), object_quat_e=quat_identity)
    assert_close(out.observations["tactile"], torch.tensor([[0.5]], dtype=torch.float32), atol=1.0e-5)
    out = backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, -0.001]), object_quat_e=quat_identity)
    assert_close(out.observations["tactile"], torch.tensor([[0.75]], dtype=torch.float32), atol=1.0e-5)


def test_hydroshear_backend_projected_surface_state_and_reset():
    samples = ObjectSurfaceSamples(
        points_o=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        normals_o=torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32),
        area=torch.tensor([1.0], dtype=torch.float32),
    )
    backend = HydroShearTactileBackend(
        HydroShearTactileBackendCfg(
            grid=TaxelGridCfg(num_rows=1, num_cols=1, point_distance=0.01, normal_axis=2),
            projected_surface=ProjectedSurfacePointTrackerCfg(lambda_d=0.1),
            projection=SurfacePointForceProjectorCfg(lambda_s=0.0),
        )
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, 0.001]), object_quat_e=quat_identity)
    backend.update(samples, object_pos_e=torch.tensor([0.0, 0.0, -0.001]), object_quat_e=quat_identity)
    out = backend.update(samples, object_pos_e=torch.tensor([0.002, 0.0, -0.001]), object_quat_e=quat_identity)
    assert out.projected_surface is not None
    assert backend.projected_surface_tracker.state is not None
    backend.reset()
    assert backend.tracker.state is None
    assert backend.projected_surface_tracker.state is None


class _Obj:
    pass


def test_aloha_hydroshear_multi_sensor_observations_and_reset():
    cfg = _Obj()
    cfg.robot = _Obj()
    cfg.tactile = _Obj()
    cfg.tactile.num_rows = 1
    cfg.tactile.num_cols = 1
    cfg.tactile.output_key = "tactile"
    cfg.tactile.backend = AlohaHydroShearTactileBackendCfg(include_force_observations=True)
    backend = AlohaHydroShearTactileBackend(cfg, patch_transform=None, robot_asset=None, device="cpu")

    class _Core:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    class _Out:
        def __init__(self, value):
            self.observations = {
                "tactile": torch.tensor([[[value, value + 0.1, value + 0.2]]], dtype=torch.float32),
                "tactile_force": torch.tensor([[[value, value + 1.0, value + 2.0]]], dtype=torch.float32),
                "tactile_shear": torch.tensor([[[value + 1.0, value + 2.0]]], dtype=torch.float32),
                "tactile_marker": torch.tensor([[[value, value + 0.1, value + 0.2]]], dtype=torch.float32),
                "tactile_marker_dilation": torch.tensor([[[value + 0.3, value + 0.4, value + 0.5]]], dtype=torch.float32),
                "tactile_marker_shear": torch.tensor([[[value + 0.6, value + 0.7, value + 0.8]]], dtype=torch.float32),
            }

    core_a = _Core()
    core_b = _Core()
    sensors = [
        HydroShearSensorState("/a", "/target_a", 2, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), core=core_a, last_output=_Out(3.0)),
        HydroShearSensorState("/b", "/target_b", 0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), core=core_b, last_output=_Out(5.0)),
    ]
    obs = backend.observations(sensors, [2, 0])
    assert obs["tactile"].shape == (4, 1, 1, 3)
    assert obs["tactile_force"].shape == (4, 1, 1, 3)
    assert obs["tactile_shear"].shape == (4, 1, 1, 2)
    assert obs["tactile_marker"].shape == (4, 1, 1, 3)
    assert obs["tactile_marker_dilation"].shape == (4, 1, 1, 3)
    assert obs["tactile_marker_shear"].shape == (4, 1, 1, 3)
    assert_close(obs["tactile"][2, 0, 0, 0], torch.tensor(3.0), atol=1.0e-6)
    assert_close(obs["tactile"][0, 0, 0, 0], torch.tensor(5.0), atol=1.0e-6)
    assert_close(obs["tactile_marker_shear"][2, 0, 0], torch.tensor([3.6, 3.7, 3.8]), atol=1.0e-6)

    backend.reset(sensors)
    assert core_a.reset_count == 1
    assert core_b.reset_count == 1
    assert sensors[0].last_output is None
    assert sensors[1].last_output is None


def test_aloha_hydroshear_core_cfg_uses_slot_signs_and_marker_scale():
    cfg = _Obj()
    cfg.robot = _Obj()
    cfg.tactile = _Obj()
    cfg.tactile.num_rows = 2
    cfg.tactile.num_cols = 3
    cfg.tactile.point_distance = 0.01
    cfg.tactile.normal_axis = 2
    cfg.tactile.normal_offset = 0.0
    cfg.tactile.output_key = "tactile"
    cfg.tactile.backend = AlohaHydroShearTactileBackendCfg(
        shear_axis_signs_by_slot=((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)),
        marker_shear_scale=123.0,
        marker_dilation_scale=456.0,
    )
    backend = AlohaHydroShearTactileBackend(cfg, patch_transform=None, robot_asset=None, device="cpu")
    core = backend._make_core_backend(slot=2)
    assert core.cfg.elastomer.half_extent_u == 0.01
    assert core.cfg.elastomer.half_extent_v == 0.015
    assert core.cfg.projection.shear_axis_signs == (1.0, -1.0)
    assert core.cfg.marker_projection.shear_axis_signs == (1.0, -1.0)
    assert core.cfg.marker_projection.shear_scale == 123.0
    assert core.cfg.marker_projection.dilation_scale == 456.0
    assert core.cfg.hydroshear.friction_coefficient == 0.5
    assert core.cfg.projection.lambda_s == 10_800.0


def test_aloha_hydroshear_defaults_match_official_marker_lambdas_and_signs():
    cfg = AlohaHydroShearTactileBackendCfg()
    assert cfg.marker_lambda_s == 10_800.0
    assert cfg.marker_lambda_d == 20_000.0
    assert cfg.shear_axis_signs_by_slot == (
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (-1.0, -1.0),
    )


def main():
    test_surface_sampler_area()
    test_signed_distance_to_mesh_cube()
    test_flat_sdf()
    test_flat_sdf_can_be_bounded_to_patch_extent()
    test_mesh_elastomer_sdf_uses_real_geometry()
    test_alpha_piecewise()
    test_hydroshear_update_and_reset()
    test_hydroshear_unit_area_displacement_state()
    test_hydroshear_displacement_stabilizers()
    test_taxel_grid_points_match_warp_sdf_layout()
    test_surface_force_projection_to_taxel_grid()
    test_projection_separates_normal_and_shear_penetration_weights()
    test_projection_calibration_and_chunking()
    test_projection_can_use_3d_distance()
    test_marker_projector_outputs_marker_field_channels()
    test_marker_projector_uses_object_sdf_for_official_dilation()
    test_projected_surface_point_tracker()
    test_hydroshear_backend_update_observations()
    test_hydroshear_backend_marker_field_output_mode()
    test_axis_aligned_motion_only_produces_matching_marker_shear_axis()
    test_force_and_marker_shear_axes_for_all_normal_axes()
    test_hydroshear_backend_queries_object_sdf_for_marker_dilation()
    test_hydroshear_backend_readout_ema()
    test_hydroshear_backend_projected_surface_state_and_reset()
    test_aloha_hydroshear_multi_sensor_observations_and_reset()
    test_aloha_hydroshear_core_cfg_uses_slot_signs_and_marker_scale()
    test_aloha_hydroshear_defaults_match_official_marker_lambdas_and_signs()
    print("[OK] HydroShear core checks passed")


if __name__ == "__main__":
    main()
