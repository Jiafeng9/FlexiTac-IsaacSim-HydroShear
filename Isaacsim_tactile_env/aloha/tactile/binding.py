from __future__ import annotations

from ..helpers.spatial import resolve_mesh_prim
from ..helpers.structure import infer_arm, infer_finger, sensor_slot


class AlohaTactileBinding:
    """Discover ALOHA tactile links and bind each link to a target mesh."""

    def __init__(self, cfg, sim_utils, prim_utils, objects, UsdPhysics, PhysxSchema):
        self.robot_cfg = cfg.robot
        self.tactile_cfg = cfg.tactile
        self.sim_utils = sim_utils
        self.prim_utils = prim_utils
        self.objects = objects
        self.UsdPhysics = UsdPhysics
        self.PhysxSchema = PhysxSchema

    def build(self) -> tuple[list[str], list[str], list[str]]:
        selected_links = self.sort_elastomer_links(self.find_elastomer_links())
        target_root_paths, target_query_paths = self.build_target_query_paths(selected_links)
        for i, (link, root, query) in enumerate(zip(selected_links, target_root_paths, target_query_paths)):
            print(
                f"  [{i}] slot={sensor_slot(link)} arm={infer_arm(link) or '?':>5s}"
                f" elastomer={link} -> query={query}",
                flush=True,
            )
        return selected_links, target_root_paths, target_query_paths

    def find_elastomer_links(self) -> list[str]:
        bodies = self.sim_utils.get_all_matching_child_prims(
            self.robot_cfg.prim_path,
            predicate=lambda p: p.HasAPI(self.UsdPhysics.RigidBodyAPI)
            and p.HasAPI(self.PhysxSchema.PhysxContactReportAPI),
            traverse_instance_prims=False,
        )
        name_token = self.tactile_cfg.link_name_contains.lower()
        elastomers = sorted(
            [p.GetPath().pathString for p in bodies if name_token in p.GetPath().pathString.lower()]
        )[: self.tactile_cfg.max_elastomers]
        if not elastomers:
            raise RuntimeError("No elastomer links found on robot.")
        return elastomers

    def sort_elastomer_links(self, elastomers: list[str]) -> list[str]:
        def sort_key(path: str):
            arm = infer_arm(path)
            arm_order = 0 if arm == "left" else 1

            finger = infer_finger(path)
            finger_order = 0 if finger == "left_finger" else 1

            return (arm_order, finger_order, path)

        return sorted(elastomers, key=sort_key)

    def build_target_query_paths(self, selected_links: list[str]) -> tuple[list[str], list[str]]:
        target_root_paths = [self.objects.target_root_for_arm(infer_arm(link_path)) for link_path in selected_links]

        target_query_paths: list[str] = []
        for root in target_root_paths:
            try:
                qp, _ = resolve_mesh_prim(root, prim_utils=self.prim_utils, sim_utils=self.sim_utils)
            except RuntimeError:
                qp = root
            target_query_paths.append(qp)

        return target_root_paths, target_query_paths
