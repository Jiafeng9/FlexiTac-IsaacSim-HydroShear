from __future__ import annotations


class AlohaPatchTransform:
    """Compute the sensor patch transform in each elastomer body frame."""

    def __init__(self, tactile_cfg, urdf_origins: dict, math_utils):
        self.tactile_cfg = tactile_cfg
        self.urdf_origins = urdf_origins
        self.math_utils = math_utils

    def compute(self, link_path: str):
        import torch

        lp = link_path.lower()
        if "_left_finger_link" in lp:
            side = "left"
        elif "_right_finger_link" in lp:
            side = "right"
        else:
            return self.tactile_cfg.patch_offset_pos, self.tactile_cfg.patch_offset_quat

        base_xyz, base_rpy = self.urdf_origins[side]

        base_xyz_t = torch.tensor(base_xyz, dtype=torch.float32).unsqueeze(0)
        base_rpy_t = torch.tensor(base_rpy, dtype=torch.float32).unsqueeze(0)
        user_pos_t = torch.tensor(self.tactile_cfg.patch_offset_pos, dtype=torch.float32).unsqueeze(0)
        user_quat_t = torch.tensor(self.tactile_cfg.patch_offset_quat, dtype=torch.float32).unsqueeze(0)

        q_be = self.math_utils.quat_from_euler_xyz(base_rpy_t[:, 0], base_rpy_t[:, 1], base_rpy_t[:, 2])
        pos_bp = (base_xyz_t + self.math_utils.quat_apply(q_be, user_pos_t)).squeeze(0)
        quat_bp = self.math_utils.quat_mul(q_be, user_quat_t).squeeze(0)

        return tuple(float(v) for v in pos_bp), tuple(float(v) for v in quat_bp)
