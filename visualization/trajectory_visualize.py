import logging
import numpy as np
import open3d as o3d

from visualization.bbox_visualize import BBoxVisualizer
from visualization.utils import force_camera_pose

logger = logging.getLogger(__name__)


class PredictorVisualizer(BBoxVisualizer):
    """
    World-frame trajectory + box visualizer driven by evaluator `forecasts`.

    Color rules (category-independent):
      • GT (matched & missed): RED   — past, box, future
      • Predictions (matched): PURPLE — past (if available), box, future
      • False positives: YELLOW — past (if available), box, future
      • Ego: GREEN
    """

    # ---- fixed palette -----------------------------------------------------
    RED    = (0.85, 0.10, 0.10)   # GT
    PURPLE = (0.60, 0.35, 0.95)   # predictions
    YELLOW = (0.95, 0.85, 0.10)   # false positives
    GREEN  = (0.10, 0.85, 0.25)   # ego
    ELLIPSE_TINT = 0.60           # lighten factor for covariance ellipses

    def __init__(self, camera_height: float = 40.0, zoom_level: float = 0.1):
        super().__init__(camera_height, zoom_level)
        self.trajectory_geometries = []

    # ---------- transforms ----------
    @staticmethod
    def _lidar_to_world(calib):
        """Return 4x4 world-from-lidar transform."""
        return calib["ego_to_world"] @ calib["lidar_to_ego"]

    def update_cloud(self, pc_dict, calib):
        xyz = pc_dict["data"][:, :3].astype(np.float32)
        T = self._lidar_to_world(calib)
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
        xyz_w = (T @ xyz_h.T).T[:, :3]
        self.cloud.points = o3d.utility.Vector3dVector(xyz_w)
        self.cloud.colors = o3d.utility.Vector3dVector(np.full_like(xyz_w, 1.0))
        self.vis.update_geometry(self.cloud)
    #---------------------------------------------------------------------------------------
    
    def _ensure_fixed_label_colors(self):
        self.label_colors["GT"]   = np.array(self.RED)
        self.label_colors["PRED"] = np.array(self.PURPLE)
        self.label_colors["FP"]   = np.array(self.YELLOW)
        self.label_colors["ego"]  = np.array(self.GREEN)

    def _draw_polyline(self, pts3, color):
        if len(pts3) < 2:
            return None
        lines = [[i, i + 1] for i in range(len(pts3) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts3),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        self.vis.add_geometry(ls)
        return ls

    def _draw_points(self, pts3, color, radius=0.08, z_offset=0.08):
        geoms = []
        for pt in pts3:
            dot = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            dot.translate(pt - np.array([0.0, 0.0, z_offset], float))
            dot.paint_uniform_color(color)
            self.vis.add_geometry(dot)
            geoms.append(dot)
        return geoms
    
    def draw_poly_pts(self, xy, z_ref, color, r_pts=0.08, z_off=0.08, bold=False):
        if xy is None or len(xy) == 0:
            return
        pts3 = np.column_stack([xy, np.full((len(xy),), z_ref, float)])
        g1 = self._draw_polyline(pts3, color)
        g2 = self._draw_points(pts3, color, radius=(0.10 if bold else r_pts), z_offset=z_off)
        if g1:
            self.trajectory_geometries.append(g1)
        self.trajectory_geometries.extend(g2)

    def _ellipse_lineset(self, center_xy, cov2x2, z_ref, color, n_pts=64, sigma_scale=1.0):
        # Expect diagonal covariance: [var_x, var_y] or [[var_x, 0],[0, var_y]]
        cov = np.asarray(cov2x2, dtype=float)
        if cov.shape == (2, 2):
            var_x, var_y = float(cov[0, 0]), float(cov[1, 1])
        else:  # shape (2,)
            var_x, var_y = float(cov[0]), float(cov[1])
    
        # Radii are k * sigma
        rx = sigma_scale * np.sqrt(max(var_x, 1e-12))
        ry = sigma_scale * np.sqrt(max(var_y, 1e-12))
    
        theta = np.linspace(0.0, 2.0*np.pi, n_pts, endpoint=True)
        xs = center_xy[0] + rx * np.cos(theta)
        ys = center_xy[1] + ry * np.sin(theta)
        pts3 = np.column_stack([xs, ys, np.full_like(xs, z_ref, dtype=float)])
    
        lines = [[i, (i + 1) % len(pts3)] for i in range(len(pts3))]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts3),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        self.vis.add_geometry(ls)
        return ls


    @staticmethod
    def _tint(color, t=ELLIPSE_TINT):
        return tuple((t * np.array(color) + (1.0 - t)).tolist())

    # ---------- main entry ----------
    def visualize_forecasts(
        self,
        point_cloud,
        calib,
        ego_pose,
        forecasts: dict,
        show_past: bool = False,
        show_future: bool = False,
        show_missing: bool = False,
        show_false: bool = False,
        sigma_scale: float = 1.0,
    ):

        self.update_cloud(point_cloud, calib)
        
        for g in self.bbox_geometries + self.trajectory_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()
        self.trajectory_geometries.clear()
        
        T = self._lidar_to_world(calib)
        ego_w = T @ np.array([ego_pose["x"], ego_pose["y"], ego_pose["z"], 1.0])
        Rlw = T[:3, :3]
        ego_world = dict(ego_pose)
        ego_world["x"], ego_world["y"], ego_world["z"] = ego_w[:3]
        ego_world["yaw"] = ego_pose["yaw"] + np.arctan2(Rlw[1, 0], Rlw[0, 0])
        cam_center = np.array([ego_world["x"], ego_world["y"], ego_world["z"]])

        self._ensure_fixed_label_colors()
        self.update_ego_bbox(ego_world)  

        dets_GT, dets_PRED, dets_FP = [], [], []

        ####################### Display macthes ########################################
        for m in forecasts.get("matched", []):
            gb = np.asarray(m["gt"]["bbox"], float)
            pb = np.asarray(m["pred"]["bbox"], float)

            #dets_GT.append({
            #    "label": "GT",
            #    "x": gb[0], "y": gb[1], "z": gb[2],
            #    "dx": gb[3], "dy": gb[4], "dz": gb[5], "yaw": gb[6],
            #})
            dets_PRED.append({
                "label": "PRED",
                "x": pb[0], "y": pb[1], "z": pb[2],
                "dx": pb[3], "dy": pb[4], "dz": pb[5], "yaw": pb[6],
            })

            z_ref = float(gb[2])  # reference z

            # GT past/future (RED)
            
            if show_past:
                gt_past = np.asarray(m["gt"]["past"], float)
                if gt_past.size:
                    self.draw_poly_pts(gt_past, z_ref, self.RED, r_pts=0.05, z_off=0.02)
            if show_future:
                gt_fut_xy = np.asarray(m["gt"].get("future"), dtype=float)
                self.draw_poly_pts(gt_fut_xy, z_ref, self.RED, bold=True)

            # Pred past/ future (PURPLE)
            if show_past:
                pred_past = np.asarray(m["pred"]["past"], float)
                if pred_past.size:
                    self.draw_poly_pts(pred_past, z_ref, self.PURPLE, r_pts=0.05, z_off=0.02)
            if show_future:
                fut_d = m["pred"].get("future", {})
                if isinstance(fut_d, dict) and fut_d:
                    ts = sorted(fut_d.keys(), key=float)
                    fut_xy = np.array([fut_d[t] for t in ts], float)
                    self.draw_poly_pts(fut_xy, z_ref, self.PURPLE, bold=True)

                    # covariance ellipses tinted purple
                    cov_d = m["pred"].get("cov", {})
                    if isinstance(cov_d, dict) and cov_d:
                        tinted = self._tint(self.PURPLE)
                        for i, t in enumerate(ts):
                            C = cov_d.get(t)
                            if C is None:
                                continue
                            C = np.asarray(C, float)
                            if C.shape == (2,):
                                C = np.diag(C)
                            if C.shape != (2, 2):
                                continue
                            xy = tuple(fut_xy[i])
                            ell = self._ellipse_lineset(xy, C, z_ref, tinted, sigma_scale=sigma_scale)
                            self.trajectory_geometries.append(ell)
        
        ###########################################################################################
        ####################### Display missing gt objects ########################################
        if show_missing:
            for g in forecasts.get("missed", []):
                gb = np.asarray(g["gt"]["bbox"], float)
                dets_GT.append({
                    "label": "GT",
                    "x": gb[0], "y": gb[1], "z": gb[2],
                    "dx": gb[3], "dy": gb[4], "dz": gb[5], "yaw": gb[6],
                })
                z_ref = float(gb[2])
                if show_past:
                    gt_past = np.asarray(g["gt"]["past"], float)
                    if gt_past.size:
                        self.draw_poly_pts(gt_past, z_ref, self.RED, r_pts=0.05, z_off=0.02)
                if show_future:
                    gt_fut_xy = np.asarray(g["gt"].get("future"), dtype=float)
                    self.draw_poly_pts(gt_fut_xy, z_ref, self.RED, bold=True)
                    
        ###########################################################################################
        ####################### Display false positives ########################################
        if show_false:
            for f in forecasts.get("false_positives", []):
                pb = np.asarray(f["pred"]["bbox"], float)
                dets_FP.append({
                    "label": "FP",
                    "x": pb[0], "y": pb[1], "z": pb[2],
                    "dx": pb[3], "dy": pb[4], "dz": pb[5], "yaw": pb[6],
                })
                z_ref = float(pb[2])

                if show_past and "past" in f.get("pred", {}):
                    pred_past = np.asarray(f["pred"]["past"], float)
                    if pred_past.size:
                        self.draw_poly_pts(pred_past, z_ref, self.YELLOW, r_pts=0.05, z_off=0.02)
                if show_future:
                    fut_d = f["pred"].get("future", {})
                    if isinstance(fut_d, dict) and fut_d:
                        ts = sorted(fut_d.keys(), key=float)
                        fut_xy = np.array([fut_d[t] for t in ts], float)
                        self.draw_poly_pts(fut_xy, z_ref, self.YELLOW, bold=True)

                        # optional: FP covariances (tinted yellow)
                        cov_d = f["pred"].get("cov", {})
                        if isinstance(cov_d, dict) and cov_d:
                            tinted = self._tint(self.YELLOW)
                            for i, t in enumerate(ts):
                                C = cov_d.get(t)
                                if C is None:
                                    continue
                                C = np.asarray(C, float)
                                if C.shape == (2,):
                                    C = np.diag(C)
                                if C.shape != (2, 2):
                                    continue
                                xy = tuple(fut_xy[i])
                                ell = self._ellipse_lineset(xy, C, z_ref, tinted, sigma_scale=sigma_scale)
                                self.trajectory_geometries.append(ell) 
        ###########################################################################################


        if dets_PRED:
            self.update_bboxes(dets_PRED, self.label_colors["PRED"])
        if dets_GT:
            self.update_bboxes(dets_GT, self.label_colors["GT"])
        if dets_FP:
            self.update_bboxes(dets_FP, self.label_colors["FP"])
            


        eye = cam_center + self.cam_offset
        force_camera_pose(self.vis, eye, cam_center, up=(0, 1, 0))
        self.vis.get_view_control().set_zoom(self.zoom)
        self.vis.poll_events()
        self.vis.update_renderer()
