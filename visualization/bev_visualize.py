"""Collaborative top-down scene and selected-frame PNG/PDF/SVG export."""

from colorsys import hls_to_rgb
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, CheckButtons
import numpy as np

from visualization.geometry import box_corners, covariance_ellipse, lidar_world_xy, world_to_bev


class BEVVisualizer:
    LAYERS = {
        "LiDAR": False, "Detections": False, "L/S boxes": True, "History": True,
        "Ego predictions": True, "Shared predictions": True,
        "Fused predictions": True, "Ground truth": True,
        "IDs": False, "Evaluation labels": False,
    }
    FUSED_COLOR = "#228b22"
    GT_COLOR = "#303030"

    def __init__(self, output_dir, lidar_loader=None, *, vehicle_names=(), interactive=True):
        if interactive and plt.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
            raise RuntimeError("--viz 2d requires an interactive Matplotlib backend, such as TkAgg.")
        self.output_dir = Path(output_dir)
        self.lidar_loader = lidar_loader
        self.vehicle_names = tuple(vehicle_names)
        self.interactive = interactive
        self.layers = dict(self.LAYERS)
        self.covariance_sources = set()
        self.scene = None
        self.closed = False
        self.paused = True
        self._step = False
        self._clouds = {}
        self._source_names = ()
        self._source_colors = {}
        self._fit_pending = True
        self.limits = ((-55.0, 55.0), (-55.0, 55.0))
        self.figure, self.ax = plt.subplots(figsize=(12, 9))
        self.figure.subplots_adjust(left=.08, right=.75, bottom=.16, top=.88)
        self.figure.canvas.manager.set_window_title("CoLTP — Collaborative BEV")
        self.title = self.figure.suptitle("CoLTP — collaborative trajectories", fontsize=13)
        self.status = self.figure.text(.055, .07, "Space: play/pause · Right: next frame · S: save · Scroll: zoom", fontsize=9)
        self._controls = []
        self._layer_ax = self.figure.add_axes([.79, .46, .20, .45])
        self._layer_ax.set_title("Layers", loc="left", fontsize=10)
        self._layers = CheckButtons(self._layer_ax, list(self.layers), list(self.layers.values()))
        self._layers.on_clicked(self._toggle_layer)
        self._cov_ax = self.figure.add_axes([.79, .13, .20, .27])
        for position, label, callback in ((.055, "Play", self._play), (.14, "Next frame", self._next),
                                           (.25, "Save frame", self._save), (.37, "Fit view", self._fit)):
            button = Button(self.figure.add_axes([position, .025, .08 if label != "Save frame" else .1, .035]), label)
            button.on_clicked(callback)
            self._controls.append(button)
        self.figure.canvas.mpl_connect("key_press_event", self._key)
        self.figure.canvas.mpl_connect("close_event", self._closed)
        self.figure.canvas.mpl_connect("scroll_event", self._zoom)
        if interactive:
            plt.show(block=False)

    def reset(self):
        self.scene = None
        self._clouds.clear()
        self._fit_pending = True
        self.limits = ((-55.0, 55.0), (-55.0, 55.0))

    def _toggle_layer(self, label):
        self.layers[label] = not self.layers[label]
        if label == "LiDAR" and not self.layers[label]:
            self._clouds.clear()
        self._redraw()

    def _toggle_covariance(self, name):
        if name in self.covariance_sources:
            self.covariance_sources.remove(name)
        else:
            self.covariance_sources.add(name)
        self._redraw()

    def _play(self, event=None):
        self.paused = not self.paused
        self._controls[0].label.set_text("Play" if self.paused else "Pause")
        self.figure.canvas.draw_idle()

    def _next(self, event=None):
        self.paused, self._step = True, True
        self._controls[0].label.set_text("Play")

    def _save(self, event=None):
        if self.scene is not None:
            destination = self.save_frame()
            self.status.set_text(f"Saved: {destination}")
            self.figure.canvas.draw_idle()

    def _key(self, event):
        if event.key == " ":
            self._play()
        elif event.key == "right":
            self._next()
        elif event.key in {"s", "S"}:
            self._save()

    def _closed(self, event):
        self.closed = True

    def _zoom(self, event):
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        scale = .8 if event.button == "up" else 1.25
        self.limits = tuple(tuple(center + (value - center) * scale for value in bounds)
                            for bounds, center in zip(self.limits, (event.xdata, event.ydata)))
        self._redraw()

    def _configure_sources(self):
        ego = self.scene["ego_name"]
        sources = set(self.scene["vehicles"]) | set(self.scene["received_vehicle_poses"])
        for obj in self.scene["vehicles"][ego]["objects"]:
            sources.update(f["source_vehicle"] for f in obj["shared"] if f["source_vehicle"] is not None)
        names = (ego, *sorted(sources - {ego}))
        self._source_colors[ego] = "#d55e00"
        palette = ("#0072b2", "#9467bd", "#8c564b", "#17becf", "#e377c2", "#bcbd22")
        for name in (*self.vehicle_names, *names):
            if name not in self._source_colors:
                index = len(self._source_colors) - 1
                color = (palette[index] if index < len(palette)
                         else to_hex(hls_to_rgb((index * .618034) % 1, .4, .7)))
                self._source_colors[name] = color
        if names != self._source_names:
            if hasattr(self, "_covariances"):
                self._covariances.disconnect_events()
            self._cov_ax.clear()
            self._cov_ax.set_title("Covariance by source (1σ)", loc="left", fontsize=10)
            self._covariances = CheckButtons(self._cov_ax, names, [name in self.covariance_sources for name in names])
            self._covariances.on_clicked(self._toggle_covariance)
            for label in self._covariances.labels:
                label.set_color(self._color(label.get_text()))
            self._source_names = names

    def _fit(self, event=None):
        if self.scene is None:
            return
        vehicle = self.scene["vehicles"][self.scene["ego_name"]]
        points = [self.scene["origin"].reshape(1, 2)]
        for obj in vehicle["objects"]:
            if self.layers["L/S boxes"]:
                points.append(obj["box"][:2].reshape(1, 2))
            if self.layers["History"]:
                points.append(obj["past"])
            for visible, forecasts in ((self.layers["Ego predictions"], [obj["ego"]]),
                                       (self.layers["Fused predictions"], [obj["fused"]]),
                                       (self.layers["Shared predictions"], obj["shared"])):
                if visible:
                    points.extend(forecast["xy"] for forecast in forecasts if forecast is not None)
        if self.layers["Ground truth"]:
            for truth in vehicle["ground_truth"]:
                points.extend((truth["box"][:2].reshape(1, 2), truth["future"]))
                if self.layers["History"]:
                    points.append(truth["past"])
        for record in self.scene["received_vehicle_poses"].values():
            points.append(np.array([[record["pose"]["x"], record["pose"]["y"]]]))
        points = world_to_bev(np.vstack(points), self.scene["origin"])
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2
            half_width = np.maximum((high - low) / 2 + 4, 10)
            self.limits = tuple(zip(center - half_width, center + half_width))
        self._fit_pending = False
        self._redraw()

    def update(self, scene):
        """Display one frame; pause stops the simulation until play or next frame."""
        if self.closed:
            return False
        self.scene = scene
        # Cache only clouds belonging to the current ego/collaborator responses.
        valid = {(name, vehicle["timestamp_ms"]) for name, vehicle in scene["vehicles"].items()}
        self._clouds = {key: value for key, value in self._clouds.items() if key in valid}
        self._configure_sources()
        if self._fit_pending:
            self._fit()
        self._redraw()
        if self.interactive:
            plt.pause(.025)
            while self.paused and not self._step and not self.closed:
                plt.pause(.05)
            self._step = False
        return not self.closed

    def _color(self, source):
        return self._source_colors.get(source, "#0072b2")

    def _cloud(self, name):
        vehicle = self.scene["vehicles"][name]
        key = (name, vehicle["timestamp_ms"])
        if key not in self._clouds:
            points = vehicle["point_cloud"]
            if points is None and vehicle["lidar_path"] and self.lidar_loader is not None:
                points = self.lidar_loader(vehicle["lidar_path"])
            if points is None:
                self._clouds[key] = None
            else:
                # Downsample display only; never modify sensor/predictor input.
                stride = max(1, int(np.ceil(len(points) / 30000)))
                self._clouds[key] = lidar_world_xy(points[::stride], vehicle["calibration"])
        return self._clouds[key]

    def _box(self, ax, box, color, label, *, dashed=False, width=1, filled=False):
        corners = box_corners(box)
        if not len(corners):
            return
        points = world_to_bev(corners, self.scene["origin"])
        ax.add_patch(Polygon(points, fill=filled, facecolor=to_rgba(color, .08), edgecolor=color, linewidth=width,
                             linestyle="--" if dashed else "-", label=label, zorder=3))
        center = world_to_bev(np.asarray(box)[:2], self.scene["origin"])
        heading = (points[0] + points[1]) / 2
        ax.plot([center[0], heading[0]], [center[1], heading[1]], color=color, linewidth=.7, zorder=3)

    def _line(self, ax, xy, color, label, style="-", width=1.5, *, steps=False):
        if len(xy):
            points = world_to_bev(xy, self.scene["origin"])
            ax.plot(points[:, 0], points[:, 1], style, color=color, linewidth=width, label=label,
                    marker="." if steps else None, markersize=3.5, zorder=4)

    def _forecast(self, ax, forecast, label, *, shared=False):
        if forecast is None:
            return
        source, covariance = forecast["source_vehicle"], forecast["cov"]
        color = self._color(source)
        self._line(ax, forecast["xy"], color, label, "--" if shared else "-", steps=True)
        if covariance is not None and source in self.covariance_sources:
            ellipses = [covariance_ellipse(xy, cov) for xy, cov in zip(forecast["xy"], covariance)]
            segments = [world_to_bev(ellipse, self.scene["origin"]) for ellipse in ellipses if ellipse is not None]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=.6, alpha=.3, zorder=2))
        if shared and self.layers["IDs"]:
            point = world_to_bev(forecast["xy"][0], self.scene["origin"])
            ax.annotate(f"{source}/{forecast['source_object_id']}", point, fontsize=6, color=color)

    def _vehicle_marker(self, ax, name, pose, timestamp_ms):
        if not pose or not all(key in pose for key in ("x", "y", "yaw")):
            return
        point = world_to_bev([pose["x"], pose["y"]], self.scene["origin"])
        is_ego = name == self.scene["ego_name"]
        color = self.FUSED_COLOR if is_ego else self._color(name)
        ax.plot(*point, marker="*" if is_ego else "o", color=color,
                markersize=10, markeredgecolor="white", markeredgewidth=.6, zorder=6)
        direction = np.array([np.cos(pose["yaw"]), np.sin(pose["yaw"])]) * 3
        ax.annotate("", point + direction, point, arrowprops={"arrowstyle": "->", "color": color}, zorder=6)
        age = (self.scene["timestamp_ms"] - timestamp_ms) / 1000
        label = "ego" if is_ego else name + (f" (pose {age:.2f}s old)" if age else "")
        ax.annotate(label, point, xytext=(5, 6), textcoords="offset points", fontsize=7, color=color, zorder=6)

    def _draw_scene(self, ax):
        name = self.scene["ego_name"]
        ax.clear()
        ax.set(xlim=self.limits[0], ylim=self.limits[1], aspect="equal",
               xlabel="World x relative to ego (m)", ylabel="World y relative to ego (m)")
        ax.grid(alpha=.2, linewidth=.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        vehicle = self.scene["vehicles"][name]
        if self.layers["LiDAR"]:
            cloud = self._cloud(name)
            if cloud is not None:
                points = world_to_bev(cloud, self.scene["origin"])
                ax.scatter(points[:, 0], points[:, 1], s=.4, c="#888888", alpha=.25,
                           rasterized=True, zorder=0)
            else:
                ax.text(.02, .02, "LiDAR unavailable", transform=ax.transAxes, fontsize=7)
        if self.layers["Detections"]:
            for box in vehicle["detections"]:
                self._box(ax, box, "#aaaaaa", "Detection", width=.7)
        if self.layers["Ground truth"]:
            for truth in vehicle["ground_truth"]:
                missed = self.layers["Evaluation labels"] and truth.get("status") == "missed"
                color = "#cc9900" if missed else self.GT_COLOR
                self._box(ax, truth["box"], color, "Missed GT" if missed else "GT box", dashed=True)
                self._line(ax, truth["future"], self.GT_COLOR, "GT future", ":", steps=True)
                if self.layers["History"]:
                    self._line(ax, truth["past"], "#888888", "GT history", ":", .7)
        for obj in vehicle["objects"]:
            kind = "L" if obj["node_type"] == 1 else "S"
            source = obj["shared"][0]["source_vehicle"] if kind == "S" and obj["shared"] else name
            color = self._color(source)
            if self.layers["Evaluation labels"] and obj.get("status") == "unmatched":
                color = "#cc9900"
            if self.layers["L/S boxes"]:
                self._box(ax, obj["box"], color, f"{kind} node", dashed=kind == "S", width=1.5, filled=True)
            if self.layers["History"]:
                self._line(ax, obj["past"], color, "Tracked history", ":", .8)
            if self.layers["Ego predictions"]:
                self._forecast(ax, obj["ego"], f"{name} prediction")
            if self.layers["Shared predictions"]:
                for forecast in obj["shared"]:
                    self._forecast(ax, forecast, f"{forecast['source_vehicle']} received prediction", shared=True)
            if self.layers["Fused predictions"] and obj["fused"] is not None:
                self._line(ax, obj["fused"]["xy"], self.FUSED_COLOR, "Fused prediction", width=2.3, steps=True)
            if self.layers["IDs"] or self.layers["Evaluation labels"]:
                label = f"{kind} {obj['id']}"
                if self.layers["Evaluation labels"] and obj.get("status"):
                    label += f" · {obj['status']}"
                ax.annotate(label, world_to_bev(obj["box"][:2], self.scene["origin"]), fontsize=6, color=color)
        self._vehicle_marker(ax, name, vehicle["pose"], vehicle["timestamp_ms"])
        for sender, record in self.scene["received_vehicle_poses"].items():
            if sender != name:
                self._vehicle_marker(ax, sender, record["pose"], record["timestamp_ms"])
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8, framealpha=.9)

    def _redraw(self):
        if self.scene is None or self.closed:
            return
        ego = self.scene["ego_name"]
        self._draw_scene(self.ax)
        covariance = ", ".join(sorted(self.covariance_sources)) or "off"
        self.title.set_text(f"{self.scene['scenario']} · {ego} collaboration · t={self.scene['timestamp_ms'] / 1000:.2f}s\n"
                            f"Covariance: {covariance} (1σ) · solid L / dashed S · receiver GT")
        self.figure.canvas.draw_idle()

    def save_frame(self):
        """Export the current collaboration view without controls."""
        if self.scene is None:
            raise ValueError("No visualization frame is available to save.")
        destination = self.output_dir / self.scene["scenario"] / str(self.scene["timestamp_ms"])
        destination.mkdir(parents=True, exist_ok=True)
        figure, ax = plt.subplots(figsize=(10, 8))
        try:
            self._draw_scene(ax)
            figure.suptitle(self.title.get_text(), fontsize=11)
            figure.tight_layout()
            for extension in ("png", "pdf", "svg"):
                figure.savefig(destination / f"combined.{extension}", dpi=300)
        finally:
            plt.close(figure)
        metadata = {
            "scenario": self.scene["scenario"], "timestamp_ms": self.scene["timestamp_ms"],
            "ego_vehicle": self.scene["ego_name"], "source_colors": self._source_colors,
            "world_origin_xy_m": self.scene["origin"].tolist(),
            "evaluation_iou_threshold": self.scene["iou_threshold"], "layers": self.layers,
            "covariance_sources": sorted(self.covariance_sources), "covariance_sigma_scale": 1,
            "vehicle_timestamps_ms": {name: value["timestamp_ms"] for name, value in self.scene["vehicles"].items()},
            "received_vehicle_timestamps_ms": {name: value["timestamp_ms"] for name, value in self.scene["received_vehicle_poses"].items()},
        }
        (destination / "frame.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return destination

    def close(self):
        self.closed = True
        self.scene = None
        self._clouds.clear()
        plt.close(self.figure)
