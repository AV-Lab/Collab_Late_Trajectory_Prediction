#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class VisOcclusionScores:
    """
    Minimal, paper-ready BEV viz — consumes the debug payload from occlusions.compute_l1_occlusion_for_frame.
    No geometry recomputation.
    """
    # aesthetics (Okabe–Ito palette; muted, colorblind-safe)
    colors: Dict[str, str] = field(default_factory=lambda: {
        "ray_clear":   "#009E73",  # teal/green (clear)
        "ray_blocked": "#D55E00",  # vermilion (blocked)
        "box_edge":    "#4D4D4D",
        "ego_edge":    "#009E73",
        "ego_face":    "#BFD3C1",  # pale green
        "text":        "#111111",
        "grid":        "#DDDDDD",
    })
    lw_ray: float = 1.0
    lw_box: float = 1.4
    lw_ego: float = 1.8
    ego_alpha: float = 0.22
    figsize: Tuple[float, float] = (16.0, 9.0)
    dpi: int = 240
    annotate: bool = True   # write occlusion score on each rectangle

    def _draw_poly(self, ax, corners: np.ndarray, edge: str, face: Optional[str], lw: float, alpha: float, z: int):
        poly = patches.Polygon(
            corners, closed=True,
            fill=(face is not None),
            edgecolor=edge,
            facecolor=(face if face else 'none'),
            linewidth=lw, alpha=alpha, zorder=z
        )
        ax.add_patch(poly)

    def render(self, debug: dict, occ_map: Dict[int, float], title: Optional[str] = None, out_path: Optional[str] = None):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        s = debug["origin"]
        rays = debug["rays"]
        corners = debug["corners"]
        ego = debug["ego"]
        bminx, bmaxx, bminy, bmaxy = debug["bounds"]

        # 1) rays (draw first)
        for tid, R in rays.items():
            if R["end"].size == 0:
                continue
            ends = R["end"]          # (M,2)
            blk  = R["blocked"]      # (M,)
            for p, is_blk in zip(ends, blk):
                ax.plot([s[0], p[0]], [s[1], p[1]],
                        color=(self.colors["ray_blocked"] if is_blk else self.colors["ray_clear"]),
                        lw=self.lw_ray, alpha=0.85, solid_capstyle="round", zorder=1)

        # 2) boxes (outline only)
        for oid, cs in corners.items():
            self._draw_poly(ax, cs, edge=self.colors["box_edge"], face=None, lw=self.lw_box, alpha=1.0, z=3)
            if self.annotate and (oid in occ_map):
                cx = float(np.mean(cs[:,0])); cy = float(np.mean(cs[:,1]))
                ax.text(cx, cy, f"{occ_map[oid]:.2f}", ha='center', va='center', fontsize=9, color=self.colors["text"], zorder=4)

        # 3) ego (rectangle + heading arrow)
        ego_cs = ego["corners"]
        self._draw_poly(ax, ego_cs, edge=self.colors["ego_edge"], face=self.colors["ego_face"],
                        lw=self.lw_ego, alpha=self.ego_alpha, z=4)
        ex = float(np.mean(ego_cs[:,0])); ey = float(np.mean(ego_cs[:,1]))
        yaw = float(ego["yaw_rad"])
        ax.arrow(ex, ey, 3.0*math.cos(yaw), 3.0*math.sin(yaw),
                 head_width=0.9, head_length=1.2,
                 fc=self.colors["ego_edge"], ec=self.colors["ego_edge"],
                 length_includes_head=True, zorder=5)

        # 4) axes (EXACT bounds from debug)
        ax.set_xlim(bminx, bmaxx)
        ax.set_ylim(bminy, bmaxy)
        ax.set_aspect('equal', adjustable='box')

        ax.grid(True, ls='--', color=self.colors["grid"], alpha=0.45)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        if title:
            ax.set_title(title)

        if out_path is not None:
            fig.savefig(out_path, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
