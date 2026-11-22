"""Depth noise of the Fetch head camera, after Nguyen, Izadi and Lovell (2012)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthNoise:
    axial_base: float = .0012 # m
    axial_quad: float = .0019 # m per m^2 of (z - axial_ref)^2
    axial_ref: float = .4 # m
    lateral_px: float = .8 # pixels
    steep_deg: float = 60. # surfaces steeper than this (to the ray) lose returns
    steep_drop: float = .5 # probability that a steep pixel returns nothing
    quantum: float = .0015 # m at 1 m depth, scaling with z^2
    scale: float = 1. # multiplies every sigma
    random_drop: float = 0. # share of pixels returning nothing at random

    def axial_sigma(self, z: np.ndarray) -> np.ndarray:
        return self.scale * (self.axial_base + self.axial_quad * (np.asarray(z, float) - self.axial_ref) ** 2)

    def apply(self, depth_m: np.ndarray, rng: np.random.Generator, normals_angle_deg: np.ndarray | None = None,
              invalid: float = np.nan) -> np.ndarray:
        """A noisy copy of a metric depth image (rows x columns). Pixels with no return are ``invalid``."""
        z = np.asarray(depth_m, float)
        h, w = z.shape
        out = z.copy()
        if self.lateral_px > 0 and self.scale > 0:
            dv = np.rint(rng.normal(0., self.scale * self.lateral_px, (h, w))).astype(int)
            du = np.rint(rng.normal(0., self.scale * self.lateral_px, (h, w))).astype(int)
            rows = np.clip(np.arange(h)[:, None] + dv, 0, h - 1)
            cols = np.clip(np.arange(w)[None, :] + du, 0, w - 1)
            out = out[rows, cols]
        out = out + rng.normal(0., 1., (h, w)) * self.axial_sigma(out)
        if self.quantum > 0:
            step = self.quantum * np.maximum(out, 1e-3) ** 2
            out = np.round(out / step) * step
        if normals_angle_deg is not None:
            steep = (np.asarray(normals_angle_deg) > self.steep_deg) & (rng.random((h, w)) < self.steep_drop)
            out[steep] = invalid
        if self.random_drop > 0:
            out[rng.random((h, w)) < self.random_drop] = invalid
        return out
