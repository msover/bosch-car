"""Frame-based lane detection with a curvature-adaptive centerline target.

Usage:
    detector = WaypointManager()
    coords = detector.getCurrent(frame)  # nonempty uint8 BGR image
    if coords is not None:
        x, y = coords  # x forward; y left; origin = rear axle midpoint

The default geometry uses BEV units and an assumed rear axle position.
To get metres, calibrate the ground homography (the ROI below), pass BOTH
metre-per-pixel scales and the measured rear axle position to the constructor.
The BEV axes must align with the vehicle axes. The rear axle may lie below
the BEV image. Lookahead defaults are tuning values, not vehicle measurements.

The module performs no capture, display, playback, or printing. The caller
owns those tasks. Requires Python 3.10+, NumPy, and OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["WaypointManager"]


@dataclass(frozen=True)
class _VehicleGeometry:
    car_u: float
    car_v: float
    scale_x: float
    scale_y: float
    unit: str
    calibrated: bool


@dataclass(frozen=True)
class _LookaheadConfig:
    minimum: float
    maximum: float
    curvature_gain: float


@dataclass(frozen=True)
class _PursuitTarget:
    # Vehicle coordinates: origin = rear axle midpoint, +x forward, +y left.
    x: float
    y: float
    # Straight-line distance; use THIS value if visibility limits selection.
    distance: float
    requested_distance: float
    preview_curvature: float
    bev_u: float
    bev_v: float
    unit: str
    calibrated: bool
    visibility_limited: bool



class WaypointManager:
    """Detect a lane-center target from each camera frame.

    `getCurrent(frame)` is the only public processing method. It returns a tuple
    `(x, y)` of floats or None if no visible, supported target is available.
    Coordinates are in metres when both scales are supplied; otherwise they
    are BEV units. +x is forward and +y is left from the rear axle midpoint.

    Use a separate detector instance for each processing thread/camera.
    """

    # Detection and BEV tunables. Calibrate the ROI for your camera.
    _ROI = [
        (0.203, 0.31),
        (0.797, 0.31),
        (1.853, 0.95),
        (-0.853, 0.95),
    ]

    _BEV_SIZE = (320, 320)  # (width, height)

    _BG_KERNEL_FRAC = 1 / 12

    _CONTRAST_THRESH = 0.35

    _MIN_AREA_FRAC = 1e-4

    _MIN_ASPECT = 1.8

    _BIG_AREA_FRAC = 2e-3

    _NEAR_FRAC = 0.6

    _MAX_HALF_WIDTH = 12

    _THICK_FRAC = 0.2

    _NEAR_START = 0.45

    _MIN_LANE_LENGTH = 16

    _LANE_WIDTH = 120

    _LANE_WIDTH_TOL = 0.35

    _LOOKAHEAD_MIN_M = 0.20

    _LOOKAHEAD_MAX_M = 0.70

    _CURVATURE_GAIN_M = 0.40

    _LOOKAHEAD_MIN_BEV = 45.0

    _LOOKAHEAD_MAX_BEV = 140.0

    _CURVATURE_GAIN_BEV = 80.0


    def __init__(
        self,
        *,
        bev_x_m_per_px: float | None = None,
        bev_y_m_per_px: float | None = None,
        rear_axle_bev: tuple[float, float] | None = None,
        rear_axle_calibrated: bool = False,
    ) -> None:
        """Configure ground scales and rear-axle position once per detector.

        Supply both scales or neither. `rear_axle_calibrated` documents whether
        the origin was measured; it does not itself calibrate the homography.
        The caller must finish calibration before using coordinates for driving.
        """
        self._bev_x_m_per_px = bev_x_m_per_px
        self._bev_y_m_per_px = bev_y_m_per_px
        self._rear_axle_bev = (
            (self._BEV_SIZE[0] / 2.0, self._BEV_SIZE[1] - 1.0)
            if rear_axle_bev is None
            else rear_axle_bev
        )
        if len(self._rear_axle_bev) != 2:
            raise ValueError("rear_axle_bev must contain (u, v)")
        self._rear_axle_calibrated = rear_axle_calibrated
        self._geometry, self._lookahead = self._target_configuration()
        self._frame_size = None
        self._transform = None
        self._valid_mask = None

    def getCurrent(self, frame: np.ndarray) -> tuple[float, float] | None:
        """Process a nonempty uint8 BGR frame without modifying it.

        Returns:
            `(x, y)` for the curvature-adaptive lane-center target, measured
            from the rear axle midpoint (+x forward, +y left), or None when
            lane detection or target selection cannot provide a valid target.

        Raises:
            ValueError: the frame is not a nonempty HxWx3 uint8 BGR array.
        """
        if (not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.shape[0] == 0
                or frame.shape[1] == 0):
            raise ValueError("process requires a nonempty HxWx3 uint8 BGR frame")

        height, width = frame.shape[:2]
        frame_size = (width, height)
        if frame_size != self._frame_size:
            _, transform, valid = self._roi_transform(width, height)
            self._transform = transform
            self._valid_mask = valid
            self._frame_size = frame_size

        bev = self._warp_roi(frame, self._transform)
        mask = self._extract_mask(bev, self._valid_mask)
        labels, lanes = self._find_lanes(mask)
        fits = self._fit_lane_curves(labels, lanes)
        target = self._select_pursuit_target(
            fits["center"], fits["center_y_range"], self._valid_mask,
            self._geometry, self._lookahead,
        )
        return None if target is None else (target.x, target.y)

    def _target_configuration(self):
        metric = self._bev_x_m_per_px is not None and self._bev_y_m_per_px is not None
        if (self._bev_x_m_per_px is None) != (self._bev_y_m_per_px is None):
            raise ValueError("Set both BEV metre-per-pixel scales, or neither")
        geometry = _VehicleGeometry(
            *self._rear_axle_bev,
            self._bev_x_m_per_px if metric else 1.0,
            self._bev_y_m_per_px if metric else 1.0,
            "m" if metric else "BEV-unit",
            metric and self._rear_axle_calibrated,
        )
        config = _LookaheadConfig(
            self._LOOKAHEAD_MIN_M if metric else self._LOOKAHEAD_MIN_BEV,
            self._LOOKAHEAD_MAX_M if metric else self._LOOKAHEAD_MAX_BEV,
            self._CURVATURE_GAIN_M if metric else self._CURVATURE_GAIN_BEV,
        )
        values = [geometry.car_u, geometry.car_v, geometry.scale_x,
                  geometry.scale_y, config.minimum, config.maximum,
                  config.curvature_gain]
        if not np.all(np.isfinite(values)):
            raise ValueError("Geometry and lookahead values must be finite")
        if geometry.scale_x <= 0 or geometry.scale_y <= 0:
            raise ValueError("BEV scales must be positive")
        if not 0 < config.minimum <= config.maximum or config.curvature_gain < 0:
            raise ValueError("Invalid lookahead bounds or curvature gain")
        return geometry, config


    def _vehicle_to_bev(self, x, y, geometry):
        return (geometry.car_u - np.asarray(y) / geometry.scale_x,
                geometry.car_v - np.asarray(x) / geometry.scale_y)


    def _center_in_vehicle_frame(self, center_fit, geometry):
        """Convert BEV u(v)=a*v^2+b*v+c into vehicle y(x)=A*x^2+B*x+C."""
        a, b, c = center_fit
        sx, sy, v0 = geometry.scale_x, geometry.scale_y, geometry.car_v
        return np.array([
            -sx * a / sy**2,
            sx * (2.0 * a * v0 + b) / sy,
            sx * (geometry.car_u - (a * v0**2 + b * v0 + c)),
        ])


    def _center_curvature(self, vehicle_fit, x):
        """Unsigned geometric curvature in inverse coordinate units."""
        A, B, _ = vehicle_fit
        return abs(2.0 * A) / (1.0 + (2.0 * A * np.asarray(x) + B)**2)**1.5


    def _select_pursuit_target(self, center_fit, y_range, valid, geometry, config):
        """Select a visible, supported centerline target; return None on failure.

        1. Measure maximum centerline curvature over the available near preview.
        2. D = clip(D_max / (1 + gain * curvature), D_min, D_max).
        3. Intersect the center polynomial with x^2+y^2=D^2.

        The farthest-forward valid intersection is chosen if there are several.
        If that circle is outside the observed path, choose the visible point
        closest to D within the configured distance bounds. Report its ACTUAL
        distance. Never extrapolate beyond detected lane rows or the valid BEV.
        """
        if center_fit is None or y_range is None:
            return None
        if not np.all(np.isfinite(center_fit)):
            return None
        bh, bw = valid.shape
        v_min = max(0.0, float(y_range[0]))
        v_max = min(bh - 1.0, float(y_range[1]))
        if v_min > v_max:
            return None

        fit = self._center_in_vehicle_frame(center_fit, geometry)
        A, B, C = fit

        def visible(x):
            y = np.polyval(fit, x)
            u, v = self._vehicle_to_bev(x, y, geometry)
            inside = ((x > 0) & (u >= 0) & (u < bw)
                      & (v >= v_min) & (v <= v_max))
            # Clip only mask indices; never clip/alter the target coordinates.
            ui = np.clip(np.rint(u), 0, bw - 1).astype(int)
            vi = np.clip(np.rint(v), 0, bh - 1).astype(int)
            return inside & (valid[vi, ui] != 0)

        # Dense samples supply the fallback and support the curvature preview.
        vs = np.linspace(v_max, v_min, max(512, bh * 2))
        xs = (geometry.car_v - vs) * geometry.scale_y
        xs = xs[visible(xs)]
        if not len(xs):
            return None

        preview_x = xs[xs <= config.maximum]
        if not len(preview_x):
            return None
        # Include the exact maximum-curvature location for a quadratic, if visible.
        if A != 0:
            vertex_x = -B / (2.0 * A)
            if (preview_x.min() <= vertex_x <= preview_x.max()
                    and bool(visible(np.asarray(vertex_x)))):
                preview_x = np.append(preview_x, vertex_x)
        curvature = float(np.max(self._center_curvature(fit, preview_x)))
        requested = float(np.clip(
            config.maximum / (1.0 + config.curvature_gain * curvature),
            config.minimum, config.maximum,
        ))

        # x^2 + (A*x^2+B*x+C)^2 - D^2 = 0 (quartic, or lower degree).
        coefficients = np.trim_zeros(np.array([
            A*A, 2*A*B, 1+B*B+2*A*C, 2*B*C, C*C-requested*requested,
        ]), "f")
        roots = np.roots(coefficients)
        real_x = roots.real[np.abs(roots.imag) <= 1e-7 * (1 + np.abs(roots.real))]
        candidates = real_x[visible(real_x)]
        limited = not len(candidates)
        if limited:
            distances = np.hypot(xs, np.polyval(fit, xs))
            usable = (distances >= config.minimum) & (distances <= config.maximum)
            if not usable.any():
                return None
            xs, distances = xs[usable], distances[usable]
            x = float(xs[np.argmin(np.abs(distances - requested))])
        else:
            x = float(candidates.max())

        y = float(np.polyval(fit, x))
        u, v = self._vehicle_to_bev(x, y, geometry)
        return _PursuitTarget(
            x, y, float(np.hypot(x, y)), requested, curvature,
            float(u), float(v), geometry.unit, geometry.calibrated, limited,
        )


    def _roi_transform(self, w, h):
        """
        self._ROI corners in pixels, homography mapping them to the BEV,
        and BEV mask of pixels that come from inside the frame.
        """

        src = np.float32([
            (x * w, y * h)
            for x, y in self._ROI
        ])

        bw, bh = self._BEV_SIZE

        dst = np.float32([
            (0, 0),
            (bw, 0),
            (bw, bh),
            (0, bh),
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        in_frame = cv2.warpPerspective(
            np.full((h, w), 255, np.uint8),
            M,
            self._BEV_SIZE,
            flags=cv2.INTER_NEAREST,
        )

        valid = cv2.erode(
            in_frame,
            np.ones((9, 9), np.uint8),
        )

        return src, M, valid


    def _warp_roi(self, frame, M):
        """
        Inverse-perspective mapping:
        self._ROI as seen from above.
        """

        return cv2.warpPerspective(
            frame,
            M,
            self._BEV_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


    def _extract_mask(self, bev, valid):
        """
        Generate lane-marking mask from BEV image.
        """

        bh, bw = bev.shape[:2]

        # ----------------------------------------------------------
        # 1. Lightness channel
        # ----------------------------------------------------------

        L = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)[:, :, 1]
        L = cv2.medianBlur(L, 3)

        # ----------------------------------------------------------
        # 2. Estimate local background
        # ----------------------------------------------------------

        k = max(
            15,
            int(bw * self._BG_KERNEL_FRAC) | 1,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (k, k),
        )

        bg = cv2.morphologyEx(
            L,
            cv2.MORPH_OPEN,
            kernel,
            borderType=cv2.BORDER_REPLICATE,
        )

        bg = cv2.GaussianBlur(
            bg,
            (k, k),
            0,
        ).astype(np.float32)

        # ----------------------------------------------------------
        # 3. Relative contrast
        # ----------------------------------------------------------

        mask = (
            (L - bg) / (bg + 20.0)
            > self._CONTRAST_THRESH
        ).astype(np.uint8) * 255

        mask &= valid

        # ----------------------------------------------------------
        # 4. Fill holes
        # ----------------------------------------------------------

        small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            small,
            iterations=2,
        )

        # ----------------------------------------------------------
        # 5. Shape filter
        # ----------------------------------------------------------

        dist = cv2.distanceTransform(
            mask,
            cv2.DIST_L2,
            3,
        )

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)

        keep = np.zeros(n, bool)

        for i in range(1, n):

            x, y, cw, ch, area = stats[i]

            if area < self._MIN_AREA_FRAC * bh * bw:
                continue

            box = np.s_[y:y + ch, x:x + cw]

            blob = lab[box] == i

            half_width = dist[box][blob]

            if (half_width > self._MAX_HALF_WIDTH).mean() > self._THICK_FRAC:
                continue

            contour = cv2.findContours(
                blob.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0][0]

            rw, rh = cv2.minAreaRect(contour)[1]

            width = max(
                2 * half_width.max() - 1,
                1.0,
            )

            aspect = max(
                max(rw, rh) / max(min(rw, rh), 1.0),
                area / width**2,
            )

            near_big = (
                area >= self._BIG_AREA_FRAC * bh * bw
                and y + ch > bh * self._NEAR_FRAC
            )

            keep[i] = (
                aspect >= self._MIN_ASPECT
                or near_big
            )

        return np.take(
            keep,
            lab,
        ).astype(np.uint8) * 255


    def _find_lanes(self, mask):
        """
        Identify the left and right lane boundaries.
        """

        bh, bw = mask.shape

        cx = bw / 2

        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)

        lanes = {
            "left": None,
            "right": None,
        }

        cost = {
            "left": np.inf,
            "right": np.inf,
        }

        for i in range(1, n):

            x, y, cw, ch, _ = stats[i]

            if max(cw, ch) < self._MIN_LANE_LENGTH:
                continue

            ys, xs = np.nonzero(
                lab[y:y + ch, x:x + cw] == i
            )

            # Nearest / lowest point
            y_near = y + ys[-1]

            if y_near < self._NEAR_START * bh:
                continue

            x_near = x + xs[
                ys >= ys[-1] - 3
            ].mean()

            side = (
                "left"
                if x_near < cx
                else "right"
            )

            c = (
                abs(x_near - cx)
                + (bh - 1 - y_near)
            )

            if c < cost[side]:
                lanes[side] = i
                cost[side] = c

        # ----------------------------------------------------------
        # Validate left/right spacing
        # ----------------------------------------------------------

        if (
            lanes["left"] is not None
            and lanes["right"] is not None
        ):

            L = lab == lanes["left"]
            R = lab == lanes["right"]

            rows = np.nonzero(
                L.any(1) & R.any(1)
            )[0]

            if len(rows):

                rows = rows[
                    rows >= rows[-1] - 10
                ]

                gap = np.median([
                    R[r].nonzero()[0].mean()
                    - L[r].nonzero()[0].mean()
                    for r in rows
                ])

                if (
                    abs(gap - self._LANE_WIDTH)
                    > self._LANE_WIDTH_TOL * self._LANE_WIDTH
                ):

                    worse_side = max(
                        cost,
                        key=cost.get,
                    )

                    lanes[worse_side] = None

        return lab, lanes


    def _fit_single_lane(self, labels, lane_id):
        """
        Fit one detected lane using

            x(y) = a*y^2 + b*y + c

        Each BEV row contributes one point: the mean x-position
        of the detected painted strip in that row.

        This prevents wide parts of the paint strip from receiving
        more weight than narrow parts.
        """

        if lane_id is None:
            return None

        ys, xs = np.nonzero(
            labels == lane_id
        )

        if len(xs) < 3:
            return None

        unique_y = np.unique(ys)

        fit_y = []
        fit_x = []

        for y in unique_y:

            row_x = xs[ys == y]

            if len(row_x) == 0:
                continue

            fit_y.append(y)

            # Center of painted strip in this row
            fit_x.append(
                row_x.mean()
            )

        if len(fit_y) < 3:
            return None

        fit_y = np.asarray(
            fit_y,
            dtype=np.float64,
        )

        fit_x = np.asarray(
            fit_x,
            dtype=np.float64,
        )

        # x = a*y^2 + b*y + c
        return np.polyfit(
            fit_y,
            fit_x,
            2,
        )


    def _fit_lane_curves(self, labels, lanes):
        """
        Fit quadratics to left/right lane boundaries and compute
        the lane-center polynomial.

        Returned format:

            {
                "left":   [a, b, c] or None,
                "right":  [a, b, c] or None,
                "center": [a, b, c] or None,
                "center_y_range": (first_row, last_row) or None,
            }

        Center calculation:

            If both lanes exist:

                center = (left + right) / 2

            If only left exists:

                center = left + self._LANE_WIDTH/2

            If only right exists:

                center = right - self._LANE_WIDTH/2
        """

        fits = {
            "left": None,
            "right": None,
            "center": None,
        }

        fits["left"] = self._fit_single_lane(
            labels,
            lanes["left"],
        )

        fits["right"] = self._fit_single_lane(
            labels,
            lanes["right"],
        )

        # ----------------------------------------------------------
        # Both boundaries visible
        # ----------------------------------------------------------

        if (
            fits["left"] is not None
            and fits["right"] is not None
        ):

            fits["center"] = (
                fits["left"]
                + fits["right"]
            ) / 2.0

        # ----------------------------------------------------------
        # Only left boundary visible
        # ----------------------------------------------------------

        elif fits["left"] is not None:

            fits["center"] = (
                fits["left"].copy()
            )

            # Shift x by half a lane width
            fits["center"][2] += (
                self._LANE_WIDTH / 2.0
            )

        # ----------------------------------------------------------
        # Only right boundary visible
        # ----------------------------------------------------------

        elif fits["right"] is not None:

            fits["center"] = (
                fits["right"].copy()
            )

            fits["center"][2] -= (
                self._LANE_WIDTH / 2.0
            )

        # A center fitted from two boundaries is supported only over their overlap.
        # A center inferred from one boundary uses that boundary's observed rows.
        ranges = []
        for side in ("left", "right"):
            if fits[side] is not None:
                rows = np.nonzero((labels == lanes[side]).any(axis=1))[0]
                ranges.append((int(rows[0]), int(rows[-1])))
        fits["center_y_range"] = None
        if ranges:
            y_min = max(r[0] for r in ranges)
            y_max = min(r[1] for r in ranges)
            if y_min <= y_max:
                fits["center_y_range"] = (y_min, y_max)

        return fits
