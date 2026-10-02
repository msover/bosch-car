import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


VIDEO_PATH = "/home/tudorgiloan/Documents/Etc/Cpp/lane_detection/captures/lane.mp4"
VIDEO_PATH_2 = "/mnt/c/Users/tudor/Downloads/Records/Records/21-09-17-13-00-00.mp4"
WINDOW_SCALE = 2  # source is 320x240, scale windows up for visibility


# ------------------------------------------------------------------ ROI / IPM

# Floor trapezoid as (x, y) fractions of the frame:
# top-left, top-right, bottom-right, bottom-left.
ROI = [
    (0.203, 0.31),
    (0.797, 0.31),
    (1.853, 0.95),
    (-0.853, 0.95),
]

BEV_SIZE = (320, 320)  # (width, height)


# ------------------------------------------------------------------ tunables

BG_KERNEL_FRAC = 1 / 12
CONTRAST_THRESH = 0.35

MIN_AREA_FRAC = 1e-4
MIN_ASPECT = 1.8

BIG_AREA_FRAC = 2e-3
NEAR_FRAC = 0.6

MAX_HALF_WIDTH = 12
THICK_FRAC = 0.2


# ------------------------------------------------------------------ lane identification

NEAR_START = 0.45
MIN_LANE_LENGTH = 16

LANE_WIDTH = 120
LANE_WIDTH_TOL = 0.35

COLORS = {
    "left": (255, 160, 0),
    "right": (0, 80, 255),
}

IGNORED_COLOR = (90, 90, 90)

# Center lane color: green
CENTER_COLOR = (0, 255, 0)


# ------------------------------------------------------------------ vehicle geometry / lookahead

# Set BOTH scales after measuring the ground rectangle represented by the
# BEV homography. The ROI currently defines a visual warp, not a calibrated
# metric map. Unequal lateral/longitudinal scales are supported.
BEV_X_M_PER_PX = None
BEV_Y_M_PER_PX = None

# Pixel location of the rear axle midpoint in the BEV coordinate system.
# This is an initial preview assumption; the real axle can lie BELOW the BEV.
# Axes must be aligned with the vehicle: image-up = forward, image-left = left.
REAR_AXLE_BEV = (BEV_SIZE[0] / 2.0, BEV_SIZE[1] - 1.0)
REAR_AXLE_CALIBRATED = False

# Tuning values, not measured vehicle parameters.
LOOKAHEAD_MIN_M = 0.20
LOOKAHEAD_MAX_M = 0.70
CURVATURE_GAIN_M = 0.40

# Allows visual experimentation before metric calibration.
LOOKAHEAD_MIN_BEV = 45.0
LOOKAHEAD_MAX_BEV = 140.0
CURVATURE_GAIN_BEV = 80.0


@dataclass(frozen=True)
class VehicleGeometry:
    car_u: float
    car_v: float
    scale_x: float
    scale_y: float
    unit: str
    calibrated: bool


@dataclass(frozen=True)
class LookaheadConfig:
    minimum: float
    maximum: float
    curvature_gain: float


@dataclass(frozen=True)
class PursuitTarget:
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


def target_configuration():
    metric = BEV_X_M_PER_PX is not None and BEV_Y_M_PER_PX is not None
    if (BEV_X_M_PER_PX is None) != (BEV_Y_M_PER_PX is None):
        raise ValueError("Set both BEV metre-per-pixel scales, or neither")
    geometry = VehicleGeometry(
        *REAR_AXLE_BEV,
        BEV_X_M_PER_PX if metric else 1.0,
        BEV_Y_M_PER_PX if metric else 1.0,
        "m" if metric else "BEV-unit",
        metric and REAR_AXLE_CALIBRATED,
    )
    config = LookaheadConfig(
        LOOKAHEAD_MIN_M if metric else LOOKAHEAD_MIN_BEV,
        LOOKAHEAD_MAX_M if metric else LOOKAHEAD_MAX_BEV,
        CURVATURE_GAIN_M if metric else CURVATURE_GAIN_BEV,
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


def bev_to_vehicle(u, v, geometry):
    """Coordinates relative to the CAR, not relative to the detected lane."""
    x = (geometry.car_v - np.asarray(v)) * geometry.scale_y
    y = (geometry.car_u - np.asarray(u)) * geometry.scale_x
    return x, y


def vehicle_to_bev(x, y, geometry):
    return (geometry.car_u - np.asarray(y) / geometry.scale_x,
            geometry.car_v - np.asarray(x) / geometry.scale_y)


def center_in_vehicle_frame(center_fit, geometry):
    """Convert BEV u(v)=a*v^2+b*v+c into vehicle y(x)=A*x^2+B*x+C."""
    a, b, c = center_fit
    sx, sy, v0 = geometry.scale_x, geometry.scale_y, geometry.car_v
    return np.array([
        -sx * a / sy**2,
        sx * (2.0 * a * v0 + b) / sy,
        sx * (geometry.car_u - (a * v0**2 + b * v0 + c)),
    ])


def center_curvature(vehicle_fit, x):
    """Unsigned geometric curvature in inverse coordinate units."""
    A, B, _ = vehicle_fit
    return abs(2.0 * A) / (1.0 + (2.0 * A * np.asarray(x) + B)**2)**1.5


def select_pursuit_target(center_fit, y_range, valid, geometry, config):
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

    fit = center_in_vehicle_frame(center_fit, geometry)
    A, B, C = fit

    def visible(x):
        y = np.polyval(fit, x)
        u, v = vehicle_to_bev(x, y, geometry)
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
    curvature = float(np.max(center_curvature(fit, preview_x)))
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
    u, v = vehicle_to_bev(x, y, geometry)
    return PursuitTarget(
        x, y, float(np.hypot(x, y)), requested, curvature,
        float(u), float(v), geometry.unit, geometry.calibrated, limited,
    )


# ------------------------------------------------------------------ perspective transform

def roi_transform(w, h):
    """
    ROI corners in pixels, homography mapping them to the BEV,
    and BEV mask of pixels that come from inside the frame.
    """

    src = np.float32([
        (x * w, y * h)
        for x, y in ROI
    ])

    bw, bh = BEV_SIZE

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
        BEV_SIZE,
        flags=cv2.INTER_NEAREST,
    )

    valid = cv2.erode(
        in_frame,
        np.ones((9, 9), np.uint8),
    )

    return src, M, valid


def warp_roi(frame, M):
    """
    Inverse-perspective mapping:
    ROI as seen from above.
    """

    return cv2.warpPerspective(
        frame,
        M,
        BEV_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_roi(frame, src):
    """
    Copy of the original frame with ROI outlined.
    """

    out = frame.copy()

    cv2.polylines(
        out,
        [np.round(src).astype(np.int32)],
        True,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


# ------------------------------------------------------------------ image processing

def process(bev, valid):
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
        int(bw * BG_KERNEL_FRAC) | 1,
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
        > CONTRAST_THRESH
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

        if area < MIN_AREA_FRAC * bh * bw:
            continue

        box = np.s_[y:y + ch, x:x + cw]

        blob = lab[box] == i

        half_width = dist[box][blob]

        if (half_width > MAX_HALF_WIDTH).mean() > THICK_FRAC:
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
            area >= BIG_AREA_FRAC * bh * bw
            and y + ch > bh * NEAR_FRAC
        )

        keep[i] = (
            aspect >= MIN_ASPECT
            or near_big
        )

    return np.take(
        keep,
        lab,
    ).astype(np.uint8) * 255


# ------------------------------------------------------------------ lane identification

def find_lanes(mask):
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

        if max(cw, ch) < MIN_LANE_LENGTH:
            continue

        ys, xs = np.nonzero(
            lab[y:y + ch, x:x + cw] == i
        )

        # Nearest / lowest point
        y_near = y + ys[-1]

        if y_near < NEAR_START * bh:
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
                abs(gap - LANE_WIDTH)
                > LANE_WIDTH_TOL * LANE_WIDTH
            ):

                worse_side = max(
                    cost,
                    key=cost.get,
                )

                lanes[worse_side] = None

    return lab, lanes


# ------------------------------------------------------------------ quadratic lane fitting

def fit_single_lane(labels, lane_id):
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


def fit_lane_curves(labels, lanes):
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

            center = left + LANE_WIDTH/2

        If only right exists:

            center = right - LANE_WIDTH/2
    """

    fits = {
        "left": None,
        "right": None,
        "center": None,
    }

    fits["left"] = fit_single_lane(
        labels,
        lanes["left"],
    )

    fits["right"] = fit_single_lane(
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
            LANE_WIDTH / 2.0
        )

    # ----------------------------------------------------------
    # Only right boundary visible
    # ----------------------------------------------------------

    elif fits["right"] is not None:

        fits["center"] = (
            fits["right"].copy()
        )

        fits["center"][2] -= (
            LANE_WIDTH / 2.0
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


# ------------------------------------------------------------------ evaluate polynomial

def curve_x(fit, y):
    """
    Evaluate

        x(y) = a*y^2 + b*y + c
    """

    a, b, c = fit

    return (
        a * y * y
        + b * y
        + c
    )


# ------------------------------------------------------------------ curve drawing

def make_curve_points(
    fit,
    height,
    width,
    y_start=0,
    y_end=None,
):
    """
    Convert polynomial x(y) into OpenCV polyline points.
    """

    if fit is None:
        return None

    if y_end is None:
        y_end = height - 1

    ys = np.arange(
        y_start,
        y_end + 1,
        dtype=np.float32,
    )

    a, b, c = fit

    xs = (
        a * ys**2
        + b * ys
        + c
    )

    # Only retain points inside / near the BEV
    valid = (
        (xs >= 0)
        & (xs < width)
    )

    ys = ys[valid]
    xs = xs[valid]

    if len(xs) < 2:
        return None

    points = np.column_stack(
        (xs, ys)
    ).astype(np.float32)

    return points.reshape(
        -1,
        1,
        2,
    )


def draw_center_curve(
    image,
    bev,
    center_fit,
    M_inv,
    target,
    geometry,
):
    """
    Draw the fitted lane center in:

      1. BEV image
      2. Original camera image

    Also marks the curvature-adaptive target selected outside drawing.
    """

    if center_fit is None:
        cv2.putText(bev, "NO VALID TARGET", (8, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CENTER_COLOR, 1, cv2.LINE_AA)
        return

    bh, bw = bev.shape[:2]

    curve = make_curve_points(
        center_fit,
        bh,
        bw,
    )

    if curve is None:
        return

    # ----------------------------------------------------------
    # Draw centerline in BEV
    # ----------------------------------------------------------

    cv2.polylines(
        bev,
        [np.round(curve).astype(np.int32)],
        False,
        CENTER_COLOR,
        2,
        cv2.LINE_AA,
    )

    # ----------------------------------------------------------
    # Transform centerline BEV -> camera
    # ----------------------------------------------------------

    camera_curve = cv2.perspectiveTransform(
        curve,
        M_inv,
    )

    cv2.polylines(
        image,
        [np.round(camera_curve).astype(np.int32)],
        False,
        CENTER_COLOR,
        2,
        cv2.LINE_AA,
    )

    # ----------------------------------------------------------
    # Mark the SAME point whose vehicle coordinates are returned to control.
    # ----------------------------------------------------------

    if target is not None:

        lookahead_x = target.bev_u
        lookahead_y = target.bev_v

        bev_point = np.array(
            [[[lookahead_x, lookahead_y]]],
            dtype=np.float32,
        )

        # BEV point
        bx = int(round(lookahead_x))
        by = int(round(lookahead_y))

        car_point = (int(round(geometry.car_u)), int(round(geometry.car_v)))
        cv2.line(bev, car_point, (bx, by), (0, 200, 200), 1, cv2.LINE_AA)
        if 0 <= car_point[0] < bw and 0 <= car_point[1] < bh:
            cv2.drawMarker(bev, car_point, (0, 200, 200),
                           cv2.MARKER_CROSS, 9, 1)

        cv2.circle(
            bev,
            (bx, by),
            5,
            CENTER_COLOR,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            bev,
            "TARGET",
            (bx + 7, by - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CENTER_COLOR,
            1,
            cv2.LINE_AA,
        )

        # Camera-image point
        camera_point = cv2.perspectiveTransform(
            bev_point,
            M_inv,
        )[0, 0]

        cx = int(
            round(camera_point[0])
        )

        cy = int(
            round(camera_point[1])
        )

        cv2.circle(
            image,
            (cx, cy),
            5,
            CENTER_COLOR,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "TARGET",
            (cx + 7, cy - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            CENTER_COLOR,
            1,
            cv2.LINE_AA,
        )

        status = "calibrated" if target.calibrated else "PREVIEW / uncalibrated"
        digits = 3 if target.unit == "m" else 1
        messages = (
            f"x={target.x:+.{digits}f} y={target.y:+.{digits}f} {target.unit}",
            f"D={target.distance:.3f} k={target.preview_curvature:.4f}",
            status,
        )
        if target.visibility_limited:
            messages += ("target limited by visibility",)
    else:
        messages = ("NO VALID TARGET",)

    for row, message in enumerate(messages):
        cv2.putText(bev, message, (8, 38 + row * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CENTER_COLOR, 1, cv2.LINE_AA)


# ------------------------------------------------------------------ existing lane drawing

def draw_lanes(
    image,
    labels,
    lanes,
    fits,
    M_inv,
    target,
    geometry,
):
    """
    Mark detected lane components and fitted lane center.
    """

    palette = np.zeros(
        (labels.max() + 1, 3),
        np.uint8,
    )

    palette[1:] = IGNORED_COLOR

    for side, i in lanes.items():

        if i is not None:
            palette[i] = COLORS[side]

    bev = np.take(
        palette,
        labels,
        axis=0,
    )

    # ----------------------------------------------------------
    # Draw detected lane blobs
    # ----------------------------------------------------------

    for side, i in lanes.items():

        if i is None:
            continue

        blob = (
            labels == i
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            blob,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        outline = max(
            contours,
            key=cv2.contourArea,
        ).astype(np.float32)

        label_lane(
            bev,
            side,
            outline,
        )

        camera_outline = cv2.perspectiveTransform(
            outline,
            M_inv,
        )

        cv2.polylines(
            image,
            [
                np.round(
                    camera_outline
                ).astype(np.int32)
            ],
            True,
            COLORS[side],
            1,
            cv2.LINE_AA,
        )

        label_lane(
            image,
            side,
            camera_outline,
        )

    # ----------------------------------------------------------
    # Draw fitted quadratic center
    # ----------------------------------------------------------

    draw_center_curve(
        image,
        bev,
        fits["center"],
        M_inv,
        target,
        geometry,
    )

    return image, bev


# ------------------------------------------------------------------ labels

def label_lane(
    img,
    side,
    outline,
):
    """
    Write L / R beside the lane's nearest point.
    """

    pts = outline.reshape(
        -1,
        2,
    )

    x, y = pts[
        pts[:, 1].argmax()
    ]

    h, w = img.shape[:2]

    x = int(
        np.clip(
            x + (
                -14
                if side == "left"
                else 6
            ),
            2,
            w - 12,
        )
    )

    y = int(
        np.clip(
            y - 6,
            12,
            h - 4,
        )
    )

    cv2.putText(
        img,
        side[0].upper(),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLORS[side],
        1,
        cv2.LINE_AA,
    )


# ------------------------------------------------------------------ debugging / output

def print_fits(fits):
    """
    Print fitted polynomial equations.
    """

    for name in (
        "left",
        "right",
        "center",
    ):

        fit = fits[name]

        if fit is None:
            continue

        a, b, c = fit

        print(
            f"{name:6s}: "
            f"x(y) = "
            f"{a:+.7f} y^2 "
            f"{b:+.5f} y "
            f"{c:+.2f}"
        )


# ------------------------------------------------------------------ main

def main():

    geometry, lookahead_config = target_configuration()
    if not geometry.calibrated:
        print("PREVIEW: calibrate the BEV ground mapping, both scales, and rear "
              "axle position before using targets to steer the car.")

    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else VIDEO_PATH
    )

    cap = cv2.VideoCapture(
        path
    )

    if not cap.isOpened():
        sys.exit(
            f"Could not open {path}"
        )

    period = 1 / (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30
    )

    w = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    h = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    src, M, valid = roi_transform(
        w,
        h,
    )

    M_inv = np.linalg.inv(
        M
    )

    # ----------------------------------------------------------
    # Windows
    # ----------------------------------------------------------

    for name, (ww, wh) in (
        (
            "Original + ROI",
            (w, h),
        ),
        (
            "ROI + processing",
            BEV_SIZE,
        ),
    ):

        cv2.namedWindow(
            name,
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            name,
            ww * WINDOW_SCALE,
            wh * WINDOW_SCALE,
        )

    paused = False

    # Prevent console output every single frame
    frame_number = 0

    while True:

        start = time.perf_counter()

        if not paused:

            ok, frame = cap.read()

            if not ok:
                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    0,
                )
                continue

            # --------------------------------------------------
            # 1. Perspective transform
            # --------------------------------------------------

            roi = warp_roi(
                frame,
                M,
            )

            # --------------------------------------------------
            # 2. Detect painted strips
            # --------------------------------------------------

            mask = process(
                roi,
                valid,
            )

            # --------------------------------------------------
            # 3. Select left/right lanes
            # --------------------------------------------------

            labels, lanes = find_lanes(
                mask
            )

            # --------------------------------------------------
            # 4. Quadratic fitting + center
            # --------------------------------------------------

            fits = fit_lane_curves(
                labels,
                lanes,
            )

            # --------------------------------------------------
            # 5. Curvature-adaptive target in the vehicle coordinate frame
            # --------------------------------------------------

            target = select_pursuit_target(
                fits["center"], fits["center_y_range"], valid,
                geometry, lookahead_config,
            )

            # Later, pass target.x and target.y to pure pursuit, with the
            # wheelbase in the SAME units. target=None means no usable target.
            # Physical control should require target.calibrated == True.

            # --------------------------------------------------
            # 6. Draw everything
            # --------------------------------------------------

            image, bev = draw_lanes(
                draw_roi(
                    frame,
                    src,
                ),
                labels,
                lanes,
                fits,
                M_inv,
                target,
                geometry,
            )

            # --------------------------------------------------
            # Display current lateral error
            # --------------------------------------------------

            center_fit = fits["center"]

            if center_fit is not None:

                # Evaluate center at bottom of BEV
                y_car = BEV_SIZE[1] - 1

                x_center = curve_x(
                    center_fit,
                    y_car,
                )

                car_axis = (
                    geometry.car_u
                )

                error = (
                    x_center
                    - car_axis
                )

                cv2.putText(
                    bev,
                    f"near-row offset right: {error:+.1f}px",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    CENTER_COLOR,
                    1,
                    cv2.LINE_AA,
                )

            # Print equations occasionally
            if frame_number % 30 == 0:
                print("\n--- quadratic fits ---")
                print_fits(fits)
                if target is None:
                    print("target: unavailable")
                else:
                    print(
                        f"target: x={target.x:+.4f}, y={target.y:+.4f} "
                        f"{target.unit}; distance={target.distance:.4f}; "
                        f"requested={target.requested_distance:.4f}; "
                        f"curvature={target.preview_curvature:.5f} "
                        f"1/{target.unit}; calibrated={target.calibrated}; "
                        f"visibility_limited={target.visibility_limited}"
                    )

            frame_number += 1

            cv2.imshow(
                "Original + ROI",
                image,
            )

            cv2.imshow(
                "ROI + processing",
                bev,
            )

        # ------------------------------------------------------
        # FPS timing
        # ------------------------------------------------------

        wait_ms = int(
            (
                period
                - (
                    time.perf_counter()
                    - start
                )
            )
            * 1000
        )

        key = cv2.waitKey(
            max(1, wait_ms)
        ) & 0xFF

        if key in (
            ord("q"),
            27,
        ):
            break

        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

