import sys

import cv2
import numpy as np

VIDEO_PATH = "lane.mp4"
WINDOW_SCALE = 2  # source is 320x240, scale windows up for visibility

# ------------------------------------------------------------------ ROI / IPM
# Trapezoid on the floor, as (x, y) fractions of the frame width/height, in
# the order top-left, top-right, bottom-right, bottom-left. Corners may lie
# outside the frame (x < 0 or x > 1): the lanes here are wider apart than the
# frame at the bottom. The sides should point at the vanishing point so that
# parallel lane lines come out parallel in the bird's-eye view.
#
# Fitted to lane.mp4 (static, i.e. best on average over the video):
#   - sides meet at the vanishing point (0.50, 0.13): 0.13 is the horizon
#     height at which lane pairs stay equally spaced far and near (median
#     over ~400 lane-pair intersections); x = 0.50 is the camera axis.
#   - top y = 0.31: highest row that is still floor in 95% of frames.
#   - bottom y = 0.95: lowest row free of the static blob on the vehicle.
#   - half-width: lateral offset = +-2.2 camera heights, which keeps ~98% of
#     lane pixels inside while keeping horizontal resolution.
ROI = [(0.203, 0.31), (0.797, 0.31), (1.853, 0.95), (-0.853, 0.95)]
BEV_SIZE = (320, 320)    # (width, height) of the bird's-eye view the ROI maps to

# ------------------------------------------------------------------ tunables
MIN_HORIZON = 0.0        # never treat anything above this as floor (fraction of height)
MAX_HORIZON = 0.50       # the floor never starts lower than this (fraction of height)
FLOOR_MARGIN = 3         # px kept clear below the detected bottom of walls/objects
HORIZON_BANDS = 8        # column bands used to fit the (possibly tilted) horizon
BG_KERNEL_FRAC = 1 / 12  # background kernel size as a fraction of width (> widest line)
CONTRAST_THRESH = 0.35   # (pixel - local background) / local background
MIN_AREA_FRAC = 1e-4     # minimum blob area as a fraction of the image area
MIN_ASPECT = 1.8         # paint strips are elongated
BIG_AREA_FRAC = 2e-3     # near-field blobs this big are kept regardless of shape
                         # (lane line + crosswalk merged into one L-shape)
NEAR_FRAC = 0.6          # a blob reaching below this height fraction is "near field"
MAX_HALF_WIDTH = 12      # px; in the bird's-eye view paint has ~constant width
THICK_FRAC = 0.2         # blobs with more than this fraction too thick are rejected

# ------------------------------------------------------------------ lane identification
NEAR_START = 0.45        # a lane's nearest point must be below this BEV height fraction
MIN_LANE_LENGTH = 16     # px; shorter strips are floor specks, not lanes
LANE_WIDTH = 120         # px between left and right lane in the BEV (lane.mp4: 109-136)
LANE_WIDTH_TOL = 0.35    # accepted relative deviation from LANE_WIDTH
LEFT_COLOR = (255, 160, 0)   # BGR
RIGHT_COLOR = (0, 80, 255)   # BGR
IGNORED_COLOR = (90, 90, 90)


def floor_top(L):
    """Per-column row where the floor starts.

    Walls, furniture and people are full of near-vertical edges; the floor
    (seen at a grazing angle) has almost none, because far paint lines are
    squashed towards horizontal. Dense vertical-edge regions hanging from the
    top of the frame are therefore "not floor", and the floor starts below them.
    """
    h, w = L.shape
    Lf = cv2.GaussianBlur(L, (3, 3), 0).astype(np.float32)
    gx = cv2.Sobel(Lf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(Lf, cv2.CV_32F, 0, 1)
    vert = ((np.abs(gx) > 40) & (np.abs(gx) > 3 * np.abs(gy))).astype(np.uint8) * 255
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, rect)  # texture -> solid regions

    # Keep regions touching the top edge; isolated marks (a near-vertical
    # stretch of lane line) don't, so they can't pull the horizon down.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(vert)
    hanging = np.isin(lab, [i for i in range(1, n) if stats[i, cv2.CC_STAT_TOP] <= 2])
    top = np.where(hanging, np.arange(h)[:, None], -1).max(axis=0).astype(np.float32) + 1

    # Walls with sparse texture leave gaps, so fit a line through the lowest
    # point of each column band. Theil-Sen (median of pairwise slopes) ignores
    # the odd band where a pole or a leg reaches down further.
    bands = np.array_split(np.arange(w), HORIZON_BANDS)
    bx = np.array([b.mean() for b in bands])
    by = np.array([top[b].max() for b in bands])
    line = np.zeros(w, np.float32)
    ok = by > 0
    if ok.sum() >= 3:
        bx, by = bx[ok], by[ok]
        i, j = np.triu_indices(len(bx), 1)
        slope = np.median((by[j] - by[i]) / (bx[j] - bx[i]))
        line = slope * np.arange(w) + np.median(by - slope * bx)

    # Objects standing on the floor still cut into it locally.
    local = cv2.dilate(top[None, :], np.ones((1, 9), np.uint8))[0]
    top = np.maximum(line, local) + FLOOR_MARGIN
    return np.clip(top, int(h * MIN_HORIZON), int(h * MAX_HORIZON)).astype(int)


def roi_transform(w, h):
    """ROI corners in pixels and the homography mapping them onto the BEV."""
    src = np.float32([(x * w, y * h) for x, y in ROI])
    bw, bh = BEV_SIZE
    dst = np.float32([(0, 0), (bw, 0), (bw, bh), (0, bh)])
    return src, cv2.getPerspectiveTransform(src, dst)


def draw_roi(frame, src):
    """Original frame with the ROI trapezoid outlined (clipped at the frame)."""
    out = frame.copy()
    cv2.polylines(out, [np.round(src).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def process(frame, M):
    """Lane mask in the bird's-eye view of the ROI. Receives the BGR frame."""
    h, w = frame.shape[:2]
    bw, bh = BEV_SIZE

    # 1. Floor detection needs the walls, so it runs on the full frame. The
    #    floor mask is then warped along with the image: where the camera
    #    pitches up and a wall drops into the ROI, it is masked out. Pixels
    #    of the ROI that lie outside the frame are invalid too.
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]
    top = floor_top(L)
    floor = (np.arange(h)[:, None] >= top[None, :]).astype(np.uint8) * 255
    valid = cv2.warpPerspective(floor, M, BEV_SIZE, flags=cv2.INTER_NEAREST)
    # Also drop a strip along the frame border: a line cut by the frame edge
    # runs along it and would merge with the lane line it touches.
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))

    # 2. Work on lightness only. A small median blur removes glare sparkle on
    #    the floor texture. Out-of-frame pixels are filled by replicating the
    #    frame edge so they don't create a fake dark background next to it.
    L = cv2.warpPerspective(
        L, M, BEV_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    L = cv2.medianBlur(L, 3)

    # 3. Estimate the local background brightness with a morphological
    #    opening (removes bright things narrower than the kernel = paint),
    #    then smooth it. Glare is wide, so it stays IN the background.
    k = max(15, int(bw * BG_KERNEL_FRAC) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)
    bg = cv2.GaussianBlur(bg, (k, k), 0).astype(np.float32)

    # 4. Relative contrast: sparkle inside glare is a small step on a bright
    #    background (rejected); paint on dark asphalt is a big step on a dark
    #    background (kept).
    contrast = (L.astype(np.float32) - bg) / (bg + 20.0)
    mask = (contrast > CONTRAST_THRESH).astype(np.uint8) * 255
    mask[valid == 0] = 0

    # 5. Fill small holes inside strips. (No opening: it would erase thin
    #    lines; specks are dropped by the area filter below instead.)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small, iterations=2)

    # 6. Shape filter: keep blobs that look like paint strips. In the
    #    bird's-eye view paint has roughly constant width, so wide blobs
    #    (glare patches, wall bases) are rejected with a fixed limit.
    out = np.zeros_like(mask)
    min_area = MIN_AREA_FRAC * bh * bw
    big_area = BIG_AREA_FRAC * bh * bw
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)  # half-width of strips
    too_thick = dist > MAX_HALF_WIDTH
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        box = np.s_[y : y + ch, x : x + cw]
        blob = lab[box] == i
        if too_thick[box][blob].mean() > THICK_FRAC:
            continue
        # Elongation: the rotated-rect aspect fails on curved strips, so also
        # use length/width estimated as area / width^2.
        contour = cv2.findContours(
            blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0][0]
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        width = max(2 * dist[box][blob].max() - 1, 1.0)
        aspect = max(max(rw, rh) / max(min(rw, rh), 1.0), area / width**2)
        near = y + ch > bh * NEAR_FRAC
        if aspect < MIN_ASPECT and not (area >= big_area and near):
            continue
        out[lab == i] = 255

    return out


def find_lanes(mask):
    """Identify the left and right boundary of the lane the car is in.

    The car sits at the bottom centre of the BEV, looking up along the camera
    axis. Its own lane lines start close to it: they enter the view at the
    bottom or at the side edges of the camera image. Lines of a later section
    of the track either start far ahead, or lie further out than the car's own
    lane line on the same side. So, per side of the camera axis, the lane is
    the strip whose nearest point is closest to the car; every other strip is
    ignored.

    Returns (labels, {"left": label or None, "right": label or None}).
    """
    bh, bw = mask.shape
    cx = bw / 2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    lanes = {"left": None, "right": None}
    cost = {"left": np.inf, "right": np.inf}
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if max(cw, ch) < MIN_LANE_LENGTH:
            continue
        ys, xs = np.nonzero(lab[y : y + ch, x : x + cw] == i)
        y_near = y + ys.max()
        if y_near < NEAR_START * bh:
            continue  # starts far ahead: a later section of the track
        x_near = x + xs[ys >= ys.max() - 3].mean()
        side = "left" if x_near < cx else "right"
        c = abs(x_near - cx) + (bh - 1 - y_near)  # distance to the car
        if c < cost[side]:
            lanes[side], cost[side] = i, c

    # Both found: their spacing must look like a lane, else one of them is a
    # line of another track section. Keep the one that starts closer.
    if lanes["left"] and lanes["right"]:
        L, R = lab == lanes["left"], lab == lanes["right"]
        rows = np.nonzero(L.any(1) & R.any(1))[0]
        rows = rows[rows >= rows.max() - 10] if len(rows) else rows
        if len(rows):
            gap = np.median([R[r].nonzero()[0].mean() - L[r].nonzero()[0].mean() for r in rows])
            if abs(gap - LANE_WIDTH) > LANE_WIDTH_TOL * LANE_WIDTH:
                worse = max(lanes, key=cost.get)
                lanes[worse] = None
    return lab, lanes


def lane_outlines(labels, lanes):
    """Outline polygons of the left / right lane strips, in BEV pixels."""
    out = {}
    for side, i in lanes.items():
        if i is None:
            continue
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out[side] = max(contours, key=cv2.contourArea).astype(np.float32)
    return out


def draw_lanes_bev(mask, labels, lanes):
    """BEV lane mask: left/right lanes in colour, ignored strips in grey."""
    out = np.zeros((*mask.shape, 3), np.uint8)
    out[mask > 0] = IGNORED_COLOR
    for side, color in (("left", LEFT_COLOR), ("right", RIGHT_COLOR)):
        if lanes[side] is not None:
            out[labels == lanes[side]] = color
    for side, c in lane_outlines(labels, lanes).items():
        label_lane(out, side, c)
    return out


def draw_lanes_original(frame, labels, lanes, M_inv):
    """Outline the left/right lanes on the camera image (BEV -> image)."""
    out = frame.copy()
    for side, c in lane_outlines(labels, lanes).items():
        c = cv2.perspectiveTransform(c, M_inv)
        color = LEFT_COLOR if side == "left" else RIGHT_COLOR
        cv2.polylines(out, [np.round(c).astype(np.int32)], True, color, 1, cv2.LINE_AA)
        label_lane(out, side, c)
    return out


def label_lane(img, side, contour):
    """Write L / R next to the lane's point nearest to the car (lowest)."""
    pts = contour.reshape(-1, 2)
    x, y = pts[pts[:, 1].argmax()]
    h, w = img.shape[:2]
    x = int(np.clip(x + (-14 if side == "left" else 6), 2, w - 12))
    y = int(np.clip(y - 6, 12, h - 4))
    color = LEFT_COLOR if side == "left" else RIGHT_COLOR
    cv2.putText(img, "L" if side == "left" else "R", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src, M = roi_transform(w, h)
    M_inv = np.linalg.inv(M)

    windows = {
        "Original + ROI": (w, h),
        "ROI + processing": BEV_SIZE,
    }
    for name, (ww, wh) in windows.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, ww * WINDOW_SCALE, wh * WINDOW_SCALE)

    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:  # end of video: loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            mask = process(frame, M)
            labels, lanes = find_lanes(mask)
            original = draw_lanes_original(draw_roi(frame, src), labels, lanes, M_inv)
            cv2.imshow("Original + ROI", original)
            cv2.imshow("ROI + processing", draw_lanes_bev(mask, labels, lanes))

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
