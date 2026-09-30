import sys
import time

import cv2
import numpy as np

VIDEO_PATH = "lane.mp4"
WINDOW_SCALE = 2  # source is 320x240, scale windows up for visibility

# ------------------------------------------------------------------ ROI / IPM
# Floor trapezoid as (x, y) fractions of the frame: top-left, top-right,
# bottom-right, bottom-left. Corners may lie outside the frame: the lanes are
# wider apart than the frame at the bottom. Fitted to lane.mp4 (static):
#   - sides meet at the vanishing point (0.50, 0.13): the horizon height at
#     which lane pairs stay equally spaced far and near; x = camera axis.
#   - top y = 0.31: highest row that is still floor in 95% of frames.
#   - bottom y = 0.95: lowest row free of the static blob on the vehicle.
#   - half-width +-2.2 camera heights: keeps ~98% of lane pixels inside.
ROI = [(0.203, 0.31), (0.797, 0.31), (1.853, 0.95), (-0.853, 0.95)]
BEV_SIZE = (320, 320)    # (width, height) of the bird's-eye view (BEV)

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
COLORS = {"left": (255, 160, 0), "right": (0, 80, 255)}  # BGR
IGNORED_COLOR = (90, 90, 90)


def floor_top(L):
    """Per-column row where the floor starts.

    Walls, furniture and people are full of near-vertical edges; the floor
    (seen at a grazing angle) has almost none. Dense vertical-edge regions
    hanging from the top of the frame are "not floor"; the floor is below them.
    """
    h, w = L.shape
    Lb = cv2.GaussianBlur(L, (3, 3), 0)
    gx = np.abs(cv2.Sobel(Lb, cv2.CV_16S, 1, 0))
    gy = np.abs(cv2.Sobel(Lb, cv2.CV_16S, 0, 1))
    vert = ((gx > 40) & (gx > 3 * gy)).astype(np.uint8)
    vert = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))  # texture -> solid

    # Only regions touching the top edge count, so an isolated near-vertical
    # stretch of lane line can't pull the horizon down.
    _, lab, stats, _ = cv2.connectedComponentsWithStats(vert)
    hangs = stats[:, cv2.CC_STAT_TOP] <= 2
    hangs[0] = False  # background
    hanging = np.take(hangs, lab)  # np.take: much faster than hangs[lab] here
    top = np.where(hanging.any(0), h - hanging[::-1].argmax(0), 0).astype(np.float32)

    # Sparse wall texture leaves gaps: fit a line through the lowest point of
    # each column band. Theil-Sen (median of pairwise slopes) ignores the odd
    # band where a pole or a leg reaches down further.
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
    return src, cv2.getPerspectiveTransform(src, np.float32([(0, 0), (bw, 0), (bw, bh), (0, bh)]))


def draw_roi(frame, src):
    """Copy of the frame with the ROI trapezoid outlined (clipped at the frame)."""
    out = frame.copy()
    cv2.polylines(out, [np.round(src).astype(np.int32)], True, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def process(frame, M):
    """Lane mask (0/255) in the bird's-eye view of the ROI, from a BGR frame."""
    h = frame.shape[0]
    bw, bh = BEV_SIZE

    # 1. Floor detection needs the walls, so it runs on the full frame; the
    #    floor mask is warped along with the image, so a wall that drops into
    #    the ROI (camera pitching up) is masked out, as are out-of-frame pixels.
    #    Eroding also drops a strip along the frame border: a line cut by the
    #    frame edge runs along it and would merge with the lane line it touches.
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)[:, :, 1]
    floor = (np.arange(h)[:, None] >= floor_top(L)).astype(np.uint8) * 255
    valid = cv2.warpPerspective(floor, M, BEV_SIZE, flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8))

    # 2. Lightness only; a small median blur removes glare sparkle. Replicating
    #    the frame edge into out-of-frame pixels avoids a fake dark background.
    L = cv2.warpPerspective(L, M, BEV_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    L = cv2.medianBlur(L, 3)

    # 3. Local background: an opening removes bright things narrower than the
    #    kernel (paint); glare is wide, so it stays in the background.
    k = max(15, int(bw * BG_KERNEL_FRAC) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)
    bg = cv2.GaussianBlur(bg, (k, k), 0).astype(np.float32)

    # 4. Relative contrast: sparkle in glare is a small step on a bright
    #    background (rejected), paint on dark floor a big step on a dark one.
    mask = ((L - bg) / (bg + 20.0) > CONTRAST_THRESH).astype(np.uint8) * 255
    mask &= valid

    # 5. Fill small holes inside strips. (No opening: it would erase thin
    #    lines; specks are dropped by the area filter below instead.)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small, iterations=2)

    # 6. Shape filter: keep blobs that look like paint strips. Paint has
    #    ~constant width in the BEV, so wide blobs (glare, wall bases) are out.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)  # half-width of strips
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < MIN_AREA_FRAC * bh * bw:
            continue
        box = np.s_[y : y + ch, x : x + cw]
        blob = lab[box] == i
        half_width = dist[box][blob]
        if (half_width > MAX_HALF_WIDTH).mean() > THICK_FRAC:
            continue
        # Elongation: the rotated-rect aspect fails on curved strips, so also
        # use length/width estimated as area / width^2.
        contour = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        rw, rh = cv2.minAreaRect(contour)[1]
        width = max(2 * half_width.max() - 1, 1.0)
        aspect = max(max(rw, rh) / max(min(rw, rh), 1.0), area / width**2)
        near_big = area >= BIG_AREA_FRAC * bh * bw and y + ch > bh * NEAR_FRAC
        keep[i] = aspect >= MIN_ASPECT or near_big
    return np.take(keep, lab).astype(np.uint8) * 255


def find_lanes(mask):
    """Identify the left and right boundary of the lane the car is in.

    The car sits at the bottom centre of the BEV, looking up the camera axis.
    Its own lane lines start close to it (entering at the bottom or the side
    edges of the camera image); lines of a later section of the track start far
    ahead, or lie further out than the car's own line on the same side. So per
    side of the axis, the lane is the strip whose nearest point is closest to
    the car, and every other strip is ignored.

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
        y_near = y + ys[-1]  # nonzero() is row-major, so the last row is the lowest
        if y_near < NEAR_START * bh:
            continue  # starts far ahead: a later section of the track
        x_near = x + xs[ys >= ys[-1] - 3].mean()
        side = "left" if x_near < cx else "right"
        c = abs(x_near - cx) + (bh - 1 - y_near)  # distance to the car
        if c < cost[side]:
            lanes[side], cost[side] = i, c

    # Both found: their spacing must look like a lane, else one of them is a
    # line of another track section. Keep the one that starts closer.
    if lanes["left"] and lanes["right"]:
        L, R = lab == lanes["left"], lab == lanes["right"]
        rows = np.nonzero(L.any(1) & R.any(1))[0]
        if len(rows):
            rows = rows[rows >= rows[-1] - 10]
            gap = np.median([R[r].nonzero()[0].mean() - L[r].nonzero()[0].mean() for r in rows])
            if abs(gap - LANE_WIDTH) > LANE_WIDTH_TOL * LANE_WIDTH:
                lanes[max(cost, key=cost.get)] = None
    return lab, lanes


def draw_lanes(image, labels, lanes, M_inv):
    """Mark the lanes: outlined on the camera image (drawn in place) and coloured
    in the BEV mask, where ignored strips are grey. Returns (image, bev)."""
    palette = np.zeros((labels.max() + 1, 3), np.uint8)
    palette[1:] = IGNORED_COLOR
    for side, i in lanes.items():
        if i is not None:
            palette[i] = COLORS[side]
    bev = np.take(palette, labels, axis=0)

    for side, i in lanes.items():
        if i is None:
            continue
        blob = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea).astype(np.float32)
        label_lane(bev, side, outline)
        outline = cv2.perspectiveTransform(outline, M_inv)  # BEV -> camera image
        cv2.polylines(image, [np.round(outline).astype(np.int32)], True, COLORS[side], 1, cv2.LINE_AA)
        label_lane(image, side, outline)
    return image, bev


def label_lane(img, side, outline):
    """Write L / R beside the lane's point nearest to the car (the lowest)."""
    pts = outline.reshape(-1, 2)
    x, y = pts[pts[:, 1].argmax()]
    h, w = img.shape[:2]
    x = int(np.clip(x + (-14 if side == "left" else 6), 2, w - 12))
    y = int(np.clip(y - 6, 12, h - 4))
    cv2.putText(img, side[0].upper(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS[side], 1, cv2.LINE_AA)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open {path}")

    period = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src, M = roi_transform(w, h)
    M_inv = np.linalg.inv(M)

    for name, (ww, wh) in (("Original + ROI", (w, h)), ("ROI + processing", BEV_SIZE)):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, ww * WINDOW_SCALE, wh * WINDOW_SCALE)

    paused = False
    while True:
        start = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok:  # end of video: loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            image, bev = draw_lanes(draw_roi(frame, src), *find_lanes(process(frame, M)), M_inv)
            cv2.imshow("Original + ROI", image)
            cv2.imshow("ROI + processing", bev)

        # Wait only for what is left of the frame period: plays at the video's fps.
        wait_ms = int((period - (time.perf_counter() - start)) * 1000)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        if key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
