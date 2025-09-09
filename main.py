from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import numpy as np
import cv2
import math
import base64

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@api.get("/")
def index():
    return {"message": "Hello World"}

def to_data_url_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")

def fill_holes(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, inv)

def feret_features(points_xy: np.ndarray):
    """
    Fiji-like Feret metrics.
    points_xy: Nx2 float array (convex hull points recommended)
    Returns: (feret_max, feret_min, feret_angle_deg, feretX, feretY)
    """
    pts = points_xy.astype(np.float32)
    if pts.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Max Feret (diameter) via brute-force on hull
    A = pts[:, None, :]
    d2 = np.sum((A - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, -1.0)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    feret_max = float(np.sqrt(d2[i, j]))
    feret_angle = float(np.degrees(np.arctan2(pts[j, 1] - pts[i, 1], pts[j, 0] - pts[i, 0])))

    # Min Feret = short side of min-area rectangle (rotating calipers width)
    rect = cv2.minAreaRect(pts)             # ((cx,cy),(w,h),angle)
    w_rect, h_rect = float(rect[1][0]), float(rect[1][1])
    feret_min = float(min(w_rect, h_rect))

    return feret_max, feret_min, feret_angle, float(pts[i, 0]), float(pts[i, 1])

@api.post("/analyze")
async def analyze(
    file: UploadFile = File(...),

    # --- Fiji-like defaults & knobs ---
    sigma_pre: float = Query(6.4, ge=0.0),                 # Gaussian sigma before threshold
    k: int = Query(5, ge=3),                               # morphology kernel (odd recommended)
    dilate_iter: int = Query(4, ge=0),                     # Fiji: Dilate x4
    erode_iter: int = Query(5, ge=0),                      # then Erode x5
    min_area: float = Query(60000, ge=0),                  # Fiji script default
    max_area: float = Query(2.0e7, ge=0),
    min_circ: float = Query(0.28, ge=0.0, le=1.0),
    edge_margin: float = Query(0.20, ge=0.0, le=0.49),     # exclude particles near borders
    smooth_circle: bool = Query(False)                     # optional: force circular contour
) -> Dict[str, Any]:

    # 0) Load and crop 2px border (Fiji crops to avoid border artifacts)
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    if min(img.shape[:2]) > 8:
        img = img[2:-2, 2:-2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Blur like Fiji
    gray_blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma_pre, sigmaY=sigma_pre)

    # 2) Threshold dark organoid (Fiji: "Default dark"; here: Otsu + INV)
    _, core = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) Morphology sequence (Fiji order)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = fill_holes(core)
    if dilate_iter > 0:
        core = cv2.dilate(core, kernel, iterations=dilate_iter)
    core = fill_holes(core)
    if erode_iter > 0:
        core = cv2.erode(core, kernel, iterations=erode_iter)
    core = fill_holes(core)

    # 4) Contours + Fiji-like filtering (area, circ, edge exclusion)
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise HTTPException(422, "No contours found")

    H, W = gray.shape
    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

    def circ_of(c):
        a = cv2.contourArea(c)
        p = cv2.arcLength(c, True)
        return ((4.0 * math.pi * a) / (p * p)) if p > 0 else 0.0

    candidates = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        if circ_of(c) < min_circ:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx_c = M["m10"] / M["m00"]
        cy_c = M["m01"] / M["m00"]
        if (cx_c < xmin) or (cx_c > xmax) or (cy_c < ymin) or (cy_c > ymax):
            continue
        candidates.append(c)

    cnt = max(candidates, key=cv2.contourArea) if candidates else max(cnts, key=cv2.contourArea)

    # Optional: replace with a smooth circle (not used by Fiji, but kept as a knob)
    if smooth_circle:
        (ccx, ccy), r = cv2.minEnclosingCircle(cnt)
        cnt = cv2.ellipse2Poly(center=(int(ccx), int(ccy)),
                               axes=(int(r), int(r)), angle=0, arcStart=0, arcEnd=360, delta=1)
        cnt = cnt.reshape(-1, 1, 2)

    # Final mask
    mask = np.zeros_like(gray, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    # ---- Measurements (geometry)
    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    bx, by, width, height = float(x), float(y), float(w), float(h)

    M = cv2.moments(cnt)
    cx = float(M["m10"] / M["m00"]) if M["m00"] else 0.0
    cy = float(M["m01"] / M["m00"]) if M["m00"] else 0.0

    major = minor = angle = 0.0
    if len(cnt) >= 5:
        (_, _), (eW, eH), ang = cv2.fitEllipse(cnt)
        major, minor, angle = float(max(eW, eH)), float(min(eW, eH)), float(ang)

    circ = float((4.0 * math.pi * area) / (perim * perim)) if perim > 0 else 0.0

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Proper Feret & MinFeret (use convex hull points)
    hull_pts = hull.reshape(-1, 2).astype(np.float64)
    feret, minFeret, feretAngle, feretX, feretY = feret_features(hull_pts)

    # Fiji-style AR and Roundness (ellipse-based, not axis-aligned box)
    ar = float(major / minor) if minor > 0 else 0.0
    roundness = float((4.0 * area) / (math.pi * major * major)) if major > 0 else 0.0

    # ---- Intensity stats on INVERTED gray (Fiji inverts brightfield)
    stat_img = 255 - gray
    vals = stat_img[mask > 0].astype(np.float64)
    if vals.size == 0:
        raise HTTPException(422, "Empty ROI after masking")

    mean = float(vals.mean())
    stdDev = float(vals.std(ddof=0))
    vmin = float(vals.min())
    vmax = float(vals.max())
    median = float(np.median(vals))
    hist, _ = np.histogram(vals, bins=256, range=(0, 256))
    mode = float(int(hist.argmax()))
    if stdDev > 0:
        z = (vals - mean) / stdDev
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    else:
        skew = kurt = 0.0
    rawIntDen = float(vals.sum())
    intDen = float(area * mean)

    # Intensity-weighted centroid on inverted plane (XM, YM)
    ys, xs = np.nonzero(mask)
    weights = stat_img[ys, xs].astype(np.float64)
    wsum = float(weights.sum())
    if wsum > 0:
        xm = float((xs * weights).sum() / wsum)
        ym = float((ys * weights).sum() / wsum)
    else:
        xm, ym = cx, cy

    # ---- Visuals
    overlay = img.copy()
    cv2.drawContours(overlay, [cnt], -1, (255, 0, 255), 8)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    results = {
        "area": area, "mean": mean, "stdDev": stdDev, "mode": mode, "min": vmin, "max": vmax,
        "x": cx, "y": cy, "xm": xm, "ym": ym, "perim": perim, "bx": bx, "by": by,
        "width": width, "height": height, "major": major, "minor": minor, "angle": angle,
        "circ": circ, "feret": feret, "intDen": intDen, "median": median, "skew": skew,
        "kurt": kurt, "rawIntDen": rawIntDen, "feretX": feretX, "feretY": feretY,
        "feretAngle": feretAngle, "minFeret": minFeret, "ar": ar, "round": roundness,
        "solidity": solidity,
    }

    return {
        "results": results,
        "roi_image": to_data_url_png(overlay),
        "mask_image": to_data_url_png(mask_vis),
    }
