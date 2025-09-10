from __future__ import annotations

import base64
import io
import logging
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, Dict

import imagej
import jpype
import numpy as np
import scyjava
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import color, filters, measure, morphology, segmentation
from skimage.morphology import square, disk
from scyjava import jimport

# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@api.get("/")
def index():
    return {"message": "Hello World"}

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log = logging.getLogger("orgprofiler")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# PyImageJ initialization
# -----------------------------------------------------------------------------
# Use core ImageJ by default (works cleanly on JDK 11/17). For full Fiji set:
#   export IMAGEJ_COORD='sc.fiji:fiji:2.14.0'
IMAGEJ_COORD = os.environ.get("IMAGEJ_COORD", "net.imagej:imagej:2.14.0")

def _detect_java_home() -> str | None:
    """Prefer an existing JAVA_HOME, otherwise ask macOS for JDK 17 then 11."""
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isdir(jh):
        return jh
    for v in ("17", "11"):
        try:
            out = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", v], text=True
            ).strip()
            if out and os.path.isdir(out):
                return out
        except Exception:
            pass
    return None

@lru_cache(maxsize=1)
def _init_ij():
    java_home = _detect_java_home()
    log.info("Initializing ImageJ…")
    log.info("Using JAVA_HOME: %s", java_home)
    if not java_home:
        raise RuntimeError(
            "No Java detected. Install OpenJDK 17 or 11 and set JAVA_HOME."
        )
    os.environ["JAVA_HOME"] = java_home
    os.environ.setdefault("SCYJAVA_FETCH_JAVA", "never")

    # JVM options BEFORE imagej.init()
    scyjava.config.add_option("-Xmx4g")
    scyjava.config.add_option("-Djava.awt.headless=true")

    log.info("Starting JVM… (coord=%s)", IMAGEJ_COORD)
    ij = imagej.init(IMAGEJ_COORD, mode="headless")
    log.info("ImageJ version: %s", ij.getVersion())

    # Use scyjava.jimport rather than ij.py.get_jclass
    IJ = jimport("ij.IJ")
    return ij, IJ

def get_ij():
    ij, IJ = _init_ij()
    if not jpype.isThreadAttachedToJVM():
        jpype.attachThreadToJVM()
    return ij, IJ

# -----------------------------------------------------------------------------
# Utilities (no OpenCV)
# -----------------------------------------------------------------------------
def to_data_url_png(arr: np.ndarray) -> str:
    """Encode HxW or HxWx3 uint8 array as PNG data URL via Pillow."""
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        im = Image.fromarray(a, mode="L")
    else:
        im = Image.fromarray(a, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def feret_features_from_points(points_xy: np.ndarray):
    """
    Feret max/min via convex hull.
    points_xy: Nx2 (x, y) float
    Returns: (feret_max, minFeret, feret_angle_deg, feretX, feretY)
    """
    if points_xy.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hull = ConvexHull(points_xy)
    H = points_xy[hull.vertices]  # Kx2 ordered hull

    # Max Feret (diameter) on hull
    d2max, i_best, j_best = 0.0, 0, 0
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            dx, dy = H[j, 0] - H[i, 0], H[j, 1] - H[i, 1]
            d2 = dx * dx + dy * dy
            if d2 > d2max:
                d2max, i_best, j_best = d2, i, j
    feret_max = math.sqrt(d2max)
    feret_angle = math.degrees(
        math.atan2(H[j_best, 1] - H[i_best, 1], H[j_best, 0] - H[i_best, 0])
    )
    feretX, feretY = float(H[i_best, 0]), float(H[i_best, 1])

    # Min Feret = minimum width across hull edges
    def width_for_edge(p0, p1):
        ux, uy = p1 - p0
        L = math.hypot(float(ux), float(uy))
        if L == 0:
            return float("inf")
        nx, ny = -uy / L, ux / L
        projs = H @ np.array([nx, ny], dtype=float)
        return float(projs.max() - projs.min())

    min_width = float("inf")
    for i in range(len(H)):
        p0, p1 = H[i], H[(i + 1) % len(H)]
        w = width_for_edge(p0, p1)
        if w < min_width:
            min_width = w

    return float(feret_max), float(min_width), float(feret_angle), feretX, feretY

# -----------------------------------------------------------------------------
# Fiji-like mask builder in scikit-image (matches macro semantics)
# -----------------------------------------------------------------------------
def build_mask_fiji_like(img_rgb: np.ndarray, sigma_pre: float,
                         dilate_iter: int, erode_iter: int) -> np.ndarray:
    """
    Reproduce the macro logic:

    1) gray -> AutoThreshold 'Default' (IsoData)
    2) Fill Holes -> Dilate x d -> Fill -> Erode x e -> Fill
    3) Gaussian Blur (on the *mask*) sigma = sigma_pre
    4) AutoThreshold 'Default dark' -> final mask (bool)
    """
    # 8-bit gray
    gray = (color.rgb2gray(img_rgb) * 255.0).astype(np.uint8)

    # 1) first threshold (‘Default’ = IsoData)
    t1 = filters.threshold_isodata(gray)
    core = gray <= t1  # dark objects

    # 2) morphology like ImageJ's "Dilate/Erode" (3x3 neighborhood)
    core = ndi.binary_fill_holes(core)
    se = square(3)
    for _ in range(int(dilate_iter)):
        core = morphology.binary_dilation(core, se)
    core = ndi.binary_fill_holes(core)
    for _ in range(int(erode_iter)):
        core = morphology.binary_erosion(core, se)
    core = ndi.binary_fill_holes(core)

    # 3) blur the *mask* (macro does this)
    soft = filters.gaussian(core.astype(float), sigma=sigma_pre, preserve_range=True)

    # 4) ‘Default dark’ on blurred mask
    t2 = filters.threshold_isodata((soft * 255).astype(np.uint8)) / 255.0
    final = soft <= t2
    return final

# -----------------------------------------------------------------------------
# Analyze endpoint
# -----------------------------------------------------------------------------
@api.post("/analyze")
async def analyze(
    file: UploadFile = File(...),

    # Fiji-like knobs
    sigma_pre: float = Query(6.4, ge=0.0),          # Gaussian blur sigma
    dilate_iter: int = Query(4, ge=0),              # Dilate x N
    erode_iter: int = Query(5, ge=0),               # Erode x M
    min_area: float = Query(60000, ge=0),           # area bounds
    max_area: float = Query(2.0e7, ge=0),
    min_circ: float = Query(0.28, ge=0.0, le=1.0),  # circularity filter
    edge_margin: float = Query(0.20, ge=0.0, le=0.49),  # exclude centroids near borders
) -> Dict[str, Any]:

    data = await file.read()
    try:
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Invalid image")

    # Crop 2 px like the macro does
    if min(img.shape[:2]) > 8:
        img = img[2:-2, 2:-2, :]

    H, W, _ = img.shape

    # ---------- Try ImageJ path ----------
    try:
        ij, IJ = get_ij()
        log.info("[IJ] Opening temp image for processing…")
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                Image.fromarray(img).save(tmp.name)
                tmp_path = tmp.name

            imp = IJ.openImage(tmp_path)
            if imp is None:
                raise RuntimeError("IJ.openImage returned None")

            log.info("[IJ] Running macro-equivalent steps…")
            IJ.run(imp, "8-bit", "")
            IJ.setAutoThreshold(imp, "Default")
            IJ.run(imp, "Convert to Mask", "")
            IJ.run(imp, "Fill Holes", "")
            for _ in range(int(dilate_iter)):
                IJ.run(imp, "Dilate", "")
            IJ.run(imp, "Fill Holes", "")
            for _ in range(int(erode_iter)):
                IJ.run(imp, "Erode", "")
            IJ.run(imp, "Fill Holes", "")
            IJ.run(imp, "Gaussian Blur...", f"sigma={sigma_pre}")
            IJ.setAutoThreshold(imp, "Default dark")
            IJ.run(imp, "Convert to Mask", "")

            # Convert ImagePlus -> numpy
            mask_u8 = ij.py.from_java(imp)  # expect HxW uint8
            if mask_u8.ndim != 2:
                mask_u8 = mask_u8[..., 0]
            mask_bool = mask_u8 > 0
            log.info("[IJ] Mask built: shape=%s, foreground=%d px",
                     mask_bool.shape, int(mask_bool.sum()))
        finally:
            try:
                if "imp" in locals() and imp is not None:
                    imp.changes = False
                    imp.close()
            except Exception:
                pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    except Exception as e:
        # ---------- scikit-image fallback that mirrors the macro ----------
        log.warning("Operating in headless mode - the original ImageJ will have limited functionality.")
        log.error("[IJ] ImageJ path FAILED; falling back to scikit-image.")
        log.exception(e)

        log.info("[SK] Building mask with Fiji-like pipeline…")
        mask_bool = build_mask_fiji_like(
            img_rgb=img,
            sigma_pre=sigma_pre,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
        )
        log.info("[SK] Mask built: foreground=%d px", int(mask_bool.sum()))

    # -------- Filter regions (area, circularity, edge), then UNION --------
    labels = measure.label(mask_bool, connectivity=2)
    if labels.max() == 0:
        raise HTTPException(422, "No contours found")

    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

    keep_mask = np.zeros_like(labels, dtype=bool)
    for r in measure.regionprops(labels):
        a = float(r.area)
        if a < min_area or a > max_area:
            continue
        p = float(r.perimeter) if r.perimeter > 0 else 0.0
        circ = (4.0 * math.pi * a) / (p * p) if p > 0 else 0.0
        if circ < min_circ:
            continue
        cy, cx = r.centroid  # (row, col)
        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
            continue
        keep_mask |= labels == r.label

    if not keep_mask.any():
        # Fallback: keep the largest region
        largest = max(measure.regionprops(labels), key=lambda rr: rr.area)
        keep_mask = labels == largest.label

    # -------- Visual overlays --------
    boundaries = segmentation.find_boundaries(keep_mask, mode="outer")
    boundaries = morphology.binary_dilation(boundaries, disk(2))
    overlay = img.copy()
    overlay[boundaries] = np.array([255, 0, 255], dtype=np.uint8)

    mask_vis = np.zeros_like(img, dtype=np.uint8)
    mask_vis[keep_mask] = 255

    # -------- Measurements on union (single ROI) --------
    union_lab = measure.label(keep_mask, connectivity=2)
    union_props = measure.regionprops(union_lab)[0]

    area = float(union_props.area)
    perim = float(measure.perimeter(keep_mask))
    minr, minc, maxr, maxc = union_props.bbox
    bx, by, width, height = float(minc), float(minr), float(maxc - minc), float(maxr - minr)

    cy, cx = union_props.centroid
    cx, cy = float(cx), float(cy)

    major = float(union_props.major_axis_length or 0.0)
    minor = float(union_props.minor_axis_length or 0.0)
    angle = float(np.degrees(union_props.orientation or 0.0))

    circ = (4.0 * math.pi * area) / (perim * perim) if perim > 0 else 0.0

    # Convex hull for solidity
    hull_img = morphology.convex_hull_image(keep_mask)
    hull_area = float(hull_img.sum())
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    # Feret from union points (x=col, y=row)
    ys, xs = np.nonzero(keep_mask)
    pts = np.column_stack((xs.astype(float), ys.astype(float)))
    if pts.shape[0] >= 3:
        feret, minFeret, feretAngle, feretX, feretY = feret_features_from_points(pts)
    else:
        feret = minFeret = feretAngle = feretX = feretY = 0.0

    ar = float(major / minor) if minor > 0 else 0.0
    roundness = float((4.0 * area) / (math.pi * major * major)) if major > 0 else 0.0

    # Intensity stats on inverted gray (brightfield style)
    gray_u8 = (color.rgb2gray(img) * 255.0).astype(np.uint8)
    stat_img = 255 - gray_u8
    vals = stat_img[keep_mask].astype(float)
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

    # Intensity-weighted centroid (XM, YM)
    ym, xm = ndi.center_of_mass(stat_img, labels=keep_mask.astype(np.uint8), index=1)
    xm, ym = float(xm), float(ym)

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
