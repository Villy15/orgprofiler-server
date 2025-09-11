# main.py
from __future__ import annotations

import base64, io, logging, math, os, subprocess, tempfile
from functools import lru_cache
from typing import Any, Dict
import cjdk 

import imagej
import jpype
import numpy as np
import scyjava
from scyjava import jimport
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import filters, measure, morphology, segmentation
from skimage.morphology import square, disk
from skimage.measure import perimeter_crofton  # perimeter closer to ImageJ

# -----------------------------------------------------------------------------
# FastAPI
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
# PyImageJ init
# -----------------------------------------------------------------------------
# Default to core ImageJ (clean with JDK 11/17). For full Fiji, set:
#   export IMAGEJ_COORD='sc.fiji:fiji:2.14.0'
IMAGEJ_COORD = os.environ.get("IMAGEJ_COORD", "net.imagej:imagej:2.14.0")

def _detect_java_home() -> str | None:
    """Prefer existing JAVA_HOME; else install a JDK 17 into the app dir."""
    jh = os.environ.get("JAVA_HOME")
    if jh and os.path.isdir(jh):
        return jh
    try:
        jh = cjdk.install("17", vendor="temurin")  # downloads a JDK tarball, returns path
        os.environ["JAVA_HOME"] = jh
        os.environ["PATH"] = f"{jh}/bin:" + os.environ.get("PATH", "")
        return jh
    except Exception:
        logging.exception("Failed to install JDK with cjdk")
        return None

@lru_cache(maxsize=1)
def _init_ij():
    java_home = _detect_java_home()
    log.info("Initializing ImageJ…")
    log.info("Using JAVA_HOME: %s", java_home)
    if not java_home:
        raise RuntimeError("No Java detected and auto-install failed.")

    os.environ.setdefault("SCYJAVA_FETCH_JAVA", "never")  # we provide Java ourselves
    scyjava.config.add_option("-Djava.awt.headless=true")
    scyjava.config.add_option("-Xmx1g")

    ij = imagej.init(os.environ.get("IMAGEJ_COORD", "net.imagej:imagej:2.14.0"), mode="headless")
    log.info("ImageJ version: %s", ij.getVersion())
    IJ = jimport("ij.IJ")
    return ij, IJ

def get_ij():
    ij, IJ = _init_ij()
    if not jpype.isThreadAttachedToJVM():
        jpype.attachThreadToJVM()
    return ij, IJ

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_data_url_png(arr: np.ndarray) -> str:
    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    im = Image.fromarray(a if a.ndim == 2 else a, mode="L" if a.ndim == 2 else "RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def rgb_to_gray_ij_u8(img_rgb: np.ndarray) -> np.ndarray:
    """ImageJ's 8-bit gray: 0.299R + 0.587G + 0.114B (no gamma)."""
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

def feret_features_from_points(points_xy: np.ndarray):
    """Return Feret max, min, angle (deg), start X, start Y from hull points (x,y) in *pixels*."""
    if points_xy.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    hull = ConvexHull(points_xy)
    H = points_xy[hull.vertices]
    d2max, i_best, j_best = 0.0, 0, 0
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            dx, dy = H[j, 0] - H[i, 0], H[j, 1] - H[i, 1]
            d2 = dx * dx + dy * dy
            if d2 > d2max:
                d2max, i_best, j_best = d2, i, j
    feret_max = math.sqrt(d2max)
    feret_angle = math.degrees(math.atan2(H[j_best, 1] - H[i_best, 1], H[j_best, 0] - H[i_best, 0]))
    feretX, feretY = float(H[i_best, 0]), float(H[i_best, 1])

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
        w = width_for_edge(H[i], H[(i + 1) % len(H)])
        if w < min_width:
            min_width = w

    return float(feret_max), float(min_width), float(feret_angle), feretX, feretY

def build_mask_fiji_like(img_rgb: np.ndarray, sigma_pre: float, dilate_iter: int, erode_iter: int) -> np.ndarray:
    """Reproduce Fiji macro: threshold -> fill/dilate/erode/fill -> blur(mask) -> threshold(dark)."""
    gray = rgb_to_gray_ij_u8(img_rgb)
    t1 = filters.threshold_isodata(gray)
    core = gray <= t1
    core = ndi.binary_fill_holes(core)
    se = square(3)
    for _ in range(int(dilate_iter)):
        core = morphology.binary_dilation(core, se)
    core = ndi.binary_fill_holes(core)
    for _ in range(int(erode_iter)):
        core = morphology.binary_erosion(core, se)
    core = ndi.binary_fill_holes(core)
    soft = filters.gaussian(core.astype(float), sigma=sigma_pre, preserve_range=True)
    t2 = filters.threshold_isodata((soft * 255).astype(np.uint8)) / 255.0
    final = soft <= t2
    return final

# -----------------------------------------------------------------------------
# Analyze
# -----------------------------------------------------------------------------
@api.post("/analyze")
async def analyze(
    file: UploadFile = File(...),

    # Fiji-like knobs
    sigma_pre: float = Query(6.4, ge=0.0),
    dilate_iter: int = Query(4, ge=0),
    erode_iter: int = Query(5, ge=0),
    min_area_px: float = Query(60000, ge=0),    # area filter in pixels (macro logic)
    max_area_px: float = Query(2.0e7, ge=0),
    min_circ: float   = Query(0.28, ge=0.0, le=1.0),
    edge_margin: float = Query(0.20, ge=0.0, le=0.49),
    pixel_size_um: float = Query(0.86, gt=0.0),  # calibration to match Fiji output
    overlay_width: int = Query(11, ge=1),        # visual only: match Fiji's line width ~11
) -> Dict[str, Any]:

    data = await file.read()
    try:
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Invalid image")

    # Crop 2 px border like the macro
    if min(img.shape[:2]) > 8:
        img = img[2:-2, 2:-2, :]

    H, W, _ = img.shape

    # ---------- Build mask (prefer ImageJ) ----------
    used_ij = True
    try:
        ij, IJ = get_ij()
        tmp_path, imp = None, None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                Image.fromarray(img).save(tmp.name)
                tmp_path = tmp.name

            imp = IJ.openImage(tmp_path)
            if imp is None:
                raise RuntimeError("IJ.openImage returned None")

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

            mask_u8 = ij.py.from_java(imp)
            if mask_u8.ndim != 2:
                mask_u8 = mask_u8[..., 0]
            mask_bool = mask_u8 > 0
        finally:
            try:
                if imp is not None:
                    imp.changes = False
                    imp.close()
            except Exception:
                pass
            if tmp_path and os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass

    except Exception as e:
        used_ij = False
        log.warning("ImageJ path failed; using scikit-image. Reason: %s", e)
        mask_bool = build_mask_fiji_like(
            img_rgb=img,
            sigma_pre=sigma_pre,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
        )

    # -------- Filter regions (area, circularity, edge), then UNION --------
    labels = measure.label(mask_bool, connectivity=2)
    if labels.max() == 0:
        raise HTTPException(422, "No contours found")

    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

    keep_mask = np.zeros_like(labels, dtype=bool)
    for r in measure.regionprops(labels):
        a_px = float(r.area)
        if a_px < min_area_px or a_px > max_area_px:
            continue
        # perimeter in px for filtering circularity
        p_px = float(perimeter_crofton(labels == r.label, directions=4))
        circ_f = (4.0 * math.pi * a_px) / (p_px * p_px) if p_px > 0 else 0.0
        if circ_f < min_circ:
            continue
        cy, cx = r.centroid
        if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
            continue
        keep_mask |= labels == r.label

    if not keep_mask.any():
        largest = max(measure.regionprops(labels), key=lambda rr: rr.area)
        keep_mask = labels == largest.label

    # -------- Visual overlays (magenta, thickness like Fiji) --------
    boundaries = segmentation.find_boundaries(keep_mask, mode="outer")
    boundaries = morphology.binary_dilation(boundaries, disk(max(1, overlay_width // 2)))
    overlay = img.copy()
    overlay[boundaries] = np.array([255, 0, 255], dtype=np.uint8)

    mask_vis = np.zeros_like(img, dtype=np.uint8)
    mask_vis[keep_mask] = 255

    # -------- Measurements (convert to µm / µm²) --------
    px = float(pixel_size_um)
    px2 = px * px

    union_lab = measure.label(keep_mask, connectivity=2)
    props = measure.regionprops(union_lab)[0]

    area_px = float(props.area)
    perim_px = float(perimeter_crofton(keep_mask, directions=4))
    minr, minc, maxr, maxc = props.bbox
    bx_px, by_px = float(minc), float(minr)
    width_px, height_px = float(maxc - minc), float(maxr - minr)

    cy_px, cx_px = props.centroid
    major_px = float(props.major_axis_length or 0.0)
    minor_px = float(props.minor_axis_length or 0.0)
    angle = float(np.degrees(props.orientation or 0.0))

    # Feret from union points (in px)
    ys, xs = np.nonzero(keep_mask)
    pts = np.column_stack((xs.astype(float), ys.astype(float)))
    if pts.shape[0] >= 3:
        feret_px, minFeret_px, feretAngle, feretX_px, feretY_px = feret_features_from_points(pts)
    else:
        feret_px = minFeret_px = feretAngle = feretX_px = feretY_px = 0.0

    # Convert to µm / µm²
    area = area_px * px2
    perim = perim_px * px
    bx, by = bx_px * px, by_px * px
    width, height = width_px * px, height_px * px
    cx, cy = cx_px * px, cy_px * px
    major, minor = major_px * px, minor_px * px
    feret, minFeret = feret_px * px, minFeret_px * px
    feretX, feretY = feretX_px * px, feretY_px * px

    circ = (4.0 * math.pi * area) / (perim * perim) if perim > 0 else 0.0
    hull_img = morphology.convex_hull_image(keep_mask)
    hull_area = float(hull_img.sum())
    solidity = float(area_px / hull_area) if hull_area > 0 else 0.0  # note: solidity is unitless (use px ratio)

    ar = float(major / minor) if minor > 0 else 0.0
    roundness = float((4.0 * area) / (math.pi * major * major)) if major > 0 else 0.0

    # ---- Intensity stats on inverted ImageJ gray (8-bit)
    gray_u8 = rgb_to_gray_ij_u8(img)
    stat_img = 255 - gray_u8
    vals = stat_img[keep_mask].astype(np.uint8)   # 0..255 integers

    if vals.size == 0:
        raise HTTPException(422, "Empty ROI after masking")

    mean   = float(vals.mean())
    median = float(np.median(vals))
    mode   = float(np.bincount(vals).argmax())
    vmin   = float(vals.min())
    vmax   = float(vals.max())
    stdDev = float(vals.std(ddof=0))
    if stdDev > 0:
        z = (vals.astype(np.float32) - mean) / stdDev
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    else:
        skew = kurt = 0.0

    # ImageJ “IntDen” is sum of pixel values in ROI on the inverted gray
    rawIntDen = float(vals.sum())
    intDen    = rawIntDen  # equals area_px * mean on 8-bit inverted gray

    # Intensity-weighted centroid (XM, YM) on inverted gray (in µm)
    ym_px, xm_px = ndi.center_of_mass(stat_img, labels=keep_mask.astype(np.uint8), index=1)
    xm, ym = float(xm_px * px), float(ym_px * px)

    # ---------- LEGACY SCHEMA (exact keys) ----------
    results = {
        "area": area, "mean": mean, "stdDev": stdDev, "mode": mode, "min": vmin, "max": vmax,
        "x": cx, "y": cy, "xm": xm, "ym": ym, "perim": perim, "bx": bx, "by": by,
        "width": width, "height": height, "major": major, "minor": minor, "angle": angle,
        "circ": circ, "feret": feret, "intDen": intDen, "median": median, "skew": -skew,  # macro outputs -Skew
        "kurt": kurt, "rawIntDen": rawIntDen, "feretX": feretX, "feretY": feretY,
        "feretAngle": feretAngle, "minFeret": minFeret, "ar": ar, "round": roundness,
        "solidity": solidity, "usedImageJ": used_ij,
    }

    return {
        "results": results,
        "roi_image": to_data_url_png(overlay),
        "mask_image": to_data_url_png(mask_vis),
    }
