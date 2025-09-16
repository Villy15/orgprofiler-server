from __future__ import annotations
import json
from typing import Tuple
import contextlib
from functools import lru_cache

import base64
import io
import math
import re
import sys
from typing import Any, Dict

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage import filters, measure, morphology, segmentation
from skimage.morphology import footprint_rectangle, disk
from skimage.measure import perimeter_crofton
from skimage.segmentation import clear_border


@lru_cache(maxsize=64)
def _disk_bool(r: int) -> np.ndarray:
    # Cached for the optional 'dilate' path
    return disk(int(r)).astype(bool, copy=False)

# ----------------------------
# Logging (Loguru)
# ----------------------------
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <7}</level> | "
           "{name}:{function}:{line} - <level>{message}</level>",
)

# ----------------------------
# psutil + peak RSS helpers
# ----------------------------
import os, time, platform
try:
    import psutil  # process CPU/memory
except Exception:
    psutil = None
    logger.warning("psutil not installed; resource profiling disabled")

def _ru_maxrss_bytes() -> int | None:
    """Best-effort peak RSS (bytes). Linux returns KB; macOS returns bytes."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("linux"):
            return int(r) * 1024
        # macOS returns bytes
        return int(r)
    except Exception:
        return None

# ----------------------------
# Tiny timing helpers
# ----------------------------

@contextlib.contextmanager
def time_block(label: str):
    _t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - _t0
        logger.info(f"[TIMER] {label}: {dt:.3f}s")


class ResourceProfiler:
    """Context manager to measure wall time, CPU time, RSS delta, and peak RSS."""
    def __init__(self, label: str = "analyze"):
        self.label = label
        self.metrics: Dict[str, float] = {}

    def __enter__(self):
        self.t0 = time.perf_counter()
        if psutil:
            self.proc = psutil.Process(os.getpid())
            self.ct0 = self.proc.cpu_times()
            self.mem0 = self.proc.memory_info().rss
        else:
            self.proc = None
            self.ct0 = None
            self.mem0 = None
        self.maxrss0 = _ru_maxrss_bytes()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.perf_counter()
        wall_s = t1 - self.t0

        cpu_user_s = cpu_sys_s = rss_now = rss_delta = None
        if self.proc:
            ct1 = self.proc.cpu_times()
            mi1 = self.proc.memory_info()
            cpu_user_s = (ct1.user - self.ct0.user)
            cpu_sys_s  = (ct1.system - self.ct0.system)
            rss_now    = mi1.rss
            rss_delta  = (rss_now - self.mem0)

        maxrss1 = _ru_maxrss_bytes()
        peak_rss_bytes = None
        if maxrss1 is not None and self.maxrss0 is not None:
            peak_rss_bytes = max(0, maxrss1 - self.maxrss0) or maxrss1  # show peak if delta 0

        def mb(x): return None if x is None else round(x / (1024*1024), 3)

        self.metrics = {
            "wall_time_s": round(wall_s, 6),
            "cpu_user_s": None if cpu_user_s is None else round(cpu_user_s, 6),
            "cpu_sys_s": None if cpu_sys_s is None else round(cpu_sys_s, 6),
            "rss_now_bytes": rss_now,
            "rss_now_mb": mb(rss_now),
            "rss_delta_bytes": rss_delta,
            "rss_delta_mb": mb(rss_delta),
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_mb": mb(peak_rss_bytes),
            "platform": platform.platform(),
        }

        logger.info(
            f"[{self.label}] wall={wall_s:.3f}s "
            f"cpu_user={self.metrics['cpu_user_s']}s cpu_sys={self.metrics['cpu_sys_s']}s "
            f"rss_now={self.metrics['rss_now_mb']}MB Δrss={self.metrics['rss_delta_mb']}MB "
            f"peak_rss={self.metrics['peak_rss_mb']}MB"
        )

# ----------------------------
# FastAPI setup
# ----------------------------
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@api.get("/healthz")
def healthz():
    return {"ok": True}

# ----------------------------
# Helpers
# ----------------------------
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

def ring_background_mean(
    u8_img: np.ndarray,
    mask: np.ndarray,
    ring_px: int = 20,
    method: str = "median",
    *,
    bbox: tuple[int, int, int, int] | None = None,  # (minr, minc, maxr, maxc) in mask coords
    pad: int = 2,
    algo: str = "edt",        # "edt" (fast) or "dilate" (compatible)
    max_samples: int = 250_000,
) -> float:
    """
    Fast mean/median intensity in a ring just *outside* the mask.
    - Restricts computation to a padded bbox around the organoid.
    - Default 'edt' path avoids slow large-kernel dilation.
    """
    r = max(1, int(ring_px))
    h, w = u8_img.shape[:2]

    # Determine bbox (use provided or compute from mask)
    if bbox is None:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return 0.0
        minr, maxr = int(ys.min()), int(ys.max())
        minc, maxc = int(xs.min()), int(xs.max())
    else:
        minr, minc, maxr, maxc = map(int, bbox)

    # Pad bbox by ring thickness, clamp to image
    minr = max(0, minr - r - pad); maxr = min(h, maxr + r + pad)
    minc = max(0, minc - r - pad); maxc = min(w, maxc + r + pad)

    roi_mask = mask[minr:maxr, minc:maxc].astype(bool, copy=False)
    roi_img  = u8_img[minr:maxr, minc:maxc]

    # Build ring
    if algo == "edt":
        # Distance on 'outside' to nearest mask pixel; ring = (0, r]
        outside = ~roi_mask
        if not outside.any():
            outside = ~mask
            roi_img = u8_img
        dist = ndi.distance_transform_edt(outside)
        ring = (dist > 0) & (dist <= r)
    else:
        # Optional: faster than skimage on many setups, still bbox-restricted
        dil = ndi.binary_dilation(roi_mask, structure=_disk_bool(r), iterations=1)
        ring = dil & ~roi_mask

    vals = roi_img[ring]
    if vals.size == 0:       # fallback if ring empty (tiny ROI, edge)
        vals = roi_img[~roi_mask]
    if vals.size == 0:
        return 0.0

    # Optional sub-sampling for huge rings (median is robust)
    if vals.size > max_samples:
        idx = np.random.randint(0, vals.size, size=max_samples, dtype=np.int64)
        vals = vals[idx]

    return float(np.median(vals) if method == "median" else float(vals.mean()))
# ----------------------------
# Mask builder (Fiji-like path)
# ----------------------------
def build_mask_fiji_like(
    img_rgb: np.ndarray,
    *,
    sigma_pre: float,
    dilate_iter: int,
    erode_iter: int,
    clear_border_artifacts: bool = True,
) -> np.ndarray:
    """
    Reproduce the Fiji macro path with memory-friendly ops.
      1) Threshold (isodata) on 8-bit gray -> dark core
      2) Fill / Dilate^n / Fill / Erode^m / Fill
      3) Gaussian Blur (sigma = sigma_pre) on float32
      4) Auto-threshold again (keep object)
    """
    gray = rgb_to_gray_ij_u8(img_rgb)

    # Step 1: initial threshold (assume darker organoid)
    t1 = filters.threshold_isodata(gray)
    core = gray <= t1
    core = ndi.binary_fill_holes(core)

    # Morphology: dilate, fill, erode, fill
    se = footprint_rectangle((3, 3))
    for _ in range(int(dilate_iter)):
        core = morphology.binary_dilation(core, footprint=se)
    core = ndi.binary_fill_holes(core)
    for _ in range(int(erode_iter)):
        core = morphology.binary_erosion(core, footprint=se)
    core = ndi.binary_fill_holes(core)

    # Gaussian on float32
    core_f32 = core.astype(np.float32, copy=False)
    soft_f32 = np.empty_like(core_f32, dtype=np.float32)
    ndi.gaussian_filter(core_f32, sigma=sigma_pre, output=soft_f32, mode="nearest")

    # Second threshold: keep object (>=), not background
    t2_u8 = filters.threshold_isodata((soft_f32 * 255).astype(np.uint8))
    thr = t2_u8 / 255.0
    final = soft_f32 >= thr

    if clear_border_artifacts:
        final = clear_border(final)

    # Safety: if >80% of frame is foreground, flip (rare inversion cases)
    if final.mean() > 0.8:
        final = ~final

    return final

# helper to parse day/organoid from filename
FILENAME_RE = re.compile(r"_d(?P<day>\d{2})_(?P<organoid>\d{1,3})(?=\.)")

def parse_day_organoid(fname: str) -> Tuple[str | None, str | None]:
    """
    Extract day ('00') and organoid number ('008') from filenames like:
      YYYY-MM-DD_d00_08.jpg
    Returns zero-padded strings or (None, None) if not matched.
    """
    m = FILENAME_RE.search(fname or "")
    if not m:
        return None, None
    day = m.group("day")
    organoid = m.group("organoid").zfill(3)  # Fiji-style 3-digit
    return day, organoid

# ----------------------------
# Core analysis
# ----------------------------
def analyze_image(
    img: np.ndarray,
    *,
    sigma_pre: float,
    dilate_iter: int,
    erode_iter: int,
    min_area_px: float,
    max_area_px: float,
    min_circ: float,
    edge_margin: float,
    pixel_size_um: float,
    overlay_width: int,
    return_images: bool,
    crop_overlay: bool,
    crop_border_px: int,
    ring_px: int,
) -> Dict[str, Any]:
    H, W, _ = img.shape
    logger.info(f"Analyze start | shape={H}x{W} sigma={sigma_pre} dilate={dilate_iter} erode={erode_iter} px_um={pixel_size_um}")

    # Optional border crop
    with time_block("optional border crop"):
        if crop_border_px > 0 and min(img.shape[:2]) > 2 * crop_border_px:
            img = img[crop_border_px:-crop_border_px, crop_border_px:-crop_border_px, :]
            H, W, _ = img.shape
            logger.debug(f"Cropped border -> {H}x{W}")

    # Build mask
    with time_block("build_mask_fiji_like"):
        mask_bool = build_mask_fiji_like(
            img_rgb=img,
            sigma_pre=sigma_pre,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
        )

    with time_block("label mask + regionprops (initial)"):
        labels = measure.label(mask_bool, connectivity=2)
        if labels.max() == 0:
            logger.warning("No contours found after masking")
            raise HTTPException(422, "No contours found")

    xmin, xmax = W * edge_margin, W * (1.0 - edge_margin)
    ymin, ymax = H * edge_margin, H * (1.0 - edge_margin)

    with time_block("filter components"):
        keep_mask = np.zeros_like(labels, dtype=bool)
        for r in measure.regionprops(labels):
            a_px = float(r.area)
            if a_px < min_area_px or a_px > max_area_px:
                continue
            p_px = float(perimeter_crofton(labels == r.label, directions=4))
            circ_f = (4.0 * math.pi * a_px) / (p_px * p_px) if p_px > 0 else 0.0
            if circ_f < min_circ:
                continue
            cy, cx = r.centroid
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                continue
            keep_mask |= labels == r.label

    with time_block("fallback to largest (if needed)"):
        if not keep_mask.any():
            largest = max(measure.regionprops(labels), key=lambda rr: rr.area)
            logger.info("Filters removed all; falling back to largest region")
            keep_mask = labels == largest.label

    # Measurements in px
    with time_block("regionprops (final) + measurements"):
        union_lab = measure.label(keep_mask, connectivity=2)
        props_list = measure.regionprops(union_lab)
        if not props_list:
            logger.error("Empty ROI after masking")
            raise HTTPException(422, "Empty ROI after masking")
        props = props_list[0]
        area_px = float(props.area)
        perim_px = float(perimeter_crofton(keep_mask, directions=4))
        minr, minc, maxr, maxc = props.bbox
        cy_px, cx_px = props.centroid
        major_px = float(props.major_axis_length or 0.0)
        minor_px = float(props.minor_axis_length or 0.0)
        angle = float(np.degrees(props.orientation or 0.0))

    # Feret from boundary points
    with time_block("feret from contours"):
        contours = measure.find_contours(keep_mask, 0.5)
        if contours:
            cont = max(contours, key=lambda c: c.shape[0])
            if cont.shape[0] > 4000:
                cont = cont[::4]
            pts = np.column_stack((cont[:, 1], cont[:, 0]))  # (x, y)
        else:
            pts = np.empty((0, 2))

        if pts.shape[0] >= 3:
            feret_px, minFeret_px, feretAngle, feretX_px, feretY_px = feret_features_from_points(pts)
        else:
            feret_px = minFeret_px = feretAngle = feretX_px = feretY_px = 0.0

    # Convert to µm / µm²
    px_um = float(pixel_size_um)
    area = area_px * (px_um ** 2)
    perim = perim_px * px_um
    cx, cy = cx_px * px_um, cy_px * px_um
    major, minor = major_px * px_um, minor_px * px_um
    feret, minFeret = feret_px * px_um, minFeret_px * px_um
    feretX, feretY = feretX_px * px_um, feretY_px * px_um
    bx_px, by_px = float(props.bbox[1]), float(props.bbox[0])
    width_px, height_px = float(props.bbox[3] - props.bbox[1]), float(props.bbox[2] - props.bbox[0])
    bx, by = bx_px * px_um, by_px * px_um
    width, height = width_px * px_um, height_px * px_um

    circ = (4.0 * math.pi * area) / (perim * perim) if perim > 0 else 0.0
    solidity = float(props.solidity) if hasattr(props, "solidity") else 0.0
    ar = float(major / minor) if minor > 0 else 0.0
    roundness = float((4.0 * area) / (math.pi * major * major)) if major > 0 else 0.0

    # Intensity stats on inverted ImageJ gray (8-bit)
    with time_block("intensity stats (inverted gray) + COM"):
        gray_u8 = rgb_to_gray_ij_u8(img)
        stat_img = (255 - gray_u8).astype(np.uint8)
        vals = stat_img[keep_mask].astype(np.uint8)
        if vals.size == 0:
            raise HTTPException(422, "Empty ROI after masking (no intensity values)")

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

        rawIntDen = float(vals.sum())  # Integrated intensity
        intDen    = rawIntDen

        # Intensity-weighted centroid (XM, YM) on inverted gray (in µm)
        ym_px, xm_px = ndi.center_of_mass(stat_img, labels=keep_mask.astype(np.uint8), index=1)
        xm, ym = float(xm_px * px_um), float(ym_px * px_um)

    # Background ring + corrected metrics
    with time_block("background ring + corrected metrics"):
        bg = ring_background_mean(
            stat_img,
            keep_mask,
            ring_px=ring_px,
            method="median",
            bbox=(minr, minc, maxr, maxc),  # <-- uses the ROI you already computed
            # algo="edt",  # default; fastest
        )
        corrected_total_intensity = rawIntDen - bg * area_px
        corrected_mean_intensity  = mean - bg
        corrected_min_intensity   = vmin - bg
        corrected_max_intensity   = vmax - bg

        # Extra derived metrics
        eq_diam = math.sqrt(4.0 * area / math.pi)
        centroid_to_com = math.hypot(float(xm - cx), float(ym - cy))

    # ---------- Visual overlays ----------
    with time_block("build overlays" if return_images else "skip overlays"):
        if return_images:
            if crop_overlay:
                local_mask = keep_mask[minr:maxr, minc:maxc]
                overlay = img[minr:maxr, minc:maxc].copy()
                boundaries = segmentation.find_boundaries(local_mask, mode="outer")
            else:
                local_mask = keep_mask
                overlay = img.copy()
                boundaries = segmentation.find_boundaries(keep_mask, mode="outer")
            boundaries = morphology.binary_dilation(boundaries, disk(max(1, overlay_width // 2)))
            overlay[boundaries] = np.array([255, 0, 255], dtype=np.uint8)
            mask_vis = (local_mask.astype(np.uint8) * 255)
            roi_image_b64 = to_data_url_png(overlay)
            mask_image_b64 = to_data_url_png(mask_vis)
        else:
            roi_image_b64 = ""
            mask_image_b64 = ""

    # Unified results mapping (frontend can format Fiji headers)
    results = {
        "area": area, "mean": mean, "stdDev": stdDev, "mode": mode, "min": vmin, "max": vmax,
        "x": cx, "y": cy, "xm": xm, "ym": ym, "perim": perim, "bx": bx, "by": by,
        "width": width, "height": height, "major": major, "minor": minor, "angle": angle,
        "circ": circ, "feret": feret, "intDen": intDen, "median": median, "skew": -skew,  # keep your legacy sign
        "kurt": kurt, "rawIntDen": rawIntDen, "feretX": feretX, "feretY": feretY,
        "feretAngle": feretAngle, "minFeret": minFeret, "ar": ar, "round": roundness,
        "solidity": solidity,
        # --- Added fields for Fiji-style export ---
        "eqDiam": eq_diam,
        "corrTotalInt": corrected_total_intensity,
        "corrMeanInt": corrected_mean_intensity,
        "corrMinInt": corrected_min_intensity,
        "corrMaxInt": corrected_max_intensity,
        "centroidToCom": centroid_to_com,
        "bgRing": bg,
    }

    logger.info(f"Analyze done | area_px={area_px:.0f} perim_px={perim_px:.1f} circ={circ:.3f} eq_diam={eq_diam:.2f}µm")
    return {
        "results": results,
        "roi_image": roi_image_b64,
        "mask_image": mask_image_b64,
    }

# ----------------------------
# API route (thin wrapper)
# ----------------------------
@api.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    # Fiji-like knobs
    sigma_pre: float = Query(6.4, ge=0.0),
    dilate_iter: int = Query(4, ge=0),
    erode_iter: int = Query(5, ge=0),
    min_area_px: float = Query(60000, ge=0),
    max_area_px: float = Query(2.0e7, ge=0),
    min_circ: float   = Query(0.28, ge=0.0, le=1.0),
    edge_margin: float = Query(0.20, ge=0.0, le=0.49),
    pixel_size_um: float = Query(0.86, gt=0.0),
    overlay_width: int = Query(11, ge=1),
    return_images: bool = Query(True),
    crop_overlay: bool = Query(False, description="Return cropped overlay (bbox) instead of full frame"),
    crop_border_px: int = Query(2, ge=0, description="Border pixels to crop before analysis"),
    ring_px: int = Query(20, ge=1, description="Background ring thickness in pixels"),
    profile: bool = Query(False),
    day0_area: float | None = Form(None),
    mean_day0_area: float | None = Form(None),
    day0_area_by_organoid: str | None = Form(None),
) -> Dict[str, Any]:
    with time_block("read upload bytes"):
        data = await file.read()
    filename = file.filename or ""
    logger.info(f"Received file: filename={file.filename!r} size={len(data)} bytes")

    day, organoid_number = parse_day_organoid(filename)

    with time_block("PIL decode + to RGB + np.array"):
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            logger.exception("Invalid image payload")
            raise HTTPException(400, "Invalid image")

    # Full analysis (already has ResourceProfiler)
    with ResourceProfiler("analyze") as prof:
        with time_block("analyze_image total"):
            payload = analyze_image(
                img,
                sigma_pre=sigma_pre,
                dilate_iter=dilate_iter,
                erode_iter=erode_iter,
                min_area_px=min_area_px,
                max_area_px=max_area_px,
                min_circ=min_circ,
                edge_margin=edge_margin,
                pixel_size_um=pixel_size_um,
                overlay_width=overlay_width,
                return_images=return_images,
                crop_overlay=crop_overlay,
                crop_border_px=crop_border_px,
                ring_px=ring_px,
            )

    # growth rate & profile (unchanged, but you can time too if you like)
    with time_block("growth-rate compute"):
        area_value = float(payload["results"]["area"])
        growth_rate = None
        if day == "00":
            growth_rate = 1.0
        else:
            baseline = None
            if day0_area_by_organoid:
                try:
                    mapping = json.loads(day0_area_by_organoid)
                    if isinstance(mapping, dict) and organoid_number:
                        maybe = mapping.get(organoid_number)
                        if isinstance(maybe, (int, float)):
                            baseline = float(maybe)
                except Exception:
                    logger.warning("Invalid JSON for day0_area_by_organoid; ignoring")
            if baseline is None and day0_area is not None:
                baseline = float(day0_area)
            if baseline is None and mean_day0_area is not None:
                baseline = float(mean_day0_area)
            if baseline and baseline > 0:
                growth_rate = area_value / baseline

        payload["results"]["day"] = day
        payload["results"]["organoidNumber"] = organoid_number
        payload["results"]["growthRate"] = growth_rate

        if profile and prof.metrics:
            payload["profile"] = prof.metrics

    return payload


@api.middleware("http")
async def log_request_timing(request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dt = time.perf_counter() - t0
        clen = request.headers.get("content-length")
        logger.info(
            f"[HTTP] {request.method} {request.url.path} -> {getattr(response, 'status_code', '?')} "
            f"in {dt:.3f}s (Content-Length={clen})"
        )
