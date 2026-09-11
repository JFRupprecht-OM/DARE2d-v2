#!/usr/bin/env python3
"""
qazi - 09/12/2025

Final aggregation and visualization of multi-model division detections.

To run:

python scripts/postprocessing/main.py `
  --output_root "path_to_all_model_outputs_in_one_folder" `
  --image_name  "7Z_e1_image2_3-ant" `
  --image_stack "path_to_your_image" `
  --save_dir "output_dir" `
  --eps 10 --min_models 6 --num_models 8 --angle_mode auto

What this code does:
 - Load per-model detection .npy files
 - Spatial clustering using HDBSCAN (or DBSCAN fallback)
 - Per-cluster aggregation into consensus:
     median x,y, pos_std, chosen signed angle, angular std, length statistics,
     list of contributing models, dominant_model (model whose angle was chosen),
     support_fraction (n_models / num_models)
 - Temporal deduplication across consecutive frames
 - Visualization per-frame:
     symmetric wedge sectors (angle +/- angle_std) on both sides of rod axis,
     positional pink halo behind red center,
     rod drawn on top (rod_color) and red center marker
 - Save per-frame consensus NPYs into `chosen_divisions/`
 - Save a stacked TIFF visual `post_processed_{movie_nam}.tiff`
 - Save a global CSV summary `post_processed_{movie_nam}_summary.csv`
 - CLI-driven, accepts default colors, alpha, thickness, max wedge radius, etc.

Notes:
 - ANGLE CONVENTION: the per-model inputs (division_position*.npy written by
   scripts/inference/multistage_detection2d.py) carry the MODEL angle theta from convert_values
   (from +row toward +col, degrees in (-90, 90]). detect_angle_units_and_convert rewrites them to
   this script's IMAGE convention angle_img = 90 - theta (from +x/col toward +y/row, y down),
   which draw_consensus_on_image (dx = cos, dy = sin) needs; the values saved by THIS script
   (chosen_divisions/*.npy 'angle', CSV 'angle_deg') are therefore in the image convention.
   In-process consumers that draw (row, col) = (cos, sin) -- the napari plugin -- must map back
   (theta = 90 - a if a >= 0 else -90 - a); napari_dare2d._api.consensus does so.
 - This script intentionally does NOT auto-install hdbscan. If you want HDBSCAN results,
   install hdbscan in your environment; otherwise the script falls back to sklearn DBSCAN.
 - The script expects model outputs saved as "division_position{frame}.npy" under
   folders like <image_nam>_1, <image_nam>_2, ..., <image_nam>_{num_models}.
"""


import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import math
from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)

# visualization + IO
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt
from skimage import io as skio
import imageio
import cv2
from datetime import datetime
import csv

# clustering
from sklearn.cluster import DBSCAN
try:
    import hdbscan
except Exception:
    hdbscan = None
    print("[WARN] hdbscan not available — will fall back to sklearn.DBSCAN for density clustering.")

# -------------------------
# Utilities: load per-model detections
# -------------------------
def load_all_model_detections(output_root, image_name, num_models=8):
    """
    Load detections saved by oracle.py for models 1..num_models.
    Returns dict: frame_idx -> list of (model_id, x, y, raw_angle, length)
    """
    all_dets = defaultdict(list)
    nam = Path(image_name).name

    for m in range(1, num_models + 1):
        model_root = os.path.join(output_root, f"{nam}_{m}")
        model_inner = os.path.join(model_root, nam)
        folder = model_inner if os.path.isdir(model_inner) else model_root

        if not os.path.isdir(folder):
            print(f"[WARN] model folder not found: {folder} (skipping model {m})")
            continue

        files = sorted(glob.glob(os.path.join(folder, "division_position*.npy")))
        if len(files) == 0:
            print(f"[INFO] no division_position*.npy in {folder} for model {m}")
            continue

        for fp in files:
            base = os.path.splitext(os.path.basename(fp))[0]
            idx_str = base.replace("division_position", "")
            try:
                frame_idx = int(idx_str)
            except ValueError:
                continue
            try:
                arr = np.load(fp, allow_pickle=True)
            except Exception as e:
                print(f"[WARN] failed to load {fp}: {e}")
                continue
            for d in arr:
                if isinstance(d, dict):
                    x = float(d.get("x", np.nan))
                    y = float(d.get("y", np.nan))
                    ang = float(d.get("angle", 0.0))
                    L = float(d.get("length", 0.0))
                else:
                    continue
                if np.isnan(x) or np.isnan(y):
                    continue
                all_dets[frame_idx].append((m, x, y, ang, L))
    return all_dets

# -------------------------
# Angle unit detection & conversion to degrees + image convention
# -------------------------
def detect_angle_units_and_convert(all_dets, mode="auto"):
    """
    Detect whether input angles are radians or degrees (auto) and convert to DEGREES.
    Additionally convert model math-angles into the IMAGE drawing convention:
      image_angle_deg = 90.0 - model_angle_deg
    This stores signed angles in degrees in range (-90, 90].
    Modifies all_dets in-place.

    Conventions (see also the module docstring):
      - model angle theta (input): from +row toward +col, as decoded by
        dare2d.datamodule.post_processing.regression2d_pp.convert_values and drawn by
        regression2d_visualisation.project_point (row <- cos, col <- sin);
      - image angle (output): from +x/col toward +y/row (y down), the standard cv2 angle used by
        draw_consensus_on_image / polygon_sector (dx <- cos, dy <- sin) and stored as-is in the
        CSV 'angle_deg' and chosen_divisions/*.npy of this script.
      The two are a matched pair: a consumer that draws (row, col) = (cos, sin) -- e.g. napari's
      detections_to_vectors -- must NOT receive image-convention angles (napari_dare2d._api.consensus
      inverts them before returning).
    Unit heuristic ('auto'): the median of the first 500 |angles| <= 2*pi*1.1 (~6.91) is taken as
    radians; in-process callers whose angles are already degrees should pass mode='degrees'.
    """
    samples = []
    for f, items in all_dets.items():
        for it in items:
            samples.append(abs(float(it[3])))
            if len(samples) >= 500:
                break
        if len(samples) >= 500:
            break

    if mode == "degrees":
        orig = "degree"
    elif mode == "radians":
        orig = "radian"
    else:
        if len(samples) == 0:
            orig = "unknown"
        else:
            med = np.median(samples)
            orig = "radian" if med <= (2.0 * math.pi * 1.1) else "degree"

    for f, items in list(all_dets.items()):
        for i, it in enumerate(items):
            m, x, y, ang_raw, L = it
            # convert to degrees if needed
            if orig == "radian":
                ang_deg = float(ang_raw) * 180.0 / math.pi
            else:
                ang_deg = float(ang_raw)

            # convert to image convention as per the inference results
            ang_img = 90 - ang_deg
            ang_img = ((ang_img + 90.0) % 180.0) - 90.0  # normalize to (-90,90]

            items[i] = (m, x, y, ang_img, L)

    print(f"[INFO] Detected original angle unit = {orig}. Converted stored angles to degrees and normalized to image convention (signed).")
    return orig

# -------------------------
# clustering wrappers
# -------------------------
def cluster_hdbscan(points_xy, eps, min_cluster_size=2, min_samples=1):
    """Cluster points using HDBSCAN if available else sklearn DBSCAN."""
    if points_xy.shape[0] == 0:
        return np.array([], dtype=int)
    if hdbscan is None:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_xy)
        return db.labels_
    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, min_cluster_size),
                                min_samples=min_samples,
                                cluster_selection_epsilon=eps)
    return clusterer.fit_predict(points_xy)

# -------------------------
# Angle consensus: Mode 1
# -------------------------
def pick_consensus_angle_signed(raw_angles_deg):
    """
    raw_angles_deg: list of signed angles in degrees (image convention, i.e. 90 - theta_model as
        produced by detect_angle_units_and_convert; the pick is invariant under that rewrite
        because |fold(90 - t)| == 90 - |t| on (-90, 90], so the same member is chosen in either
        convention -- only its label differs)
    Returns chosen_signed_angle (one member's angle, a representative pick -- not an average)
    and a debug dict.
    """
    arr = np.array(raw_angles_deg, dtype=float)
    if arr.size == 0:
        debug = {"raw_angles": [], "abs_angles": [], "mean_abs": None, "chosen_index": None}
        return 0.0, debug

    abs_arr = np.abs(arr)

    mean_abs = float(np.mean(abs_arr))
    dists = np.abs(abs_arr - mean_abs)
    idx = int(np.argmin(dists))
    chosen_signed = float(arr[idx])

    debug = {
        "raw_angles": [float(x) for x in arr.tolist()],
        "abs_angles": [float(x) for x in abs_arr.tolist()],
        "mean_abs": mean_abs,
        "distances": [float(x) for x in dists.tolist()],
        "chosen_index": idx,
        "chosen_signed_angle": chosen_signed,
        "chosen_abs_angle": float(abs_arr[idx]),
        "chosen_distance": float(dists[idx])
    }
    return chosen_signed, debug

# -------------------------
# Aggregate per-cluster
# -------------------------
def aggregate_cluster_pick_signed(items, min_models, total_models):
    """
    items: list of tuples (model_id, x, y, angle_deg, length)
    min_models: minimum distinct models required to accept cluster
    total_models: total number of models expected (for support_fraction)
    Returns consensus dict or None.
    """
    if len(items) == 0:
        return None
    unique_models = set(it[0] for it in items)
    if len(unique_models) < min_models:
        return None

    xs = np.array([it[1] for it in items], dtype=float)
    ys = np.array([it[2] for it in items], dtype=float)
    centroid = np.array([xs.mean(), ys.mean()], dtype=float)

    per_model = {}
    for it in items:
        m, x, y, ang, L = it
        d2 = (x - centroid[0])**2 + (y - centroid[1])**2
        if (m not in per_model) or (d2 < per_model[m]['d2']):
            per_model[m] = {'x': x, 'y': y, 'angle': ang, 'length': L, 'd2': d2}

    reps = list(per_model.values())
    raw_angles = [r['angle'] for r in reps]
    lengths = np.array([r['length'] for r in reps], dtype=float)
    xs_rep = np.array([r['x'] for r in reps], dtype=float)
    ys_rep = np.array([r['y'] for r in reps], dtype=float)

    # spatial dispersion: positional std (use combined std of x and y)
    pos_std_x = float(np.std(xs_rep)) if xs_rep.size > 0 else 0.0
    pos_std_y = float(np.std(ys_rep)) if ys_rep.size > 0 else 0.0
    pos_std = float(np.sqrt(pos_std_x ** 2 + pos_std_y ** 2))

    # angle consensus (Mode 1)
    chosen_signed, debug = pick_consensus_angle_signed(raw_angles)

    # angular standard deviation (wrap-aware)
    if len(raw_angles) > 0:
        angs_rad = np.unwrap(np.deg2rad(np.array(raw_angles, dtype=float)))
        ang_std_deg = float(np.rad2deg(np.std(angs_rad)))
    else:
        ang_std_deg = 0.0

    # Determine which model provided the chosen angle
    # find the rep whose angle equals chosen_signed (may be multiple; pick first)
    dominant_model = None
    for model_id, rep in per_model.items():
        if float(rep['angle']) == float(chosen_signed):
            dominant_model = model_id
            break
    if dominant_model is None:
        # fallback: choose model with minimal distance to centroid
        dominant_model = sorted(per_model.items(), key=lambda kv: kv[1]['d2'])[0][0]

    consensus = {
        "n_models": len(unique_models),
        "models": sorted(list(unique_models)),
        "x": float(np.median(xs_rep)),
        "y": float(np.median(ys_rep)),
        "angle": float(chosen_signed),
        "angle_std_deg": float(ang_std_deg),
        "length": float(np.median(lengths)) if lengths.size>0 else 0.0,
        "length_std": float(np.std(lengths)) if lengths.size>0 else 0.0,
        "pos_std": float(pos_std),
        "support_count": len(items),
        "dominant_model": dominant_model,
        "support_fraction": float(len(unique_models)) / float(max(1, total_models))
    }

    print(f"    Cell center ({consensus['x']:.1f},{consensus['y']:.1f}) -> "
          f"(angle) {abs(consensus['angle']):.2f}° ±{consensus['angle_std_deg']:.2f}°, pos_std {consensus['pos_std']:.2f}, models {consensus['models']}")
    return consensus

# -------------------------
# Visualization helpers
# -------------------------
def alpha_blend(base_img, overlay_img, alpha):
    """
    Add overlay_img over base_img ONLY where overlay is nonzero.
    Does NOT dim the base image.
    """
    base = base_img.astype(np.float32)
    overlay = overlay_img.astype(np.float32)

    mask = np.any(overlay > 0, axis=2, keepdims=True).astype(np.float32)

    result = base.copy()
    result = result * (1 - mask * alpha) + overlay * (mask * alpha)

    return result.clip(0, 255).astype(np.uint8)

def parse_color_arg(s):
    """Parse a color argument which may be either:
      - hex string: '#RRGGBB' or 'RRGGBB'
      - comma separated 'B,G,R' (integers 0..255)
      - tuple/list (B,G,R)
    Returns (B,G,R) tuple of ints.
    """
    if s is None:
        return None
    if isinstance(s, (tuple, list)) and len(s) == 3:
        return tuple(int(x) for x in s)
    s = str(s).strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) == 6 and all(c in '0123456789abcdefABCDEF' for c in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 3:
            b, g, r = [int(float(p)) for p in parts]
            return (b, g, r)
    raise ValueError(f"Cannot parse color argument: {s}")

def polygon_sector(cx, cy, ang_start_deg, ang_end_deg, radius, num_points=20):
    """
    Generate a polygon approximating a wedge/sector.
    cx, cy : center
    ang_start_deg, ang_end_deg : angles in degrees (image coordinates)
    radius : radius of the wedge
    num_points : number of points along the arc
    """
    angles = np.linspace(math.radians(ang_start_deg),
                         math.radians(ang_end_deg), num_points)
    pts = [[int(round(cx)), int(round(cy))]]  # start with center
    for a in angles:
        x = int(round(cx + radius * math.cos(a)))
        y = int(round(cy + radius * math.sin(a)))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    return pts

def draw_consensus_on_image(frame_gray_or_color, consensus_list,
                            band_alpha=0.25, thickness=3,
                            default_band_color=(200,200,255),
                            default_rod_color=(80,80,255),
                            max_wedge_radius=30):
    """
    Draw consensus overlays with symmetric wedge-shaped angular uncertainty and pink halo.

    Parameters:
      - frame_gray_or_color: single image (H x W) or (H x W x 3)
      - consensus_list: list of consensus dicts (see aggregate_cluster_pick_signed)
      - band_alpha: alpha used for blending wedge+halo overlay
      - thickness: rod line thickness
      - default_band_color / default_rod_color: fallback BGR tuples
      - max_wedge_radius: maximum wedge radius in px to avoid enormous wedges
    """
    if frame_gray_or_color is None:
        return None

    # Ensure 3-channel color image
    if frame_gray_or_color.ndim == 2:
        base_vis = cv2.cvtColor(frame_gray_or_color, cv2.COLOR_GRAY2BGR)
    else:
        base_vis = frame_gray_or_color.copy()
    H, W = base_vis.shape[:2]

    # Overlay for wedge + halo
    overlay = np.zeros_like(base_vis, dtype=np.uint8)

    # Colors (BGR)
    pink_bgr = (255, 80, 80)   # halo
    red_bgr  = (255, 0, 0)     # red center

    for c in consensus_list:
        cx, cy = float(c["x"]), float(c["y"])
        pos_std = float(c.get("pos_std", 0.0))
        ang_std = float(c.get("angle_std_deg", 0.0))
        signed_angle_deg = float(c["angle"])
        L = float(c.get("length", 20.0))

        # Per-consensus colors
        band_color = c.get('band_color', default_band_color)
        rod_color  = c.get('rod_color', default_rod_color)
        if isinstance(band_color, str):
            band_color = parse_color_arg(band_color)
        if isinstance(rod_color, str):
            rod_color = parse_color_arg(rod_color)

        # -----------------------------
        # Draw symmetric wedge sector around rod
        # -----------------------------
        if ang_std > 0.1:
            # radius scaled from L and pos_std but clamped
            wedge_radius = max(L * 0.6, pos_std * 2.5, 20.0)
            wedge_radius = min(wedge_radius, max_wedge_radius)  # clamp to maximum

            # wedge covers signed_angle_deg +/- ang_std
            ang_start = signed_angle_deg - ang_std
            ang_end   = signed_angle_deg + ang_std
            pts = polygon_sector(cx, cy, ang_start, ang_end, wedge_radius)
            cv2.fillPoly(overlay, [pts], band_color)

            # opposite wedge (rotated 180 deg) — makes uncertainty symmetric across rod axis
            ang_start_op = signed_angle_deg + 180.0 - ang_std
            ang_end_op   = signed_angle_deg + 180.0 + ang_std
            pts_op = polygon_sector(cx, cy, ang_start_op, ang_end_op, wedge_radius)
            cv2.fillPoly(overlay, [pts_op], band_color)

        # -----------------------------
        # Pink halo behind red center
        # -----------------------------
        MAX_R = 15
        halo_radius = int(round(max(3.0, min(MAX_R, pos_std * 2.5))))
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cx_i = max(0, min(W-1, cx_i))
        cy_i = max(0, min(H-1, cy_i))
        cv2.circle(overlay, (cx_i, cy_i), halo_radius, pink_bgr, -1)

    # Blend overlay (wedge + halo) over base image
    vis = alpha_blend(base_vis, overlay, band_alpha)

    # -----------------------------
    # Draw rods and red centers on top
    # -----------------------------
    for c in consensus_list:
        cx, cy = float(c["x"]), float(c["y"])
        signed_angle_deg = float(c["angle"])
        L = float(c.get("length", 20.0))
        pos_std = float(c.get("pos_std", 0.0))

        rod_color  = c.get('rod_color', default_rod_color)
        if isinstance(rod_color, str):
            rod_color = parse_color_arg(rod_color)

        ang_rad = math.radians(signed_angle_deg)
        dx, dy = (L/2.0)*math.cos(ang_rad), (L/2.0)*math.sin(ang_rad)

        # Integer coordinates
        x1 = int(round(cx - dx)); y1 = int(round(cy - dy))
        x2 = int(round(cx + dx)); y2 = int(round(cy + dy))
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))

        # Draw rod
        cv2.line(vis, (x1, y1), (x2, y2), rod_color, thickness)

        # Draw red center
        MAX_R = 15
        pos_radius = int(round(max(3.0, min(MAX_R, pos_std * 2.5))))
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cx_i = max(0, min(W-1, cx_i))
        cy_i = max(0, min(H-1, cy_i))
        center_r = max(2, int(round(pos_radius*0.45)))
        cv2.circle(vis, (cx_i, cy_i), center_r, red_bgr, -1)

    return vis

# -------------------------
# CSV writer for global summary
# -------------------------
def write_global_summary(all_consensus_rows, movie_name, save_dir):
    """
    all_consensus_rows: list of dicts with keys matching fieldnames below.
    movie_name: the nam to use in the filename
    save_dir: directory to write CSV into
    """
    csv_path = os.path.join(save_dir, f"post_processed_{movie_name}_summary.csv")
    fieldnames = [
        "cell_id","frame","x","y","pos_std","angle_deg","angle_std_deg",
        "length","length_std","n_models","contributing_models",
        "dominant_model","support_fraction"
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_consensus_rows:
            w.writerow(row)
    print(f"[INFO] Wrote CSV summary: {csv_path}")

# -------------------------
# Main processing (keeps alignment) with simple temporal deduplication
# -------------------------
def process_all_frames(all_dets, image_stack, args):
    """
    Main per-frame processing loop:
      - cluster model detections spatially
      - aggregate cluster into consensus (requires min_models distinct model support)
      - temporal dedup (drop repeated near-duplicates across consecutive frames)
      - save per-frame consensus npy in chosen_divisions/
      - render visualization frame and collect frames for final TIFF
      - accumulate a global list of consensus rows for CSV
    """
    # 0/1-based key fix: if keys start at 0 but not 1, shift them to 1-based so loop 1..n works.
    keys = sorted(all_dets.keys())
    if (len(keys) > 0) and (0 in keys) and (1 not in keys):
        print("[INFO] Detected 0-based frame keys in detections; shifting to 1-based indexing for alignment.")
        new_all = defaultdict(list)
        for k, v in all_dets.items():
            new_all[k + 1] = v
        all_dets.clear()
        all_dets.update(new_all)

    if image_stack is not None and hasattr(image_stack, "shape"):
        if image_stack.ndim == 2:
            n_frames = 1
        else:
            n_frames = image_stack.shape[0]
    else:
        if len(all_dets) == 0:
            print("[ERROR] No image stack and no detections; nothing to do.")
            return
        n_frames = int(max(all_dets.keys()))

    print(f"[INFO] Processing {n_frames} frames (will produce exactly {n_frames} output frames).")

    nam = Path(args.image_name).name

    chosen_npy_dir = os.path.join(args.save_dir, "chosen_divisions")
    os.makedirs(chosen_npy_dir, exist_ok=True)

    vis_frames = []
    all_consensus_rows = []  # accumulate rows for global CSV

    # Temporal dedup parameters
    temporal_radius = 8.0  # pixels
    temporal_angle_tol = 20.0  # degrees
    prev_centers = []

    # default colors parsed from args
    default_rod_color = parse_color_arg(args.rod_color) if getattr(args, 'rod_color', None) else (80,80,255)
    default_band_color = parse_color_arg(args.band_color) if getattr(args, 'band_color', None) else (200,200,255)

    # Per-frame loop
    cell_global_id = 0
    for fidx in range(1, n_frames + 1):
        items = all_dets.get(fidx, [])
        print(f"[INFO] Frame {fidx}: total detections across models = {len(items)}")
        pts = np.array([[d[1], d[2]] for d in items], dtype=float) if len(items) > 0 else np.empty((0, 2))

        # Spatial clustering
        labels = cluster_hdbscan(pts, eps=args.eps, min_cluster_size=2, min_samples=1)
        consensus_list = []
        if labels.size > 0:
            for lab in np.unique(labels):
                mask = (labels == lab)
                idxs = np.where(mask)[0]
                cluster_items = [items[i] for i in idxs]
                c = aggregate_cluster_pick_signed(cluster_items, min_models=args.min_models, total_models=args.num_models)
                if c is not None:
                    # attach default colors so drawing uses them
                    c['band_color'] = default_band_color
                    c['rod_color'] = default_rod_color
                    consensus_list.append(c)

        # Temporal deduplication against prev_centers
        filtered = []
        for c in consensus_list:
            cx, cy = c['x'], c['y']
            ang = c['angle']
            is_dup = False
            for p in prev_centers:
                dx = cx - p['x']; dy = cy - p['y']
                if (dx*dx + dy*dy) <= (temporal_radius ** 2):
                    diff = abs(((ang - p['angle'] + 180) % 360) - 180)
                    if diff <= temporal_angle_tol:
                        is_dup = True
                        break
            if not is_dup:
                filtered.append(c)
        consensus_list = filtered
        prev_centers = consensus_list

        # Save per-frame consensus NPY
        np.save(os.path.join(chosen_npy_dir, f"processed_division_{fidx}.npy"), consensus_list)

        # Prepare base image for rendering
        if image_stack is not None and image_stack.ndim >= 2:
            if image_stack.ndim == 2:
                base_img = image_stack
            else:
                idx0 = fidx - 1
                if 0 <= idx0 < image_stack.shape[0]:
                    base_img = image_stack[idx0]
                else:
                    base_img = image_stack[0]
        else:
            base_img = np.zeros((512, 512), dtype=np.uint8)

        # Render visualization for this frame
        vis = draw_consensus_on_image(base_img, consensus_list, band_alpha=args.band_alpha,
                                      thickness=args.thickness,
                                      default_band_color=default_band_color,
                                      default_rod_color=default_rod_color,
                                      max_wedge_radius=args.max_wedge_radius)
        vis_frames.append(vis)

        # Append consensus rows for CSV summary
        for c in consensus_list:
            cell_global_id += 1
            row = {
                "cell_id": cell_global_id,
                "frame": fidx,
                "x": c["x"],
                "y": c["y"],
                "pos_std": c["pos_std"],
                "angle_deg": c["angle"],
                "angle_std_deg": c["angle_std_deg"],
                "length": c["length"],
                "length_std": c["length_std"],
                "n_models": c["n_models"],
                "contributing_models": ",".join([str(m) for m in c["models"]]),
                "dominant_model": c.get("dominant_model", ""),
                "support_fraction": c.get("support_fraction", 0.0)
            }
            all_consensus_rows.append(row)

    # Write final TIFF and CSV
    out_post = os.path.join(args.save_dir, f"post_processed_{nam}.tiff")
    imageio.mimwrite(out_post, vis_frames, format='TIFF')
    print(f"[DONE] Wrote TIFF (frames={len(vis_frames)}): {out_post}")

    write_global_summary(all_consensus_rows, nam, args.save_dir)

    print("[DONE] NPY outputs in:", chosen_npy_dir)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate multi-model division detections: final consensus and visualization")
    p.add_argument("--output_root", required=True, help="Root folder containing model outputs (folders named <image_nam>_1 ...)")
    p.add_argument("--image_name", required=True, help="Image filename used by oracle (e.g. myMovie.tif)")
    p.add_argument("--image_stack", required=True, help="Path to original tiff stack for visualization")
    p.add_argument("--save_dir", required=True, help="Directory to save outputs (TIFF and npys)")
    p.add_argument("--eps", type=float, required=True, help="eps for HDBSCAN.cluster_selection_epsilon and DBSCAN fallback")
    p.add_argument("--min_models", type=int, default=4, help="Minimum distinct models required to accept a cluster")
    p.add_argument("--num_models", type=int, default=8, help="Number of model folders expected")
    p.add_argument("--angle_mode", choices=["auto", "degrees", "radians"], default="auto", help="Force or auto-detect angle units")
    p.add_argument("--rod_color", type=str, default=None, help="Default rod color: '#RRGGBB' or 'B,G,R'")
    p.add_argument("--band_color", type=str, default=None, help="Default band color: '#RRGGBB' or 'B,G,R'")
    p.add_argument("--band_alpha", type=float, default=0.25, help="Alpha for band blending")
    p.add_argument("--thickness", type=int, default=3, help="Rod thickness (pixels)")
    p.add_argument("--max_wedge_radius", type=float, default=30.0, help="Maximum wedge radius in pixels to avoid huge wedges")
    return p.parse_args()

def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = args.save_dir + "_" + stamp
    os.makedirs(args.save_dir, exist_ok=True)

    all_dets = load_all_model_detections(args.output_root, args.image_name, num_models=args.num_models)
    if len(all_dets) == 0:
        print("[WARN] No detections found across models. Blank TIFFs will be produced if image_stack provided.")

    detect_angle_units_and_convert(all_dets, mode=args.angle_mode)

    image_stack = None
    if args.image_stack and os.path.isfile(args.image_stack):
        try:
            image_stack = skio.imread(args.image_stack)
            print(f"[INFO] Loaded image stack {args.image_stack} shape={image_stack.shape}")
        except Exception as e:
            print(f"[WARN] Failed to read image stack: {e}")
            image_stack = None
    else:
        print("[WARN] image_stack not found or invalid; visuals will be blank frames.")

    process_all_frames(all_dets, image_stack, args)

if __name__ == "__main__":
    main()
