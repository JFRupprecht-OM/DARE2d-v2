"""Thin in-process API over the DARE2D inference + consensus pipeline.

The shipping DARE2D entry points are CLIs (click for inference, argparse for
postprocessing) that the notebook drives via ``subprocess``. A napari widget
needs to call the same logic *in process* (image arrays in memory, progress
callbacks, no disk round-trip), so this module exposes a few plain functions
that REUSE the existing code rather than reimplementing it:

  - model architectures are built via the project's Hydra configs;
  - the per-frame sliding-window helpers are imported from the inference
    script (``inference_strategy``, ``crop_img_from_center``);
  - the consensus aggregation is imported from the postprocessing script.

DARE2d-main/ itself is left untouched.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

# segmentation_models must see the framework BEFORE it is imported anywhere.
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")

# --- locate the DARE2D core that this plugin wraps -------------------------
# Merged-project layout:
#   DARE2d/                              <- PROJECT_ROOT (repo root = the DARE2D core)
#     dare2d/  scripts/  config/         (the wrapped code + Hydra config)
#     regression_checkpoints/ segmentation_checkpoints/
#     napari-dare2d/napari_dare2d/_api.py  <- this file
# ponytail: paths are derived from __file__ relative to that fixed layout.
# Ceiling: move the package elsewhere and you must pass config_dir / checkpoint
# dirs explicitly (the widget already exposes the checkpoint dirs as inputs).
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]
_REPO = PROJECT_ROOT
CONFIG_DIR = str(_REPO / "config")
# Curated demo checkpoints live under models/demo/<dataset>/ (read-only; retraining
# writes new runs to sibling models/<run>/ folders, never here). neuroepithelium is the
# first demo dataset; others can be added under models/demo/ alongside it.
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REG_DIR = MODELS_DIR / "demo" / "neuroepithelium" / "regression_checkpoints"
DEFAULT_SEG_DIR = MODELS_DIR / "demo" / "neuroepithelium" / "segmentation_checkpoints"
# Dataset root produced by the "Download DARE2D data" button (and consumed by the
# retraining widget): data/demo/neuroepithelium/set_{1..8}/, each holding
# the movie .tiff + ground-truth division_position*.npy.
DATA_DIR = PROJECT_ROOT / "data" / "demo" / "neuroepithelium"
# Annotations viewer defaults to set 8's ground truth, in the downloaded layout.
DEFAULT_ANNOT_DIR = DATA_DIR / "set_8"

# scripts/ has no __init__.py; it imports as an implicit namespace package
# once the repo root is on sys.path.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

# IMPORTANT: keep this module import-light. The heavy stack — cv2, hydra/omegaconf, the
# dare2d core (which pulls in TensorFlow via dare2d.evaluation) and the scripts/ helpers — is
# imported LAZILY inside the functions that use it (_build_model, infer_stack, consensus), NOT
# here. The napari widgets import this module at construction time only for the path constants
# + the pure (numpy-only) helpers below; importing the ~11 s TF/hydra stack here would freeze
# napari for that long on first widget open. Deferred, it loads on the first Run instead, in
# the worker thread.

__all__ = [
    "build_models",
    "infer_stack",
    "run_ensemble",
    "consensus",
    "detections_to_points",
    "detections_to_vectors",
    "to_layer_data",
    "annotation_pairs",
    "annotations_to_layer_data",
    "read_stack",
    "default_movie",
    "find_checkpoints",
    "parse_sets",
    "resolve_frames",
    "CONFIG_DIR",
    "MODELS_DIR",
    "DATA_DIR",
    "DEFAULT_REG_DIR",
    "DEFAULT_SEG_DIR",
    "DEFAULT_ANNOT_DIR",
]


# ---------------------------------------------------------------------------
# Input resolution helpers (used by the widget; pure / no TF)
# ---------------------------------------------------------------------------
def parse_sets(spec, available=range(1, 9)):
    """Parse a model-set spec like ``"1-8"``, ``"1,3,5"`` or ``"8"`` -> sorted ints.

    Values are intersected with ``available`` so a bad entry can't request a
    non-existent set.
    """
    avail = set(available)
    out = set()
    for tok in str(spec).replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    sets = sorted(out & avail)
    if not sets:
        raise ValueError(f"no valid model sets in {spec!r} (available: {sorted(avail)})")
    return sets


def _sets_in_dir(d, ckpt_name):
    """Set numbers of ``<d>/checkpoints_set_{n}_all_but_target/<ckpt_name>`` that exist."""
    import re
    d = Path(d)
    found = set()
    for sub in d.glob("checkpoints_set_*_all_but_target"):
        m = re.fullmatch(r"checkpoints_set_(\d+)_all_but_target", sub.name)
        if m and (sub / ckpt_name).exists():
            found.add(int(m.group(1)))
    return found


def discover_sets(reg_dir, seg_dir, backend="pytorch"):
    """Auto-detect the model sets present in the selected checkpoint folders.

    Used when the user leaves the model-set field blank: scan ``reg_dir`` and ``seg_dir``
    for ``checkpoints_set_{n}_all_but_target`` folders holding the right checkpoint file
    (``best.pt`` for pytorch, ``best.h5`` for keras) and return the sorted set numbers
    present in BOTH stages. Raises ValueError (with the paths) if none are found.
    """
    ckpt = "best.pt" if backend == "pytorch" else "best.h5"
    common = sorted(_sets_in_dir(reg_dir, ckpt) & _sets_in_dir(seg_dir, ckpt))
    if not common:
        raise ValueError(
            f"no model sets found automatically: looked for "
            f"checkpoints_set_*_all_but_target/{ckpt} under both\n  {reg_dir}\n  {seg_dir}\n"
            f"Type a set number (e.g. 8) or range (1-7), or point the checkpoint folders at "
            f"a run dir that contains them (or fetch the weights via 'Download DARE2D data')."
        )
    return common


def find_checkpoints(reg_dir, seg_dir, sets):
    """Resolve ``best.h5`` paths for the given model ``sets``.

    Expects the shipping layout ``<dir>/checkpoints_set_{n}_all_but_target/best.h5``.
    Raises FileNotFoundError naming the first missing file.
    """
    reg_dir, seg_dir = Path(reg_dir), Path(seg_dir)
    reg, seg = [], []
    for n in sets:
        r = reg_dir / f"checkpoints_set_{n}_all_but_target" / "best.h5"
        s = seg_dir / f"checkpoints_set_{n}_all_but_target" / "best.h5"
        if not r.exists():
            raise FileNotFoundError(f"regression checkpoint missing: {r}")
        if not s.exists():
            raise FileNotFoundError(f"segmentation checkpoint missing: {s}")
        reg.append(str(r))
        seg.append(str(s))
    return reg, seg


def resolve_frames(n_frames, start=0, end=-1):
    """Clamp a [start, end] frame range to ``[0, n_frames)`` -> list of indices.

    ``end == -1`` means the last frame. ``end`` is inclusive.
    """
    start = max(0, int(start))
    end = n_frames - 1 if end is None or end < 0 else min(int(end), n_frames - 1)
    if start > end:
        raise ValueError(f"empty frame range: start={start} > end={end}")
    return list(range(start, end + 1))


# ---------------------------------------------------------------------------
# Model building (Hydra)
# ---------------------------------------------------------------------------
def _build_model(experiment: str, weights: str | None, config_dir: str = CONFIG_DIR):
    """Instantiate one model wrapper from its Hydra experiment config.

    Uses ``initialize_config_dir`` (absolute path) instead of the script's
    ``initialize(config_path="../../config")`` which is resolved relative to
    the *calling file* and breaks when called from a plugin. GlobalHydra is a
    process-wide singleton, so it is cleared around each build to stay
    re-entrant across repeated runs inside a long-lived napari session.
    """
    import hydra  # deferred (heavy) — see the module-top note on import-light _api
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={experiment}"],
                return_hydra_config=True,
            )
            HydraConfig.instance().set_config(cfg)
            cfg = OmegaConf.create(cfg)
            model = hydra.utils.instantiate(cfg.model)
    finally:
        GlobalHydra.instance().clear()

    if weights is not None:
        # .h5 are weights only (load_weights, not load_model) -> no custom
        # objects needed; the architecture above (incl. sm.Unet) must match.
        model.model.load_weights(weights)
    return model


def build_models(reg_ckpt: str, seg_ckpt: str, config_dir: str = CONFIG_DIR):
    """Return ``(regression_model, segmentation_model)`` wrappers."""
    reg = _build_model("regression2d", reg_ckpt, config_dir)
    seg = _build_model("segmentation2d", seg_ckpt, config_dir)
    return reg, seg


# ---------------------------------------------------------------------------
# Single-model inference over a stack
# ---------------------------------------------------------------------------
def infer_stack(
    stack,
    reg_model,
    seg_model,
    frames=None,
    progress_cb=None,
    window_size: int = 256,
    crop_size: int = 64,
    threshold: float = 0.5,
):
    """Run the two-stage detection on a ``(T, Y, X)`` stack.

    Returns ``{frame_index: [ {x, y, angle, length}, ... ]}`` keyed by the
    0-based frame index. This is the body of ``multistage_detection2d.main``
    with the disk I/O and visualization stripped out. ``threshold`` is the
    probability cutoff on the segmentation map (default 0.5, the original value).

    ponytail: assumes 8-bit input (cv2.equalizeHist + /255), like the original
    script. Ceiling: 16-bit stacks would need rescaling first.
    """
    import cv2  # deferred (heavy) — see the module-top note on import-light _api
    from dare2d.datamodule.post_processing.regression2d_pp import convert_values
    from dare2d.evaluation.center_metrics import extract_centers
    from scripts.inference.multistage_detection2d import (
        crop_img_from_center,
        inference_strategy,
    )

    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {stack.shape}")
    if stack.dtype != np.uint8:
        raise ValueError(f"expected 8-bit stack, got dtype {stack.dtype}")

    n_frames = stack.shape[0]
    if frames is None:
        frames = range(n_frames)
    frames = list(frames)
    half = crop_size // 2

    results = {}
    for n, i in enumerate(frames):
        prev = stack[i - 1] if i > 0 else stack[i]
        curr = stack[i]
        nxt = stack[i + 1] if i < n_frames - 1 else stack[i]

        prev = cv2.equalizeHist(prev)
        curr = cv2.equalizeHist(curr)
        nxt = cv2.equalizeHist(nxt)

        x = np.stack([prev, curr, nxt], axis=-1).astype(np.float32) / 255.0

        seg_raw = inference_strategy(x, seg_model, window_size=window_size)
        seg_bin = np.where(seg_raw > threshold, 255, 0).astype(np.uint8)
        centers = extract_centers(seg_bin)

        xp = np.pad(
            x, ((half, half), (half, half), (0, 0)), mode="constant", constant_values=0
        )

        dets = []
        for c in centers:
            inverted_center = (c[1], c[0])
            crop = crop_img_from_center(xp, inverted_center, half)
            length_pred, angle_pred = reg_model.model.predict(
                np.expand_dims(crop, axis=0), verbose=0
            )
            values = convert_values(length_pred, angle_pred, im_size=crop_size)
            dets.append(
                {
                    "x": int(c[0]),
                    "y": int(c[1]),
                    "angle": float(values[0, 1]),
                    "length": float(values[0, 0]),
                }
            )
        results[i] = dets
        if progress_cb is not None:
            progress_cb(n + 1, len(frames))

    return results


# ---------------------------------------------------------------------------
# Ensemble (8 model sets) -> detections in the shape the consensus expects
# ---------------------------------------------------------------------------
def run_ensemble(stack, reg_ckpts, seg_ckpts, frames=None, config_dir=CONFIG_DIR,
                 progress_cb=None):
    """Run every model set and collect detections.

    Returns ``{frame_1based: [(model_id, x, y, angle, length), ...]}`` -- the
    exact structure ``consensus`` (and the original ``process_all_frames``)
    consumes. Frame keys are 1-based to match the CLI's ``division_position{i+1}``.
    """
    import tensorflow.keras.backend as K

    if len(reg_ckpts) != len(seg_ckpts):
        raise ValueError("reg_ckpts and seg_ckpts must have the same length")

    all_dets = defaultdict(list)
    n_models = len(reg_ckpts)
    for m, (r, s) in enumerate(zip(reg_ckpts, seg_ckpts), start=1):
        reg, seg = build_models(r, s, config_dir)
        res = infer_stack(stack, reg, seg, frames=frames)
        for i, dets in res.items():
            for d in dets:
                all_dets[i + 1].append(
                    (m, float(d["x"]), float(d["y"]), float(d["angle"]), float(d["length"]))
                )
        if progress_cb is not None:
            progress_cb(m, n_models)
        del reg, seg
        K.clear_session()  # free graph between model sets (CPU/RAM hygiene)
    return dict(all_dets)


# ---------------------------------------------------------------------------
# Consensus across models (in memory, no file I/O / no drawing)
# ---------------------------------------------------------------------------
def consensus(all_dets, n_frames, eps=10, min_models=6, num_models=8, angle_mode="auto",
              min_cluster_size=2, min_samples=1):
    """Aggregate per-model detections into per-frame consensus detections.

    Returns ``{frame_1based: [consensus_dict, ...]}`` where each consensus dict
    has x, y, angle, angle_std_deg, length, length_std, pos_std, n_models, ...

    ponytail: mirrors the clustering/aggregation core of
    ``scripts.postprocessing.main.process_all_frames`` but drops the wedge/halo
    drawing, TIFF/CSV writing and temporal dedup, which a napari overlay does
    not need. Reuses the same aggregation primitives (same picked model, same
    median/std/support fields as the CLI). The ONLY difference: ``angle`` is
    returned in the MODEL convention (theta from +row toward +col, exactly like
    ``infer_stack`` dicts), not in main.py's image convention (90 - theta) that
    its cv2 renderer needs -- so ``to_layer_data`` / ``save_results`` draw
    consensus and single-set dicts alike.
    """
    from scripts.postprocessing.main import (  # deferred (heavy); see module-top note
        aggregate_cluster_pick_signed,
        cluster_hdbscan,
        detect_angle_units_and_convert,
    )

    # work on a shallow copy: detect_angle_units_and_convert mutates in place.
    dets = {k: list(v) for k, v in all_dets.items()}
    detect_angle_units_and_convert(dets, mode=angle_mode)

    out = {}
    for fidx in range(1, n_frames + 1):
        items = dets.get(fidx, [])
        pts = (
            np.array([[d[1], d[2]] for d in items], dtype=float)
            if len(items) > 0
            else np.empty((0, 2))
        )
        labels = cluster_hdbscan(pts, eps=eps, min_cluster_size=min_cluster_size,
                                 min_samples=min_samples)
        cons = []
        if labels.size > 0:
            for lab in np.unique(labels):
                idxs = np.where(labels == lab)[0]
                cluster_items = [items[i] for i in idxs]
                c = aggregate_cluster_pick_signed(
                    cluster_items, min_models=min_models, total_models=num_models
                )
                if c is not None:
                    # main.py aggregates in ITS image convention (ang_img = fold(90 - theta),
                    # measured from +x/col toward +y/row), which only its own cv2 renderer
                    # needs (draw_consensus_on_image: dx = cos, dy = sin). napari's
                    # detections_to_vectors and _data.save_results expect the MODEL
                    # convention theta (from +row toward +col, as convert_values and the
                    # single-set dicts), so map the picked angle back. Exact inverse of
                    # fold(90 - theta) on (-90, 90]; every other field (picked model,
                    # angle_std_deg, medians, support) stays exactly as the CLI computes it.
                    a = float(c["angle"])
                    c["angle"] = 90.0 - a if a >= 0.0 else -90.0 - a
                    cons.append(c)
        out[fidx] = cons
    return out


# ---------------------------------------------------------------------------
# Map detections / consensus -> napari layer data
# ---------------------------------------------------------------------------
# Both infer_stack detections and consensus() dicts share the keys
# {x, y, angle, length} AND the same angle convention (theta from +row toward
# +col, as decoded by convert_values): consensus() maps main.py's image-convention
# angle (90 - theta, needed only by its cv2 renderer) back before returning, so the
# same mapping serves both. NOTE the standalone CLI's outputs (post_processed CSV
# ``angle_deg``, chosen_divisions/*.npy) keep the image convention and are NOT
# interchangeable with these dicts without that mapping. The other difference
# is the frame-key base: infer_stack keys are 0-based (use frame_base=0),
# consensus keys are 1-based (use frame_base=1). napari frame index t = key - frame_base.
#
# Coordinate convention (locked in step 3): detection x = COLUMN, y = ROW, so a
# napari point over a (T,Y,X) stack is (t, y, x). The division axis direction in
# (row, col) is (cos(angle), sin(angle)) -- matching the script's project_point.

_PROP_KEYS = ("angle", "length", "angle_std_deg", "length_std", "pos_std",
              "n_models", "support_fraction")


def _points_border_key():
    """napari renamed the Points marker outline ``edge_*`` -> ``border_*`` in 0.5.

    Return the kwarg name the *installed* napari expects, so the same plugin works
    under napari 0.4.18 (TF/Keras env) and >=0.5 (modern env). Uses package
    metadata, not a napari import, to stay napari-free here. (Vectors kept
    ``edge_color``/``edge_width`` in 0.5, so only Points needs this.)
    """
    try:
        from importlib.metadata import version

        major, minor = (int(x) for x in version("napari").split(".")[:2])
        if (major, minor) >= (0, 5):
            return "border_color"
    except Exception:
        pass
    return "edge_color"


def _iter_dets(per_frame, frame_base):
    for key in sorted(per_frame):
        t = int(key) - frame_base
        for d in per_frame[key]:
            yield t, d


def detections_to_points(per_frame, frame_base=0):
    """``{frame: [{x,y,angle,length,...}]}`` -> (points ``(N,3)`` ``[t,y,x]``, properties dict).

    ``properties`` carries whatever of _PROP_KEYS each detection has (angle &
    length always; consensus dicts also bring the std/support fields), so a
    widget can colour/size points by them.
    """
    pts = []
    props = {k: [] for k in _PROP_KEYS}
    present = set()
    for t, d in _iter_dets(per_frame, frame_base):
        pts.append((t, float(d["y"]), float(d["x"])))
        for k in _PROP_KEYS:
            if k in d:
                props[k].append(float(d[k]))
                present.add(k)
    points = np.asarray(pts, dtype=float).reshape(-1, 3)
    properties = {k: np.asarray(props[k], dtype=float) for k in present}
    return points, properties


def detections_to_vectors(per_frame, frame_base=0):
    """``{frame: [{x,y,angle,length}]}`` -> vectors ``(N,2,3)``.

    ``v[:,0]`` = origin ``(t, y, x)`` at one rod end, ``v[:,1]`` = full direction
    ``(0, cos·L, sin·L)``. With a Vectors layer ``length=1`` this draws the rod
    from ``centre-half`` to ``centre+half`` (same segment as the script's p1-p2),
    i.e. the rod passes through the detection centre.
    """
    vecs = []
    for t, d in _iter_dets(per_frame, frame_base):
        th = np.radians(float(d["angle"]))
        L = float(d["length"])
        drow = np.cos(th) * L
        dcol = np.sin(th) * L
        origin = (t, float(d["y"]) - drow / 2.0, float(d["x"]) - dcol / 2.0)
        direction = (0.0, drow, dcol)
        vecs.append([origin, direction])
    return np.asarray(vecs, dtype=float).reshape(-1, 2, 3)


def to_layer_data(per_frame, frame_base=0, name="DARE2D", point_size=24,
                  edge_width=5, vector_style="line"):
    """Build napari ``LayerDataTuple``s: one Points (centres) + one Vectors (axes).

    Returns ``[(points, meta, "points"), (vectors, meta, "vectors")]`` -- directly
    returnable from a magicgui widget annotated ``-> List[LayerDataTuple]``.
    Pure data (no napari import needed here).
    """
    points, properties = detections_to_points(per_frame, frame_base)
    vectors = detections_to_vectors(per_frame, frame_base)
    points_meta = {
        "name": f"{name} centers",
        "size": point_size,
        "face_color": "red",
        _points_border_key(): "white",  # edge_color (0.4.18) / border_color (>=0.5)
    }
    if properties:
        points_meta["properties"] = properties
    vectors_meta = {
        "name": f"{name} axes",
        "edge_width": edge_width,
        "edge_color": "cyan",
        "length": 1,
        "vector_style": vector_style,  # "line" (no arrowhead) — the division axis
    }
    return [(points, points_meta, "points"), (vectors, vectors_meta, "vectors")]


# ---------------------------------------------------------------------------
# Ground-truth annotations: paired daughter cells per frame
# ---------------------------------------------------------------------------
# division_position{n}.npy holds an int (2K, 3) array of [row, col, frame] rows
# (i.e. [y, x, frame] -- the order the training pipeline reads; see
# annotator/preprocessing/format_gastru.py), where CONSECUTIVE rows are paired: row 2k and
# 2k+1 are the two daughter cells of one division. ``frame`` is 1-based (file n -> frame
# label n), so napari t = frame - frame_base (frame_base=1, same convention as consensus).

def _annot_file_num(p):
    digits = "".join(ch for ch in Path(p).stem if ch.isdigit())
    return int(digits) if digits else 0


def annotation_pairs(rows, frame_base=1):
    """``(2K, 3)`` ``[row, col, frame]`` rows -> list of ``((t,yA,xA), (t,yB,xB))`` pairs.

    The GT stores ``[row, col, frame]`` (``[y, x, frame]`` -- the order the training pipeline
    reads, see annotator/preprocessing/format_gastru.py), so the napari point is
    ``(t, row, col)``. Pairs consecutive rows (the two daughter cells); a trailing unpaired
    row is ignored. Pure / no napari, so it's unit-testable without files.
    """
    rows = np.atleast_2d(np.asarray(rows))
    out = []
    for k in range(0, len(rows) - 1, 2):
        ya, xa, fa = rows[k][:3]        # GT row = [row(y), col(x), frame]
        yb, xb, _ = rows[k + 1][:3]
        t = int(fa) - frame_base
        out.append(((t, float(ya), float(xa)), (t, float(yb), float(xb))))
    return out


def read_stack(path):
    """Read a ``(T, Y, X)`` image stack from a .tif/.tiff (tifffile, falling back to skimage.io)."""
    try:
        import tifffile
        return tifffile.imread(str(path))
    except Exception:
        from skimage import io as skio
        return skio.imread(str(path))


def default_movie(folder=DATA_DIR):
    """The demo movie that sits beside the ``set_N`` folders -- the .tif/.tiff directly inside
    ``folder`` (NOT inside a ``set_N`` subfolder). Returns its ``Path``, or ``None`` if absent.

    Robust to renaming: it globs by extension at the top level, so the specific filename can
    change as long as the movie stays in this folder (one top-level stack is expected).
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    # match by extension case-insensitively (uppercase .TIF/.TIFF on a case-sensitive FS)
    hits = sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    return hits[0] if hits else None


def annotations_to_layer_data(folder=DEFAULT_ANNOT_DIR, name="annotations",
                              point_size=12, show_links=True, frame_base=1):
    """Load ``division_position*.npy`` and build napari ``LayerDataTuple``s.

    Returns a Points layer (every annotated daughter cell at ``(t, y, x)``) and,
    if ``show_links``, a Vectors layer drawing a segment between each pair (the
    division: cell A -> cell B). Raises FileNotFoundError if the folder has none.
    """
    folder = Path(folder)
    files = sorted(folder.glob("division_position*.npy"), key=_annot_file_num)
    if not files:
        raise FileNotFoundError(f"no division_position*.npy in {folder}")

    pts, vecs = [], []
    for f in files:
        rows = np.load(f)  # plain int array; no pickle needed
        if np.asarray(rows).size == 0:
            continue
        for a, b in annotation_pairs(rows, frame_base):
            pts.append(a)
            pts.append(b)
            vecs.append([a, (0.0, b[1] - a[1], b[2] - a[2])])

    points = np.asarray(pts, dtype=float).reshape(-1, 3)
    points_meta = {"name": f"{name} cells", "size": point_size, "face_color": "yellow"}
    points_meta[_points_border_key()] = "black"
    out = [(points, points_meta, "points")]
    if show_links and vecs:
        vectors = np.asarray(vecs, dtype=float).reshape(-1, 2, 3)
        vectors_meta = {
            "name": f"{name} pairs",
            "edge_color": "magenta",
            "edge_width": 2,
            "length": 1,
            "vector_style": "line",
        }
        out.append((vectors, vectors_meta, "vectors"))
    return out
