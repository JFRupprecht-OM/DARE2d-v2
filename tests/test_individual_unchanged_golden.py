"""Golden-snapshot tests proving the INDIVIDUAL-model path is unchanged by consensus-only edits.

Three snapshots, written once BEFORE a change (``--write-golden``) and compared exactly afterwards:
  1. ``convert_values`` on fixed synthetic (length, cos2t, sin2t) batches      -> tests/golden/convert_values_v1.json
  2. ``detections_to_points`` / ``detections_to_vectors`` / ``to_layer_data`` on
     the verify_layers fixtures + an extended sweep (wrap angles, both frame bases) -> tests/golden/layer_mapping_v1.json
  3. (``--with-infer``) ``_api.infer_stack`` with the shipped model-set-8 weights on set_8 frames 27-29 and
     set_test frames 4-5 (torch by default; ``--backend keras`` in a separate, stub-free process)
     -> docs/audit/results/golden/infer_stack_<backend>_v1.json (machine-specific, git-ignored)

Run:  python tests/test_individual_unchanged_golden.py --write-golden [--with-infer]   (once, pre-change)
      python tests/test_individual_unchanged_golden.py [--with-infer]                  (after the change)
   or: python -m pytest tests/test_individual_unchanged_golden.py -q   (snapshots 1-2 only; infer skipped)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("SM_FRAMEWORK", "tf.keras")
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "napari-dare2d", _ROOT / "dare2d-torch"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

GOLDEN_DIR = _ROOT / "tests" / "golden"                          # committed JSON
LOCAL_GOLDEN = _ROOT / "docs" / "audit" / "results" / "golden"   # machine-specific, git-ignored

WRITE = "--write-golden" in sys.argv or os.environ.get("DARE2D_WRITE_GOLDEN") == "1"
WITH_INFER = "--with-infer" in sys.argv or os.environ.get("DARE2D_WITH_INFER") == "1"
BACKEND = (sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv
           else os.environ.get("DARE2D_BACKEND", "torch"))

from napari_dare2d import _api  # noqa: E402
from dare2d.datamodule.post_processing.regression2d_pp import convert_values  # noqa: E402


def _golden(path, current):
    path = Path(path)
    if path.exists() and not WRITE:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    if not path.exists() and not WRITE:
        raise AssertionError(f"golden missing: {path} - run with --write-golden BEFORE the change")
    if path.exists():
        print(f"  [warn] overwriting golden {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(current, fh, indent=1)
    print(f"  [golden written] {path}")
    return current


# --------------------------------------------------------------------------- 1. convert_values
def _cv_inputs():
    rng = np.random.default_rng(0)
    t = np.deg2rad(np.linspace(-90, 90, 37))
    unit = np.stack([np.cos(2 * t), np.sin(2 * t)], axis=1)
    angle = np.concatenate([unit, 0.3 * unit, 0.9 * unit, [[0.0, 0.0]], rng.normal(0, 0.7, (50, 2))]).astype(np.float32)
    length = np.concatenate([np.linspace(0, 1, 37)] * 3 + [[0.5], rng.uniform(0, 1, 50)]).astype(np.float32)
    return length, angle


def test_convert_values_golden():
    length, angle = _cv_inputs()
    out = convert_values(length, angle, 64)
    assert out.dtype == np.float32 and out.shape == (len(length), 2)
    # atan2 -> (-180, 180] halved gives [-90, 90]; -90.0 is reached exactly (float32 compare, no epsilon)
    assert np.all(out[:, 1] >= -90.0) and np.all(out[:, 1] <= 90.0)
    g = _golden(GOLDEN_DIR / "convert_values_v1.json",
                {"im_size": 64, "length": length.tolist(), "angle": angle.tolist(), "out": out.tolist()})
    assert np.array_equal(np.asarray(g["length"], np.float32), length)
    assert np.array_equal(np.asarray(g["angle"], np.float32), angle)
    assert np.array_equal(np.asarray(g["out"], np.float32), out), "convert_values output changed"


# --------------------------------------------------------------------------- 2. layer mapping
def _fixtures():
    base = {0: [{"x": 100, "y": 200, "angle": 0.0, "length": 40.0}],
            2: [{"x": 300, "y": 50, "angle": 90.0, "length": 20.0},
                {"x": 10, "y": 10, "angle": 45.0, "length": 10.0}]}
    ext = {}
    k = 0
    for a in (-89.9, -89.0, -45.0, -1e-6, 0.0, 1e-6, 45.0, 89.0, 89.9, 90.0):
        for L in (0.0, 10.0, 63.5):
            for (x, y) in ((0, 0), (1023, 0), (511, 1023)):
                ext.setdefault(k % 4, []).append({"x": x, "y": y, "angle": a, "length": L})
                k += 1
    cons_like = {1: [{"x": 5, "y": 6, "angle": 0.0, "length": 8.0, "angle_std_deg": 1.0, "n_models": 3,
                      "length_std": 0.5, "pos_std": 0.2, "support_fraction": 0.375}],
                 3: [{"x": 7.5, "y": 8.25, "angle": -30.0, "length": 12.0, "n_models": 6}]}
    return [("base", base, 0), ("base_fb1", base, 1), ("ext", ext, 0), ("cons_like", cons_like, 1)]


def _norm(obj):
    return json.loads(json.dumps(obj, default=str))


def test_layer_mapping_golden():
    snap = {}
    for name, per_frame, fb in _fixtures():
        pts, props = _api.detections_to_points(per_frame, fb)
        vecs = _api.detections_to_vectors(per_frame, fb)
        layers = _api.to_layer_data(per_frame, fb, name="G")
        snap[name] = {"points": pts.tolist(),
                      "properties": {k: np.asarray(v).tolist() for k, v in sorted(props.items())},
                      "vectors": vecs.tolist(),
                      "meta": [[_norm(meta), lt] for _, meta, lt in layers]}
    g = _golden(GOLDEN_DIR / "layer_mapping_v1.json", snap)
    assert set(g) == set(snap)
    for name in snap:
        assert np.array_equal(np.asarray(g[name]["points"], float), np.asarray(snap[name]["points"], float)), name
        assert np.array_equal(np.asarray(g[name]["vectors"], float), np.asarray(snap[name]["vectors"], float)), name
        assert set(g[name]["properties"]) == set(snap[name]["properties"]), name
        for k in snap[name]["properties"]:
            assert np.array_equal(np.asarray(g[name]["properties"][k], float),
                                  np.asarray(snap[name]["properties"][k], float)), (name, k)
        assert _norm(g[name]["meta"]) == _norm(snap[name]["meta"]), name


# --------------------------------------------------------------------------- 3. infer_stack
def test_infer_stack_golden():
    if not WITH_INFER:
        print("  SKIP test_infer_stack_golden (pass --with-infer)")
        return
    movie8 = _ROOT / "data" / "demo" / "neuroepithelium" / "set_8" / "siractinE2_14-03-23_1_post_z9-celldivisionlevel.tiff"
    movie_t = _ROOT / "data" / "demo" / "neuroepithelium" / "set_test" / "movie_M-z5.tif"
    reg_dir = _ROOT / "models" / "demo" / "neuroepithelium" / "regression_checkpoints" / "checkpoints_set_8_all_but_target"
    seg_dir = _ROOT / "models" / "demo" / "neuroepithelium" / "segmentation_checkpoints" / "checkpoints_set_8_all_but_target"
    ext = ".pt" if BACKEND == "torch" else ".h5"
    needed = [movie8, movie_t, reg_dir / f"best{ext}", seg_dir / f"best{ext}"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(f"  SKIP test_infer_stack_golden (missing: {missing})")
        return
    if BACKEND == "torch":
        import _tf_stub
        _tf_stub.install()
        import torch
        import torch_backend as tb
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dev_name = f"torch-{device}-" + (torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
        reg = tb.load_torch_regression(reg_dir / "best.pt", device)
        seg = tb.load_torch_segmentation(seg_dir / "best.pt", device)
    else:
        reg, seg = _api.build_models(str(reg_dir / "best.h5"), str(seg_dir / "best.h5"))
        dev_name = "keras-cpu"
    snap = {"backend": BACKEND, "device": dev_name, "sets": {}}
    for tag, movie, frames in (("set8", movie8, [27, 28, 29]), ("set_test", movie_t, [4, 5])):
        stack = _api.read_stack(movie)
        res = _api.infer_stack(stack, reg, seg, frames=frames)
        assert set(res) == set(frames)
        for i in frames:
            for d in res[i]:
                assert set(d) == {"x", "y", "angle", "length"}
                assert isinstance(d["x"], int) and isinstance(d["y"], int)
                assert isinstance(d["angle"], float) and isinstance(d["length"], float)
        snap["sets"][tag] = {str(i): sorted([[d["x"], d["y"], d["angle"], d["length"]] for d in res[i]]) for i in frames}
        print(f"  infer_stack[{BACKEND}] {tag}: " + ", ".join(f"f{i}:{len(res[i])}" for i in frames))
    g = _golden(LOCAL_GOLDEN / f"infer_stack_{BACKEND}_v1.json", snap)
    same_dev = g["device"] == snap["device"]
    if not same_dev:
        print(f"  [warn] golden device {g['device']!r} != current {snap['device']!r}: angle/length compared at 1e-3")
    for tag in snap["sets"]:
        for i in snap["sets"][tag]:
            a = np.asarray(g["sets"][tag][i], float).reshape(-1, 4)
            b = np.asarray(snap["sets"][tag][i], float).reshape(-1, 4)
            assert a.shape == b.shape, (tag, i, a.shape, b.shape)
            assert np.array_equal(a[:, :2], b[:, :2]), (tag, i, "centres changed")
            if same_dev:
                assert np.array_equal(a, b), (tag, i, "angle/length changed")
            else:
                assert np.max(np.abs(a[:, 2:] - b[:, 2:]), initial=0.0) <= 1e-3, (tag, i)


# --------------------------------------------------------------------------- plain-script runner
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {str(e)[:300]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed; failed: {failed}")
    sys.exit(1 if failed else 0)
