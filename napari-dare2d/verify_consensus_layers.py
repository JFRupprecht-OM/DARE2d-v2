"""Runnable check: the CONSENSUS layer draws the same axis as the single-set layer (napari, headless).

Feeds 8 identical synthetic detections through the real ``_api.consensus`` and then through the same
``to_layer_data`` path the widget uses (``frame_base=1``), adds the layers to a headless napari viewer
and reads them back:
  - the consensus Vectors direction must be parallel to the individual (single-set) direction
    for every theta (nematic angle ~ 0), including the wrap cases +-89.9;
  - the consensus Points ``angle`` feature (what ``_data.save_results`` writes to CSV) must equal theta
    (nematically);
  - both Points layers sit at the same (t, y, x).
Before the consensus-convention fix this FAILS with a |90 - 2 theta| error (mirror about the diagonal).

Run:  <env>/python.exe napari-dare2d/verify_consensus_layers.py   (exit 0 = OK)
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from napari_dare2d import _api as api  # noqa: E402

failures = []


def check(cond, msg):
    print(("  [OK] " if cond else "  [FAIL] ") + msg)
    if not cond:
        failures.append(msg)


def nematic(a, b):
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def vec_nematic(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(v1, v2))) / n)))


def main():
    import napari

    v = napari.Viewer(show=False)
    v.add_image(np.zeros((2, 64, 64), np.uint8), name="blank")
    worst = 0.0
    names = ("ind centers", "ind axes", "cons centers", "cons axes")
    for theta in (-89.9, -60.0, -45.0, -30.0, 0.0, 30.0, 45.0, 60.0, 89.9):
        ind = {0: [{"x": 32, "y": 32, "angle": theta, "length": 20.0}]}
        with contextlib.redirect_stdout(io.StringIO()):
            cons = api.consensus({1: [(m, 32.0, 32.0, theta, 20.0) for m in range(1, 9)]}, 1,
                                 eps=10, min_models=6, num_models=8)   # widget defaults incl. angle_mode='auto'
        for data, meta, ltype in api.to_layer_data(ind, frame_base=0, name="ind") + \
                api.to_layer_data(cons, frame_base=1, name="cons"):
            v._add_layer_from_data(data, meta, ltype)
        by_name = {ly.name: ly for ly in v.layers}
        d_ind = by_name["ind axes"].data[0, 1, 1:]
        d_con = by_name["cons axes"].data[0, 1, 1:]
        err = vec_nematic(d_ind, d_con)
        worst = max(worst, err)
        check(err <= 1e-6, f"theta={theta:+6.1f}: consensus axis parallel to single-set axis (nematic err {err:.3f} deg)")
        p_ind = by_name["ind centers"].data[0]
        p_con = by_name["cons centers"].data[0]
        check(np.allclose(p_ind, p_con) and np.allclose(p_ind, [0, 32, 32]),
              f"theta={theta:+6.1f}: both centres at (t,y,x)=(0,32,32): ind {p_ind.tolist()} cons {p_con.tolist()}")
        pts = by_name["cons centers"]
        try:
            feat = float(np.asarray(pts.features["angle"])[0])
        except Exception:
            feat = float(np.asarray(pts.properties["angle"])[0])
        check(nematic(feat, theta) <= 1e-6,
              f"theta={theta:+6.1f}: consensus Points 'angle' feature == theta (got {feat:.3f})")
        for nm in names:
            if nm in v.layers:
                v.layers.remove(v.layers[nm])
    v.close()
    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed; worst consensus-vs-single axis error {worst:.2f} deg)")
        return 1
    print("RESULT: OK - consensus layers draw the same axis convention as single-set layers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
