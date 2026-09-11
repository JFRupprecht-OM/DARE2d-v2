"""Fix-surface guard: only consensus-specific code may differ from HEAD; the individual-model path,
checkpoints and data must be untouched.

  --snapshot  (run BEFORE the change) records the pre-change `git status --porcelain`, HEAD sha and an
              md5 map of models/**/best.* and data/demo/**/*.{npy,tif,tiff} into
              docs/audit/results/golden/fix_surface_baseline.json (git-ignored).
  plain run   (after the change) asserts:
    - every newly changed/untracked path is under the consensus whitelist and none under a forbidden tree;
    - napari_dare2d/_api.py: every top-level def except `consensus` is AST-identical to HEAD, and the
      module-level imports/statements are unchanged (import-light invariant);
    - scripts/postprocessing/main.py: only the three aggregation helpers may differ;
    - napari_dare2d/_widget.py: only the widget function containing the ensemble branch may differ;
    - files of the individual path are byte-identical to HEAD; models/ and data/ md5s unchanged.

Run:  python tests/test_fix_surface.py --snapshot     (once, pre-change)
      python tests/test_fix_surface.py                (after the change)
   or: python -m pytest tests/test_fix_surface.py -q
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "docs" / "audit" / "results" / "golden" / "fix_surface_baseline.json"
SNAPSHOT = "--snapshot" in sys.argv

WHITELIST = ("napari-dare2d/napari_dare2d/_api.py", "scripts/postprocessing/main.py",
             "napari-dare2d/napari_dare2d/_widget.py", "tests/", "napari-dare2d/verify_",
             "docs/", "README.md", "napari-dare2d/README.md", "CLAUDE.md")
FORBIDDEN = ("dare2d/", "scripts/inference/", "dare2d-torch/", "training/", "config/", "models/",
             "data/", "annotator/", "napari-dare2d/napari_dare2d/_data.py",
             "napari-dare2d/napari_dare2d/__init__.py", "napari-dare2d/napari_dare2d/napari.yaml")
INDIVIDUAL_PATH_FILES = ("napari-dare2d/napari_dare2d/_data.py", "scripts/inference/multistage_detection2d.py",
                         "dare2d/datamodule/post_processing/regression2d_pp.py",
                         "dare2d/datamodule/visualization/regression2d_visualisation.py",
                         "dare2d-torch/torch_backend.py", "dare2d-torch/models_torch.py",
                         "dare2d/evaluation/center_metrics.py")


def _git(*args, check=True):
    r = subprocess.run(["git", "-C", str(_ROOT), *args], capture_output=True, check=check)
    return r.stdout.decode("utf-8", errors="replace"), r.returncode


def _status_paths():
    out, _ = _git("status", "--porcelain")
    paths = set()
    for line in out.splitlines():
        if not line.strip():
            continue
        p = line[3:].split(" -> ")[-1].strip()
        if p.startswith('"') and p.endswith('"'):
            p = p[1:-1]
        paths.add(p)
    return sorted(paths)


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _tree_md5():
    files = list((_ROOT / "models").rglob("best.*"))
    for pat in ("*.npy", "*.tif", "*.tiff"):
        files += list((_ROOT / "data" / "demo").rglob(pat))
    return {p.relative_to(_ROOT).as_posix(): _md5(p) for p in sorted(files)}


def _baseline():
    if not BASELINE.exists():
        raise AssertionError(f"baseline missing: {BASELINE} - run `python tests/test_fix_surface.py --snapshot` BEFORE the change")
    with open(BASELINE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _top_level(src):
    tree = ast.parse(src)
    defs, imports, others = {}, [], []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[n.name] = ast.dump(n)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            imports.append(ast.dump(n))
        elif isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) \
                and isinstance(n.value.value, str):
            continue  # module docstring: free to change
        else:
            others.append(ast.dump(n))
    return defs, imports, others


def _compare_file(rel, allowed):
    head, _ = _git("show", f"HEAD:{rel}")
    now = (_ROOT / rel).read_text(encoding="utf-8")
    d0, i0, o0 = _top_level(head)
    d1, i1, o1 = _top_level(now)
    assert i0 == i1, f"{rel}: module-level imports changed"
    assert o0 == o1, f"{rel}: module-level statements changed"
    assert set(d0) == set(d1), f"{rel}: top-level definitions added/removed: {sorted(set(d0) ^ set(d1))}"
    changed = {n for n in d0 if d0[n] != d1[n]}
    assert changed <= set(allowed), f"{rel}: definitions changed outside the allowed set {sorted(allowed)}: {sorted(changed - set(allowed))}"
    return changed


def _widget_allowed():
    """The top-level function(s) of _widget.py whose HEAD source contains the ensemble/consensus call."""
    rel = "napari-dare2d/napari_dare2d/_widget.py"
    head, _ = _git("show", f"HEAD:{rel}")
    tree = ast.parse(head)
    names = set()
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            seg = ast.get_source_segment(head, n) or ""
            if "_api.consensus(" in seg:
                names.add(n.name)
    assert names, "could not locate the widget function containing _api.consensus("
    return names


# --------------------------------------------------------------------------- tests
def test_only_whitelisted_files_changed():
    base = set(_baseline()["git_status"])
    new = [p for p in _status_paths() if p not in base]
    bad = [p for p in new if p.startswith(FORBIDDEN) or not p.startswith(WHITELIST)]
    assert not bad, f"changed/untracked paths outside the consensus whitelist: {bad}"


def test_api_functions_unchanged_except_consensus():
    _compare_file("napari-dare2d/napari_dare2d/_api.py", {"consensus"})


def test_main_functions_unchanged_except_aggregation():
    _compare_file("scripts/postprocessing/main.py",
                  {"detect_angle_units_and_convert", "pick_consensus_angle_signed", "aggregate_cluster_pick_signed"})


def test_widget_unchanged_except_ensemble_branch():
    _compare_file("napari-dare2d/napari_dare2d/_widget.py", _widget_allowed())


def test_individual_path_files_identical_to_head():
    dirty = []
    for rel in INDIVIDUAL_PATH_FILES:
        _, rc = _git("diff", "--quiet", "HEAD", "--", rel, check=False)
        if rc != 0:
            dirty.append(rel)
    assert not dirty, f"individual-path files differ from HEAD: {dirty}"


def test_models_and_data_unchanged():
    base = _baseline()["tree_md5"]
    now = _tree_md5()
    added = sorted(set(now) - set(base))
    removed = sorted(set(base) - set(now))
    changed = sorted(k for k in set(base) & set(now) if base[k] != now[k])
    assert not (added or removed or changed), f"models/data changed: added={added} removed={removed} changed={changed}"


# --------------------------------------------------------------------------- runner
if __name__ == "__main__":
    if SNAPSHOT:
        head_sha, _ = _git("rev-parse", "HEAD")
        snap = {"head": head_sha.strip(), "git_status": _status_paths(), "tree_md5": _tree_md5()}
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as fh:
            json.dump(snap, fh, indent=1)
        print(f"baseline written: {BASELINE}\n  HEAD {snap['head']}\n  git status: {snap['git_status']}\n  hashed files: {len(snap['tree_md5'])}")
        sys.exit(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {str(e)[:400]}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed; failed: {failed}")
    sys.exit(1 if failed else 0)
