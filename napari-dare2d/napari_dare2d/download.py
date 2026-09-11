"""Command-line downloader for the DARE2D dataset + pretrained weights (Zenodo).

    python -m napari_dare2d.download                 # data + models -> project root
    python -m napari_dare2d.download --only models   # or --only data
    python -m napari_dare2d.download --root /path/to/DARE2d-v2
    python -m napari_dare2d.download --record 21644564   # pin an exact Zenodo version
    python -m napari_dare2d.download --check         # report present/missing, download nothing

Thin wrapper over ``napari_dare2d._data.download_dataset`` (the same code the napari
"Download DARE2D data" button runs), so CLI, plugin and notebooks lay files out identically:

    <root>/data/demo/neuroepithelium/set_{1..8}/
    <root>/models/demo/neuroepithelium/{regression,segmentation}_checkpoints/
                                       checkpoints_set_{1..8}_all_but_target/best.{h5,pt}

Archives are cached in ``<root>/_zenodo_cache/`` (md5-verified) next to a provenance file,
``zenodo_manifest.json``, recording the resolved Zenodo version. Existing files are never
overwritten. Exit status 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _default_root() -> Path:
    # The same anchor every other consumer uses (_api.PROJECT_ROOT), without importing _api
    # (which would drag in its lazy imports' module-level cost for nothing).
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    from . import _data

    p = argparse.ArgumentParser(
        prog="python -m napari_dare2d.download",
        description=f"Download the DARE2D dataset and pretrained weights from Zenodo "
                    f"({_data.ZENODO_URL}) into the project layout.")
    p.add_argument("--root", type=Path, default=_default_root(),
                   help="project root; data/ and models/ are created beneath it "
                        "(default: the DARE2d-v2 checkout this package lives in)")
    p.add_argument("--only", choices=sorted(_data._ONLY), action="append",
                   help="fetch only 'data' or only 'models' (repeatable; default: both)")
    p.add_argument("--record", default=_data.ZENODO_CONCEPT, metavar="ID",
                   help=f"Zenodo record id (default {_data.ZENODO_CONCEPT} = concept record, "
                        "i.e. the latest version; give a version id to pin it)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the md5 check of the archives against Zenodo's checksums")
    p.add_argument("--check", action="store_true",
                   help="only report which expected files are present/missing; no download")
    return p


def _report(root: Path, only) -> int:
    from . import _data

    missing = _data.missing_files(root, only)
    expected = _data.expected_paths(root, only)
    print(f"root: {root}")
    print(f"{len(expected) - len(missing)}/{len(expected)} expected paths present")
    for m in missing:
        print(f"  missing: {m}")
    manifest = root / _data.CACHE_DIRNAME / _data.MANIFEST_NAME
    if manifest.exists():
        print(f"provenance: {manifest}")
    return 1 if missing else 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from . import _data

    only = tuple(dict.fromkeys(args.only)) if args.only else ("data", "models")
    root = args.root.resolve()
    if args.check:
        return _report(root, only)

    state = {"t0": time.time(), "last": 0.0}

    def progress(done, total):
        now = time.time()
        if total and (now - state["last"] > 0.5 or done >= total):
            state["last"] = now
            mb_s = done / 1e6 / max(now - state["t0"], 1e-6)
            sys.stdout.write(f"\r  {100 * done / total:5.1f}%  {done / 1e6:8.0f}/{total / 1e6:.0f} MB"
                             f"  {mb_s:5.1f} MB/s   ")
            sys.stdout.flush()

    def log(msg):
        sys.stdout.write("\n" if state["last"] else "")
        state["last"] = 0.0
        print(f"[dare2d-download] {msg}", flush=True)

    print(f"[dare2d-download] root = {root}   only = {', '.join(only)}", flush=True)
    try:
        _data.download_dataset(root, progress_cb=progress, log=log, record_id=args.record,
                               only=only, verify=not args.no_verify)
    except KeyboardInterrupt:
        print("\n[dare2d-download] interrupted; partial *.part files can be deleted from "
              f"{root / _data.CACHE_DIRNAME}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - one clear line for users, the class for debugging
        print(f"\n[dare2d-download] FAILED: {e}  ({type(e).__name__})", file=sys.stderr)
        return 1
    return _report(root, only)


if __name__ == "__main__":
    raise SystemExit(main())
