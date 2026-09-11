"""Download DARE2D data/models from Zenodo and save plugin results.

Zenodo concept DOI 10.5281/zenodo.17442226 (https://doi.org/10.5281/zenodo.17442226): the
concept record always resolves to the *latest* version, so ``download_dataset`` follows new
releases automatically and records which version it actually fetched (id, DOI, md5s) in
``<root>/_zenodo_cache/zenodo_manifest.json``. Pass ``record_id`` to pin an exact version.

The record ships two archives whose top-level folders ARE the project layout, so both are
extracted at the project root:

    data.zip    -> data/demo/neuroepithelium/set_{1..8}/  (+ set_test/, a test movie)
    models.zip  -> models/demo/neuroepithelium/{regression,segmentation}_checkpoints/
                       checkpoints_set_{1..8}_all_but_target/best.h5 + best.pt

Stdlib-only (``urllib`` + ``zipfile`` + ``hashlib``), so no install-time dependency is added.
Used by the napari "Download DARE2D data" button, ``python -m napari_dare2d.download`` and
the ``notebooks/Run_dare2d_*.ipynb`` setup cells; also hosts the "Save DARE2D results" writer.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

#: Zenodo *concept* record: resolves to the latest published version of the DARE2D archive.
ZENODO_CONCEPT = "17442226"
ZENODO_DOI = f"10.5281/zenodo.{ZENODO_CONCEPT}"
ZENODO_URL = f"https://doi.org/{ZENODO_DOI}"
ZENODO_RECORD = ZENODO_CONCEPT  # backward-compatible alias
_API = "https://zenodo.org/api/records/{}"

#: Zenodo archive -> the single top-level folder it must contain. Every archive is extracted
#: at the project root, so these prefixes double as a zip-slip guard: any member outside its
#: archive's folder is refused.
ARCHIVES = {
    "data.zip": "data/",
    "models.zip": "models/",
}
#: short names accepted by ``only=`` / the CLI ``--only`` flag
_ONLY = {"data": "data.zip", "models": "models.zip"}

CACHE_DIRNAME = "_zenodo_cache"
MANIFEST_NAME = "zenodo_manifest.json"

_DATASET = Path("data") / "demo" / "neuroepithelium"
_MODELS = Path("models") / "demo" / "neuroepithelium"
_SETS = range(1, 9)


# --------------------------------------------------------------------------- #
# Zenodo API                                                                   #
# --------------------------------------------------------------------------- #
def zenodo_record(record_id: str = ZENODO_CONCEPT) -> dict:
    """Return the record's JSON (``id``, ``doi``, ``metadata``, ``files`` ...).

    Querying the concept id returns the latest version's record, so ``record["id"]`` is the
    concrete version that will be downloaded.
    """
    url = _API.format(record_id)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Zenodo returned HTTP {e.code} for {url} - check the record id") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach Zenodo ({e.reason}) - check your network/proxy") from e


def zenodo_files(record_id: str = ZENODO_CONCEPT):
    """Return the record's file list (each has ``key``, ``size``, ``checksum``, ``links.self``)."""
    return zenodo_record(record_id)["files"]


def _record_summary(rec: dict) -> dict:
    md = rec.get("metadata", {})
    version = md.get("version") or ""
    if not version:
        # Zenodo omits metadata.version unless the depositor typed one; the 0-based position in
        # the concept's version chain is always there (index 1 == the "v2" shown on the web page).
        try:
            version = f"v{int(md['relations']['version'][0]['index']) + 1}"
        except (KeyError, IndexError, TypeError, ValueError):
            version = ""
    return {
        "concept_doi": rec.get("conceptdoi") or ZENODO_DOI,
        "record_id": str(rec.get("id", "")),
        "record_doi": rec.get("doi", ""),
        "version": version,
        "publication_date": md.get("publication_date", ""),
        "title": md.get("title", ""),
    }


# --------------------------------------------------------------------------- #
# Download / verify / extract primitives                                       #
# --------------------------------------------------------------------------- #
def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.startswith("__MACOSX") or base == ".DS_Store" or base.startswith("._")


def _download(url: str, out: Path, total, progress_cb=None) -> None:
    """Stream ``url`` to ``out`` (atomic via a .part temp), reporting bytes done."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        while True:
            buf = resp.read(1 << 20)  # 1 MiB
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if progress_cb is not None:
                progress_cb(done, total or 0)
    tmp.replace(out)


def md5sum(path, chunk: int = 1 << 20) -> str:
    """Hex md5 of a file, streamed (the archives are up to ~2.2 GB)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _expected_md5(f: dict):
    """``"md5:abc..."`` from the Zenodo file entry -> ``"abc..."`` (None if absent/other algo)."""
    cs = f.get("checksum") or ""
    return cs.split(":", 1)[1].lower() if cs.lower().startswith("md5:") else None


def _member_ok(name: str, prefix: str) -> bool:
    """Accept only members inside the archive's top-level folder with a clean relative path."""
    if not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name:
        return False
    if not name.startswith(prefix):
        return False
    return all(part not in ("", ".", "..") for part in name.rstrip("/").split("/"))


def extract_archive(zpath, root, prefix: str, log=print, progress_cb=None):
    """Extract ``zpath`` at ``root`` keeping only members under ``prefix``.

    NEVER overwrites: an existing file is skipped, so a populated ``models/`` (e.g. retrained
    checkpoints) is left intact and re-running only fills in what is missing. Returns
    ``(added, skipped, refused)`` counts; ``refused`` members (outside ``prefix`` or with an
    unsafe path) are reported and dropped.
    """
    root = Path(root)
    added = skipped = refused = 0
    with zipfile.ZipFile(zpath) as z:
        members = z.infolist()
        n = len(members)
        for i, info in enumerate(members, 1):
            m = info.filename
            if _is_junk(m):
                continue
            if not _member_ok(m, prefix):
                refused += 1
                log(f"  refused (outside {prefix}): {m}")
                continue
            dest = root / m
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            elif dest.exists():
                skipped += 1
            else:
                z.extract(info, root)
                added += 1
            if progress_cb is not None:
                progress_cb(i, n)
    return added, skipped, refused


# --------------------------------------------------------------------------- #
# What the project expects on disk                                             #
# --------------------------------------------------------------------------- #
def expected_paths(root, only=("data", "models")):
    """The files/folders the plugin, notebooks and tests read (relative to ``root``)."""
    root = Path(root)
    out = []
    if "data" in only:
        out += [root / _DATASET / f"set_{n}" for n in _SETS]
    if "models" in only:
        for stage in ("regression", "segmentation"):
            for n in _SETS:
                d = root / _MODELS / f"{stage}_checkpoints" / f"checkpoints_set_{n}_all_but_target"
                out += [d / "best.h5", d / "best.pt"]
    return out


def missing_files(root, only=("data", "models")):
    """Relative POSIX paths from ``expected_paths`` that are absent under ``root``."""
    root = Path(root)
    return [p.relative_to(root).as_posix() for p in expected_paths(root, only) if not p.exists()]


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def download_dataset(root, progress_cb=None, log=print, *, record_id: str = ZENODO_CONCEPT,
                     only=("data", "models"), verify: bool = True) -> Path:
    """Download the DARE2D Zenodo archives and lay them out under ``root``.

    Args:
        root: project root (``data/`` and ``models/`` are created beneath it).
        progress_cb: ``cb(done_bytes, total_bytes)`` cumulative over ALL archives to download.
        log: one-line status messages (``print`` for the CLI, a dict-setter for the widget).
        record_id: Zenodo record to fetch; the default concept id resolves to the latest version.
        only: subset of ``("data", "models")``.
        verify: md5-check every archive (cached or fresh) against the Zenodo checksum.

    Archives are cached under ``root/_zenodo_cache`` and reused when their size (and, with
    ``verify``, md5) matches. Extraction never overwrites existing files. A provenance record
    is written to ``root/_zenodo_cache/zenodo_manifest.json``. Raises ``RuntimeError`` with a
    one-line explanation on any failure.
    """
    root = Path(root)
    only = tuple(only)
    bad = [o for o in only if o not in _ONLY]
    if bad:
        raise ValueError(f"only= accepts {sorted(_ONLY)}, got {bad}")
    wanted = {_ONLY[o] for o in only}

    rec = zenodo_record(record_id)
    info = _record_summary(rec)
    log(f"Zenodo {info['concept_doi']} -> record {info['record_id']}"
        f" (version {info['version'] or '?'}, {info['publication_date']})")
    files = [f for f in rec["files"] if f["key"] in wanted]
    if not files:
        have = ", ".join(sorted(f["key"] for f in rec["files"])) or "nothing"
        raise RuntimeError(f"record {info['record_id']} has none of {sorted(wanted)} (it has: {have})"
                           " - the archive layout changed; update ARCHIVES in napari_dare2d/_data.py")

    cache = root / CACHE_DIRNAME
    cache.mkdir(parents=True, exist_ok=True)
    total = sum(int(f.get("size") or 0) for f in files)
    base = 0  # bytes of archives already completed (for the cumulative progress)

    def _cb(done, _):
        if progress_cb is not None:
            progress_cb(base + done, total)

    manifest_files = {}
    for f in files:
        key, size, want_md5 = f["key"], int(f.get("size") or 0), _expected_md5(f)
        prefix = ARCHIVES[key]
        zpath = cache / key

        reuse = zpath.exists() and (size == 0 or zpath.stat().st_size == size)
        if reuse and verify and want_md5:
            log(f"verifying cached {key}...")
            if md5sum(zpath) != want_md5:
                log(f"  cached {key} is corrupt (md5 mismatch) - re-downloading")
                zpath.unlink()
                reuse = False
        if reuse:
            log(f"already downloaded: {key}")
            _cb(size, total)
        else:
            log(f"downloading {key} ({size / 1e6:.0f} MB)...")
            _download(f["links"]["self"], zpath, size, _cb)
            if verify and want_md5:
                log(f"verifying {key}...")
                got = md5sum(zpath)
                if got != want_md5:
                    zpath.unlink(missing_ok=True)
                    raise RuntimeError(f"md5 mismatch for {key} (expected {want_md5}, got {got});"
                                       " the download was discarded - please retry")
        base += size

        log(f"extracting {key} -> {prefix}...")
        added, skipped, refused = extract_archive(zpath, root, prefix, log)
        log(f"  {key}: {added} files added, {skipped} already present"
            + (f", {refused} refused" if refused else ""))
        manifest_files[key] = {"size": size, "md5": want_md5, "verified": bool(verify and want_md5),
                               "added": added, "skipped_existing": skipped}

    manifest = {**info, "downloaded_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "root": str(root), "files": manifest_files}
    (cache / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    still = missing_files(root, only)
    if still:
        log(f"WARNING: {len(still)} expected paths are still missing, e.g. {still[0]}")
    log(f"done -> {root}")
    return root


# --------------------------------------------------------------------------- #
# Saving plugin results                                                        #
# --------------------------------------------------------------------------- #
def save_results(image, points, features, out_dir, image_name="dare2d", draw=True):
    """Write DARE2D detections to disk.

    Args:
        image: the analysed ``(T, Y, X)`` stack — used to render the overlay movie.
        points: ``(N, 3)`` array of detection coords ``[t, y, x]`` (napari Points data).
        features: dict of per-detection arrays, e.g. ``{"angle": (N,), "length": (N,)}``.
        out_dir: folder to write into (created if missing).
        image_name: stem used for the output filenames.
        draw: also render a ``(T, Y, X, 3)`` overlay tiff ("result movie").

    Writes per-frame ``division_position{t}.npy`` ([x, y] pairs), a ``*_summary.csv``
    (frame, x, y, angle, length, …) and, if ``draw``, ``*_result.tiff``. Returns out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("no detections to save (the Points layer is empty)")
    t = points[:, 0].round().astype(int)
    y, x = points[:, 1], points[:, 2]
    n = len(points)
    angle = np.asarray(features.get("angle", np.full(n, np.nan)), float).reshape(-1)
    length = np.asarray(features.get("length", np.full(n, np.nan)), float).reshape(-1)
    extra = [k for k in features if k not in ("angle", "length")]

    # per-frame [x, y] npy (the project's division_position layout)
    for f in np.unique(t):
        sel = t == f
        np.save(out_dir / f"division_position{int(f)}.npy",
                np.stack([x[sel], y[sel]], axis=1))

    # summary csv
    cols = ["frame", "x", "y", "angle", "length"] + extra
    with open(out_dir / f"{image_name}_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(n):
            row = [int(t[i]), float(x[i]), float(y[i]), float(angle[i]), float(length[i])]
            row += [float(np.asarray(features[k], float).reshape(-1)[i]) for k in extra]
            w.writerow(row)

    # overlay "result movie": grayscale -> RGB with a red centre + cyan division axis
    if draw:
        import cv2
        import tifffile

        g = np.asarray(image)
        if g.ndim == 2:
            g = g[None]
        if g.dtype != np.uint8:
            gmin, gptp = float(g.min()), float(np.ptp(g)) or 1.0
            g = (255 * (g.astype(float) - gmin) / gptp).astype(np.uint8)
        movie = np.repeat(g[..., None], 3, axis=-1)  # (T, Y, X, 3)
        T = movie.shape[0]
        for i in range(n):
            ti = int(t[i])
            if not (0 <= ti < T):
                continue
            cx, cy = int(round(x[i])), int(round(y[i]))
            cv2.circle(movie[ti], (cx, cy), 4, (255, 0, 0), -1)  # red centre
            if np.isfinite(angle[i]) and np.isfinite(length[i]):
                th = np.radians(angle[i])
                # match detections_to_vectors: direction (Δy=cos·L, Δx=sin·L)
                dcol, drow = np.sin(th) * length[i] / 2.0, np.cos(th) * length[i] / 2.0
                cv2.line(movie[ti], (int(cx - dcol), int(cy - drow)),
                         (int(cx + dcol), int(cy + drow)), (0, 255, 255), 2)  # cyan axis
        tifffile.imwrite(out_dir / f"{image_name}_result.tiff", movie)
    return out_dir
