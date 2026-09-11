"""Offline tests for the Zenodo downloader's layout logic (napari_dare2d/_data.py).

No network, no real data: synthetic ``data.zip`` / ``models.zip`` with the real top-level
prefixes are built in a temp dir and pushed through ``extract_archive`` / ``download_dataset``
(with the Zenodo API and the HTTP download monkey-patched). Pins:

  - both archives extract at ``root`` into the exact ``data/demo/...`` / ``models/demo/...`` tree;
  - existing files are NEVER overwritten (re-running only fills in what is missing);
  - junk (``__MACOSX``, ``.DS_Store``) and members outside the archive's folder / with ``..``
    or absolute paths are refused;
  - md5 verification: a corrupt cached zip is re-downloaded, a corrupt fresh download raises and
    is deleted; ``--no-verify`` skips both;
  - ``only=`` selects archives; unknown archive keys are skipped; a record without our archives
    raises a clear error;
  - the provenance manifest records the resolved record id / DOI / version / md5s;
  - ``missing_files`` is non-empty before and empty after; the CLI ``--check`` exit codes follow it.

Run:  python tests/test_download_layout.py
  or: python -m pytest tests/test_download_layout.py -q
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "napari-dare2d"))

from napari_dare2d import _data, download  # noqa: E402

SETS = range(1, 9)


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _zip_bytes(members: dict) -> bytes:
    """``{name: bytes}`` -> zip file bytes (a name ending in "/" is a directory entry)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, payload in members.items():
            if name.endswith("/"):
                z.writestr(zipfile.ZipInfo(name), b"")
            else:
                z.writestr(name, payload)
    return buf.getvalue()


def _data_members(extra=None):
    m = {"data/": b"", "data/demo/": b"", "data/demo/neuroepithelium/": b""}
    for n in SETS:
        m[f"data/demo/neuroepithelium/set_{n}/"] = b""
        m[f"data/demo/neuroepithelium/set_{n}/movie_{n}.tiff"] = b"TIFF" + bytes([n])
        m[f"data/demo/neuroepithelium/set_{n}/division_position1.npy"] = b"NPY" + bytes([n])
    m["data/demo/neuroepithelium/set_test/movie_M-z5.tif"] = b"test movie"
    m.update(extra or {})
    return m


def _models_members(extra=None):
    m = {"models/": b"", "models/demo/": b"", "models/demo/neuroepithelium/": b""}
    for stage in ("regression", "segmentation"):
        for n in SETS:
            d = f"models/demo/neuroepithelium/{stage}_checkpoints/checkpoints_set_{n}_all_but_target/"
            m[d] = b""
            m[d + "best.h5"] = f"h5-{stage}-{n}".encode()
            m[d + "best.pt"] = f"pt-{stage}-{n}".encode()
    m.update(extra or {})
    return m


class FakeZenodo:
    """Monkey-patches ``_data.zenodo_record`` and ``_data._download`` for one test."""

    def __init__(self, archives: dict, record_id="21644564", corrupt_md5_for=()):
        self.archives = archives  # key -> bytes
        self.record_id = record_id
        self.corrupt = set(corrupt_md5_for)
        self.downloads = []

    def record(self, record_id=None):
        self.queried = record_id
        files = []
        for key, blob in self.archives.items():
            md5 = hashlib.md5(blob).hexdigest()
            if key in self.corrupt:
                md5 = "0" * 32
            files.append({"key": key, "size": len(blob), "checksum": f"md5:{md5}",
                          "links": {"self": f"fake://{key}"}})
        return {"id": int(self.record_id), "doi": f"10.5281/zenodo.{self.record_id}",
                "conceptdoi": _data.ZENODO_DOI,
                "metadata": {"version": "v2", "publication_date": "2026-07-28", "title": "DARE2d"},
                "files": files}

    def download(self, url, out, total, progress_cb=None):
        key = url.split("://", 1)[1]
        self.downloads.append(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        blob = self.archives[key]
        out.write_bytes(blob)
        if progress_cb is not None:
            progress_cb(len(blob), total)

    def __enter__(self):
        self._saved = (_data.zenodo_record, _data._download)
        _data.zenodo_record = self.record
        _data._download = self.download
        return self

    def __exit__(self, *exc):
        _data.zenodo_record, _data._download = self._saved


def _tmp():
    return Path(tempfile.mkdtemp(prefix="dare2d_dl_"))


def _tree(root: Path):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# tests                                                                        #
# --------------------------------------------------------------------------- #
def test_archives_map_to_project_layout():
    assert _data.ARCHIVES == {"data.zip": "data/", "models.zip": "models/"}
    assert _data.ZENODO_CONCEPT == "17442226"
    assert _data.ZENODO_DOI == "10.5281/zenodo.17442226"
    assert _data.ZENODO_URL == "https://doi.org/10.5281/zenodo.17442226"


def test_extract_places_tree_and_never_overwrites():
    root = _tmp()
    try:
        z = root / "models.zip"
        z.write_bytes(_zip_bytes(_models_members()))
        pre = root / "models/demo/neuroepithelium/regression_checkpoints/checkpoints_set_3_all_but_target/best.pt"
        pre.parent.mkdir(parents=True)
        pre.write_bytes(b"MY RETRAINED WEIGHTS")
        added, skipped, refused = _data.extract_archive(z, root, "models/", log=lambda m: None)
        assert (added, skipped, refused) == (31, 1, 0), (added, skipped, refused)
        assert pre.read_bytes() == b"MY RETRAINED WEIGHTS"  # untouched
        assert (root / "models/demo/neuroepithelium/segmentation_checkpoints/checkpoints_set_8_all_but_target/best.h5").read_bytes() == b"h5-segmentation-8"
        assert _data.missing_files(root, ("models",)) == []
        # second run: everything present -> nothing added
        assert _data.extract_archive(z, root, "models/", log=lambda m: None) == (0, 32, 0)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_extract_refuses_junk_and_escapes():
    root = _tmp()
    try:
        bad = {
            "__MACOSX/data/._x": b"junk",
            "data/demo/.DS_Store": b"junk",
            "models/demo/neuroepithelium/stray.txt": b"outside data/ prefix",
            "data/../escape.txt": b"zip slip",
            "/abs/data/x.txt": b"absolute",
            "README.txt": b"top-level file",
        }
        z = root / "data.zip"
        z.write_bytes(_zip_bytes(_data_members(bad)))
        refused_log = []
        added, skipped, refused = _data.extract_archive(z, root, "data/", log=refused_log.append)
        assert refused == 4, refused          # junk is silently dropped, the 4 others refused
        assert added == 17, added             # 8 movies + 8 npy + set_test movie
        files = _tree(root)
        assert not any("escape" in f or "stray" in f or "abs" in f or "README" in f
                       or "DS_Store" in f or "MACOSX" in f for f in files), files
        assert all(f.startswith("data/demo/neuroepithelium/") for f in files if f != "data.zip"), files
        # the path guard itself (zipfile normalises backslashes on write, so test it directly)
        ok = _data._member_ok
        assert ok("data/demo/x.tif", "data/") and ok("data/", "data/")
        for bad_name in ("data\\demo\\win.txt", "data/../x", "../data/x", "/data/x", "C:data/x",
                         "models/x", "", "data//x", "data/./x"):
            assert not ok(bad_name, "data/"), bad_name
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_download_dataset_end_to_end_with_manifest():
    root = _tmp()
    try:
        fake = FakeZenodo({"data.zip": _zip_bytes(_data_members()),
                           "models.zip": _zip_bytes(_models_members()),
                           "README.txt": b"ignored"})
        assert len(_data.missing_files(root)) == 8 + 32
        progress, logs = [], []
        with fake:
            out = _data.download_dataset(root, progress_cb=lambda d, t: progress.append((d, t)),
                                         log=logs.append)
        assert out == root
        assert fake.queried == _data.ZENODO_CONCEPT          # asked the concept record by default
        assert fake.downloads == ["data.zip", "models.zip"]  # README.txt skipped
        assert _data.missing_files(root) == []
        # cumulative progress: monotone, one shared total, ends at total
        totals = {t for _, t in progress}
        assert len(totals) == 1 and progress[-1][0] == progress[-1][1], progress
        assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:])), progress
        # cache + manifest
        cache = root / _data.CACHE_DIRNAME
        assert (cache / "data.zip").exists() and (cache / "models.zip").exists()
        man = json.loads((cache / _data.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert man["record_id"] == "21644564" and man["record_doi"] == "10.5281/zenodo.21644564"
        assert man["concept_doi"] == _data.ZENODO_DOI and man["version"] == "v2"
        assert set(man["files"]) == {"data.zip", "models.zip"}
        assert man["files"]["models.zip"]["verified"] is True and man["files"]["models.zip"]["added"] == 32
        assert man["files"]["models.zip"]["md5"] == hashlib.md5(fake.archives["models.zip"]).hexdigest()
        assert any("record 21644564" in m for m in logs), logs
        # re-run: cached archives verified + reused, nothing re-downloaded, nothing added
        with fake:
            _data.download_dataset(root, log=logs.append)
        assert fake.downloads == ["data.zip", "models.zip"]
        assert any("already downloaded: models.zip" in m for m in logs)
        man2 = json.loads((cache / _data.MANIFEST_NAME).read_text(encoding="utf-8"))
        assert man2["files"]["models.zip"]["added"] == 0 and man2["files"]["models.zip"]["skipped_existing"] == 32
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_only_and_pinned_record():
    root = _tmp()
    try:
        fake = FakeZenodo({"data.zip": _zip_bytes(_data_members()),
                           "models.zip": _zip_bytes(_models_members())}, record_id="17442227")
        with fake:
            _data.download_dataset(root, log=lambda m: None, only=("models",), record_id="17442227")
        assert fake.queried == "17442227" and fake.downloads == ["models.zip"]
        assert _data.missing_files(root, ("models",)) == []
        assert len(_data.missing_files(root, ("data",))) == 8
        assert not (root / "data").exists()
        try:
            _data.download_dataset(root, only=("weights",))
        except ValueError as e:
            assert "only=" in str(e)
        else:
            raise AssertionError("bad only= must raise")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_md5_mismatch_handling():
    root = _tmp()
    try:
        good = _zip_bytes(_models_members())
        # 1) a corrupt CACHED zip (right size, wrong bytes) is detected and re-downloaded
        cache = root / _data.CACHE_DIRNAME
        cache.mkdir()
        (cache / "models.zip").write_bytes(b"x" * len(good))
        fake = FakeZenodo({"models.zip": good})
        logs = []
        with fake:
            _data.download_dataset(root, log=logs.append, only=("models",))
        assert fake.downloads == ["models.zip"]
        assert any("corrupt" in m for m in logs), logs
        assert _data.md5sum(cache / "models.zip") == hashlib.md5(good).hexdigest()
        # 2) a fresh download whose md5 does not match Zenodo's raises and is deleted
        shutil.rmtree(root / "models"); (cache / "models.zip").unlink()
        fake = FakeZenodo({"models.zip": good}, corrupt_md5_for=("models.zip",))
        with fake:
            try:
                _data.download_dataset(root, log=lambda m: None, only=("models",))
            except RuntimeError as e:
                assert "md5 mismatch for models.zip" in str(e)
            else:
                raise AssertionError("md5 mismatch must raise")
        assert not (cache / "models.zip").exists()
        assert not (root / "models").exists()
        # 3) verify=False accepts it
        with fake:
            _data.download_dataset(root, log=lambda m: None, only=("models",), verify=False)
        assert _data.missing_files(root, ("models",)) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_record_without_our_archives_raises():
    root = _tmp()
    try:
        fake = FakeZenodo({"neuroepithelium.zip": b"old layout", "README.txt": b""})
        with fake:
            try:
                _data.download_dataset(root, log=lambda m: None)
            except RuntimeError as e:
                assert "data.zip" in str(e) and "neuroepithelium.zip" in str(e)
            else:
                raise AssertionError("must raise when the archive layout changed")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cli_check_and_run(capsys=None):
    root = _tmp()
    try:
        assert download.main(["--check", "--root", str(root)]) == 1   # nothing present yet
        fake = FakeZenodo({"data.zip": _zip_bytes(_data_members()),
                           "models.zip": _zip_bytes(_models_members())})
        with fake:
            assert download.main(["--root", str(root), "--only", "data", "--only", "models"]) == 0
        assert download.main(["--check", "--root", str(root)]) == 0
        assert _data.missing_files(root) == []
        # failure path -> exit 1, no traceback
        fake = FakeZenodo({"README.txt": b""})
        with fake:
            assert download.main(["--root", str(root)]) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_default_root_is_repo_root():
    assert download._default_root() == _ROOT
    assert (download._default_root() / "setup.py").exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
