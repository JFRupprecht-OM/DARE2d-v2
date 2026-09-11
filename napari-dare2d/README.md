# `napari-dare2d/` — the DARE2D napari plugin

In-process napari plugin for DARE2D: detect cell divisions in a 2D `(T, Y, X)` movie and overlay
each division's centre and axis, inspect ground-truth annotations, and launch retraining /
fine-tuning — all from the napari GUI. It is a thin layer over the core `dare2d/` (TF/Keras) and
`dare2d-torch/` (PyTorch) code, reusing their inference and consensus rather than reimplementing
them, so the results match the CLI and notebooks.

![The DARE2D division-detection widget in napari: detected division centres (red Points) and axes
(cyan Vectors) overlaid on a 2D cell movie, with the detection widget docked at the right.](DARE2Dnapari.png)

The plugin is a separate installable package (`pip install -e ./napari-dare2d`) discovered by napari
through the **npe2 manifest** (`napari.yaml`).

<details>
<summary><b>Folder map</b></summary>

```
napari-dare2d/
├── pyproject.toml                  # packaging; dependencies intentionally EMPTY (see "Install")
├── napari_dare2d/
│   ├── __init__.py                 # import-light; sets KMP_DUPLICATE_LIB_OK so TF+Torch coexist
│   ├── napari.yaml                 # npe2 manifest: registers the 3 widgets below
│   ├── _api.py                     # napari-free in-process API over the DARE2D pipeline
│   ├── _widget.py                  # the magicgui dock widgets (the GUI)
│   ├── _data.py                    # Zenodo download + "save results" (stdlib-only)
│   └── download.py                 # CLI: python -m napari_dare2d.download (wraps _data)
├── verify_layers.py                # fast check: geometry + napari layer mapping (no models)
└── verify_api.py                   # real check: builds set-8 models, runs inference + consensus
```
</details>

## The three widgets (registered in `napari.yaml`)

| Widget (Plugins → DARE2D …) | Function | What it does |
|---|---|---|
| **DARE2D division detection** | `_widget.dare2d_widget` | Runs inference on an open Image layer (or a movie you browse to), overlaying a **Points** layer (centres) + **Vectors** layer (axes). Selects the **keras**/CPU or **pytorch**/GPU backend, the model set(s), frame range, and consensus parameters. After a run it reveals an inline **save results** section and, if the data is missing, a **Download data** button. |
| **DARE2D annotations viewer** | `_widget.annotations_widget` | Loads ground-truth `division_position*.npy` from a set folder and overlays the paired daughter cells (Points) + pair links (Vectors). |
| **DARE2D retraining & fine-tuning (beta)** | `_widget.retrain_widget` | **Beta / experimental.** Launches leave-one-out retraining (or PyTorch fine-tuning / transfer learning) as a subprocess (PyTorch GPU / TF CPU / TF WSL-GPU), with an epoch progress bar, an inline **Stop** button, and a **Download data** button when the dataset is absent. Writes to `models/<run>/…`; never touches the curated `models/demo/<dataset>/`. |

Inference runs in a `thread_worker` to keep the napari UI responsive; retraining runs in a killable
subprocess.

## `_api.py` — inference API (napari-free, headless-testable)

Pure functions that wrap the repository's own inference and consensus code, with no Qt/napari import,
so they can be unit-tested headlessly (see `verify_*.py`) and reused outside napari:

- **Model building / inference** — `build_models` (Hydra-instantiate the two TF models from
  `best.h5`), `infer_stack` (two-stage detection over a `(T,Y,X)` stack), `run_ensemble`
  (every model set), `consensus` (HDBSCAN/DBSCAN clustering + aggregation across models; the
  returned `angle` is in the same model convention as `infer_stack` dicts, see below).
- **Layer mapping** — `detections_to_points` / `detections_to_vectors` / `to_layer_data`
  turn `{frame: [{x,y,angle,length}]}` into napari `LayerDataTuple`s.
- **Annotations** — `annotation_pairs` / `annotations_to_layer_data` read `division_position*.npy`.
- **Helpers** — `read_stack`, `find_checkpoints`, `parse_sets`, `resolve_frames`, and the default
  paths `MODELS_DIR`, `DEFAULT_REG_DIR` / `DEFAULT_SEG_DIR` (= `models/demo/neuroepithelium/…`), `DATA_DIR`,
  `DEFAULT_ANNOT_DIR` (= set 8's ground truth).

### Coordinate conventions (locked — don't change without updating `verify_layers.py`)

- A detection's `x` = **column**, `y` = **row**; a napari point over a `(T,Y,X)` stack is
  therefore `(t, y, x)`.
- A division axis of angle θ and length L has direction `(0, cos θ·L, sin θ·L)` in `(t, row, col)`,
  drawn centred on the detection (a Vectors layer with `length=1`).
- Ground-truth `division_position{n}.npy` rows are `[row, col, frame]` (= `[y, x, frame]`, the order
  the training pipeline reads); `frame` is **1-based**, so napari `t = frame − 1`. Consecutive rows
  are the two daughter cells of one division.
- Consensus dicts (and the saved `*_summary.csv` `angle` column) use the **same θ** as single-set
  runs (from +row toward +col, as `convert_values` decodes it). The standalone CLI
  `scripts/postprocessing/main.py` instead stores `angle_deg = 90 − θ` folded to (−90, 90] (from
  +x/col toward +y/row, for its cv2 renderer); `_api.consensus` maps that back with
  `θ = 90 − a` if `a ≥ 0` else `−90 − a`. `verify_consensus_layers.py` asserts that a consensus of
  identical detections draws parallel to the single-set axis.

## Backends

Both backends feed the *same* `infer_stack` / consensus / layer-mapping code; only the model-building
step differs:

- **`keras`** — the TF/Keras models from `dare2d/`, loaded from `best.h5` (CPU on native Windows).
- **`pytorch`** — the faithful port in `dare2d-torch/`, loaded from `best.pt` in the *same*
  checkpoint folders (GPU). Produces identical detections (verified: 0 px centre shift, 0 angle/
  length difference). Selecting `keras` never imports torch and vice-versa (lazy import).

## `_data.py` — download & save

- `download_dataset(root, …, record_id=…, only=…, verify=True)` — fetches the Zenodo record
  (concept DOI `10.5281/zenodo.17442226` → latest version; `data.zip` + `models.zip`) with stdlib
  `urllib`/`zipfile`/`hashlib` (no extra dependency), md5-verifies each archive against Zenodo's
  checksum, caches the zips under `_zenodo_cache/`, and extracts them at `root` — the archives are
  rooted at `data/` and `models/`, so this yields `models/demo/neuroepithelium/…` and
  `data/demo/neuroepithelium/…` directly (members outside those folders are refused). Extraction
  **only adds missing files** — it never overwrites, so a populated `models/` (e.g. retrained
  checkpoints) is left intact. Writes `_zenodo_cache/zenodo_manifest.json` (resolved record id,
  DOI, version, md5s). `progress_cb(done, total)` is cumulative over all archives; the widget's
  progress bar is driven off it via a `QTimer`. `missing_files(root)` / `expected_paths(root)`
  report which of the 8 sets and 32 checkpoint files are present.
- `python -m napari_dare2d.download [--root R] [--only data|models] [--record ID] [--no-verify] [--check]`
  (`download.py`) is the CLI over the same function; `dare2d-download` is its console-script alias.
- `save_results(image, points, features, out_dir, …)` — writes per-frame `division_position*.npy`
  (`[x, y]` pairs), a `*_summary.csv` (frame, x, y, angle, length, …) and an optional overlay
  `*_result.tiff` movie.

## Install & packaging

`pyproject.toml` declares **no dependencies on purpose**. The plugin runs inside the carefully
version-pinned env (`napari-env-for-DARE2D`, numpy `1.23.5`); letting `pip` re-resolve dependencies
here would pull a newer numpy and break TensorFlow 2.12. Install it without touching the pins:

```bash
pip install --no-build-isolation --no-deps -e ./napari-dare2d
```

The `_api` adapts to the installed napari version (the Points outline kwarg changed from
`edge_color` in 0.4.x to `border_color` in ≥0.5), so the same code works across napari versions.

## Checks

```bash
python napari-dare2d/verify_layers.py   # fast: geometry + napari LayerDataTuple mapping (no models)
python napari-dare2d/verify_api.py      # real: builds set-8 models, runs inference + consensus
```

`verify_layers.py` asserts the coordinate conventions above and that napari accepts the produced
layers; `verify_api.py` builds the set-8 models from `models/demo/neuroepithelium/` and runs the full pipeline on the
set-8 stack (needs the Zenodo checkpoints + data — fetch them with `python -m napari_dare2d.download`
or the widget's **Download data** button).

See the repository [`README.md`](../README.md) for the end-to-end install, inference and retraining
walkthroughs, and [`../dare2d/`](../dare2d/) for the core package this plugin wraps.

## Authors

Romain Karpinski, Alice Gros, Marc Karnat, Qazi Saaheelur Rahaman, Jules Vanaret, Mehdi Saadaoui,
Sham L. Tlili, and Jean-François Rupprecht.

Aix Marseille Univ, Inserm U1067, CNRS, LAI (UMR 7333), Laboratoire Adhésion Inflammation, Turing
Centre for Living Systems, Marseille, France.

Maintainer: Jean-François Rupprecht — <rupprecht.jf@gmail.com>.

## Citation

Please cite the references relevant to your use:

- **Method paper (preprint).** Karpinski R., Gros A., Karnat M., Saaheelur Rahaman Q., Vanaret J.,
  Saadaoui M., Tlili S., Rupprecht J.-F. (2026). *DARE: Division Axis and Region Estimation from
  2D and 3D Time-Lapse Images.* bioRxiv. DOI `10.1101/2024.02.05.578987`.
- **Data & pretrained models (Zenodo archive).** Concept DOI `10.5281/zenodo.17442226`
  (<https://doi.org/10.5281/zenodo.17442226>, resolves to the latest version): the model checkpoints
  (`models.zip`, Keras `.h5` + PyTorch `.pt`) and the neuroepithelium dataset (`data.zip`).

BibTeX for the paper:

```bibtex
@article{Karpinski2024.02.05.578987,
  author       = {Karpinski, Romain and Gros, Alice and Karnat, Marc and Saaheelur Rahaman, Qazi and
                  Vanaret, Jules and Saadaoui, Mehdi and Tlili, Sham and Rupprecht, Jean-Fran{\c c}ois},
  title        = {DARE: Division Axis and Region Estimation from 2D and 3D Time-Lapse Images},
  elocation-id = {2024.02.05.578987},
  year         = {2026},
  doi          = {10.1101/2024.02.05.578987},
  publisher    = {Cold Spring Harbor Laboratory},
  journal      = {bioRxiv},
  url          = {https://www.biorxiv.org/content/early/2026/03/27/2024.02.05.578987}
}
```

## License & acknowledgements

MIT — see [`LICENSE`](../LICENSE). This work was granted access to the HPC resources of IDRIS under
the allocation AD010314339 made by GENCI.
