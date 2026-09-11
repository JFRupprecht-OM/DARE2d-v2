"""magicgui dock widget: run DARE2D on a loaded Image layer, overlay results.

Runs inference in a ``thread_worker`` (keeps napari responsive) and adds the
results as a Points layer (division centres) + a Vectors layer (division axes).
All heavy lifting / data mapping lives in ``_api`` (napari-free, tested headless).

Two inference backends, selectable in the widget:
  - **keras**  : the original TF/Keras models (CPU on native Windows).
  - **pytorch**: the faithful torch port in ``dare2d-torch/`` (GPU), loading ``best.pt``
    from the SAME checkpoint dirs as Keras. Produces the same detections.
Both feed the SAME ``infer_stack`` / consensus / layer-mapping code, so only the
model-building step differs.
"""

import datetime
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import napari
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import CheckBox, FileEdit, Label, ProgressBar, PushButton
from napari.qt.threading import thread_worker
from napari.utils import notifications

from . import _api


def _pytorch_builder(sets, reg_dir, seg_dir, device=None):
    """Return ``build(i) -> (reg, seg)`` torch models, or raise a clear error.

    Loads ``best.pt`` from the SAME checkpoint dirs as the Keras backend
    (``<dir>/checkpoints_set_{n}_all_but_target/best.pt``), so a retrained run dir works
    with no rename. The torch port lives in the sibling ``dare2d-torch/`` folder; add it
    to sys.path lazily, inside the worker, so selecting Keras never imports torch.

    ``device`` picks where inference runs: ``None``/"auto" -> CUDA if available else CPU;
    "cpu" -> force CPU; "cuda" -> force GPU (errors clearly if no CUDA device is present).
    """
    dt = _api.PROJECT_ROOT / "dare2d-torch"
    if str(dt) not in sys.path:
        sys.path.insert(0, str(dt))
    try:
        import torch_backend as tb
    except Exception as e:  # torch not installed, etc.
        raise RuntimeError(
            f"PyTorch backend unavailable: {e}. Install torch into this env "
            f"(see requirements-torch.txt)."
        ) from e
    if device in (None, "auto"):
        device = tb.default_device()                 # CUDA if available, else CPU
    elif device == "cuda" and not tb.torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference requested but no CUDA device is available in this env. "
            "Set the device to 'cpu' (or 'auto'), or install a CUDA-enabled torch."
        )
    try:
        reg_pts, seg_pts = tb.find_torch_checkpoints(reg_dir, seg_dir, sets)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{e}. Switch the backend to 'keras', or fetch the weights via 'Download "
            f"DARE2D data' (ships best.pt next to best.h5 in the checkpoint folders); "
            f"devs regenerate with dare2d-torch/convert_to_torch.py."
        ) from e
    return lambda i: (tb.load_torch_regression(reg_pts[i], device=device),
                      tb.load_torch_segmentation(seg_pts[i], device=device))


def _checkpoints_present():
    """True once the inference checkpoints have been downloaded."""
    return (any(_api.DEFAULT_REG_DIR.glob("checkpoints_set_*/best.h5"))
            and any(_api.DEFAULT_SEG_DIR.glob("checkpoints_set_*/best.h5")))


def _dataset_present():
    """True once the raw neuroepithelium dataset has been downloaded."""
    return any(_api.DATA_DIR.glob("set_*"))


def _data_complete():
    """True only when BOTH the checkpoints and the dataset are present (the full Zenodo
    download). Both widgets show their 'Download data' button until this holds."""
    return _checkpoints_present() and _dataset_present()


def _add_download_section(widget, present_fn):
    """Append a 'Download DARE2D data' button (+ progress bar) that shows only when the
    data ``present_fn`` checks for is missing. Clicking downloads from Zenodo with a live
    progress bar; the button hides itself once the data is present.
    """
    btn = PushButton(text="Download DARE2D data (Zenodo, ~2 GB)")
    btn.tooltip = ("Download the DARE2D checkpoints + neuroepithelium dataset from Zenodo "
                   "(~2 GB) into this project, then this button disappears.")
    bar = ProgressBar(value=0)
    bar.min, bar.max = 0, 100
    bar.visible = False
    btn.visible = not present_fn()
    widget.insert(0, btn)   # top of the widget, above the inputs
    widget.insert(1, bar)

    def _on_click():
        from qtpy.QtCore import QTimer

        from ._data import download_dataset

        btn.enabled = False
        bar.visible = True
        bar.label = "starting…"
        state = {"pct": 0, "msg": "downloading…"}

        def _progress(done, total):
            if total:
                state["pct"] = int(100 * done / total)

        @thread_worker
        def run():
            return download_dataset(_api.PROJECT_ROOT, progress_cb=_progress,
                                    log=lambda m: state.__setitem__("msg", m))

        # The worker runs off-thread; a GUI-thread timer marshals progress to the bar.
        timer = QTimer()
        timer.setInterval(200)
        timer.timeout.connect(lambda: (setattr(bar, "value", state["pct"]),
                                       setattr(bar, "label", state["msg"])))

        def _finish(ok, info):
            timer.stop()
            bar.visible = False
            btn.enabled = True
            if ok:
                btn.visible = False          # data now present -> hide the button
                notifications.show_info(f"DARE2D data ready under {info}")
            else:
                notifications.show_error(f"DARE2D download failed: {info}")

        worker = run()
        worker.returned.connect(lambda root: _finish(True, root))
        worker.errored.connect(lambda e: _finish(False, e))
        widget._dl_timer = timer             # keep the timer alive for the run
        worker.start()
        timer.start()

    btn.clicked.connect(_on_click)


_DIV = {}  # holds the division widget's 'save results' controls (revealed after a run)


def _default_out_dir():
    """A fresh, date-stamped run folder under output/ so each run's results land in their own
    directory: ``output/dare2d_<YYYY-MM-DD>`` (a numeric suffix is added if it already exists)."""
    base = _api.PROJECT_ROOT / "output"
    stamp = datetime.date.today().isoformat()
    cand = base / f"dare2d_{stamp}"
    n = 2
    while cand.exists():
        cand = base / f"dare2d_{stamp}_{n}"
        n += 1
    return cand


def _add_save_section(widget):
    """Append a hidden 'save results' section to the division widget; ``_on_return`` reveals
    it once a detection run finishes (replaces the old standalone save-results widget)."""
    out = FileEdit(mode="d", label="Output folder", value=_default_out_dir())
    out.tooltip = ("Folder to write per-frame division_position*.npy, a *_summary.csv and "
                   "(optionally) an overlay *_result.tiff into. Defaults to a fresh date-stamped "
                   "run folder under output/ (refreshed each run).")
    overlay = CheckBox(value=True, label="Also save overlay movie (.tiff)")
    overlay.tooltip = "Render a (T, Y, X) RGB overlay tiff with the detections drawn on the movie."
    btn = PushButton(text="Save DARE2D results")
    btn.tooltip = "Save the detections from the last run (the DARE2D Points layer) to disk."
    for w_ in (out, overlay, btn):
        w_.visible = False
        widget.append(w_)
    _DIV["out"] = out

    def _save():
        import napari.layers as nl
        viewer = napari.current_viewer()
        if viewer is None:
            return
        pts = next((ly for ly in reversed(viewer.layers)
                    if isinstance(ly, nl.Points) and "DARE2D" in str(ly.name)), None)
        if pts is None:
            notifications.show_warning("No DARE2D Points layer found — run DARE2D first.")
            return
        img = next((ly for ly in reversed(viewer.layers) if isinstance(ly, nl.Image)), None)
        if img is None:
            notifications.show_warning("No Image layer found to render the overlay.")
            return
        feats = {}
        try:  # napari 0.5: a features DataFrame; fall back to the older properties dict
            for c in pts.features.columns:
                feats[c] = np.asarray(pts.features[c])
        except Exception:
            feats = {k: np.asarray(v) for k, v in (pts.properties or {}).items()}
        from ._data import save_results
        saved = save_results(np.asarray(img.data), np.asarray(pts.data), feats,
                             out.value, image_name=str(img.name), draw=overlay.value)
        notifications.show_info(f"DARE2D: saved results → {saved}")

    btn.clicked.connect(_save)
    _DIV["save_widgets"] = (out, overlay, btn)


def _add_advanced_section(widget):
    """Group the fine-tuning controls under an 'Advanced parameters' toggle. magicgui has no
    native collapsible, so a PushButton flips the controls' .visible (collapsed by default;
    click to expand, click again to collapse)."""
    advanced = (widget.device, widget.seg_threshold, widget.eps, widget.min_models,
                widget.angle_mode, widget.min_cluster_size, widget.min_samples)
    toggle = PushButton(text="▸ Advanced parameters")
    toggle.tooltip = ("Show/hide compute device (CPU/GPU), segmentation threshold and "
                      "consensus tuning (eps, min models).")
    for w_ in advanced:
        w_.visible = False                                  # collapsed by default
    widget.insert(list(widget).index(widget.device), toggle)  # toggle sits above them

    def _toggle():
        show = not advanced[0].visible
        for w_ in advanced:
            w_.visible = show
        toggle.text = "▾ Advanced parameters" if show else "▸ Advanced parameters"

    toggle.clicked.connect(_toggle)


def _division_widget_init(widget):
    """Download-data button + hidden save-results section + collapsible advanced parameters;
    show a picked movie immediately and pre-load the demo movie; tooltips."""
    _add_download_section(widget, _data_complete)
    _add_save_section(widget)
    _add_advanced_section(widget)

    def _show_movie(path):
        """Load the picked movie into the viewer and select it as the Run input, so choosing a
        file shows it immediately (reusing an existing layer of the same name -- no duplicates)."""
        viewer = napari.current_viewer()
        if viewer is None or not path:
            return
        p = Path(path)
        if not p.is_file():
            return
        import napari.layers as nl
        layer = next((ly for ly in viewer.layers
                      if isinstance(ly, nl.Image) and ly.name == p.stem), None)
        if layer is None:
            layer = viewer.add_image(_api.read_stack(p), name=p.stem)
        try:
            widget.image.value = layer      # use it as the Run input (movie shows regardless)
        except Exception:
            pass

    widget.movie.changed.connect(_show_movie)
    # Pre-load the demo movie shipped beside the set_N folders (found by extension, so a rename
    # of that file is fine as long as it stays there). Setting .value fires _show_movie.
    dm = _api.default_movie()
    if dm is not None:
        widget.movie.value = dm
        # current_viewer() may not be ready during widget construction; defer one load to the
        # next event-loop tick so the demo movie reliably appears (deduped by layer name).
        from qtpy.QtCore import QTimer
        QTimer.singleShot(0, lambda: _show_movie(widget.movie.value))

    cb = getattr(widget, "_call_button", None)
    _DIV["call_button"] = cb                       # so the run can toggle Run <-> Stop
    if cb is not None:
        cb.tooltip = ("Detect divisions in the selected stack and overlay the centres "
                      "(Points layer) and division axes (Vectors layer). While running, this "
                      "button becomes Stop DARE2D to interrupt the run.")


@magic_factory(
    widget_init=_division_widget_init,
    call_button="Run DARE2D",
    image={"label": "Image layer (already open)",
           "tooltip": "Run on an Image layer already open in napari. Leave empty if you "
                      "load a movie file below instead. Default: none (use the open layer)."},
    movie={"mode": "r", "label": "…or load a movie (.tif)",
           "tooltip": "Browse for an 8-bit (T, Y, X) .tif/.tiff stack; it loads and displays "
                      "immediately and becomes the Run input. Default: the demo movie beside "
                      "the set folders (auto-loaded if present)."},
    backend={"choices": ["pytorch", "keras"], "label": "Inference backend",
             "tooltip": "pytorch = GPU/CUDA (default; loads .pt weights). keras = TensorFlow "
                        "on CPU (loads .h5) — same detections. Default: pytorch."},
    reg_dir={"label": "Regression checkpoints", "mode": "d",
             "tooltip": "Folder of regression checkpoints "
                        "(…/checkpoints_set_N_all_but_target/best.h5 or best.pt); point it at "
                        "a retrained run dir to use those weights. "
                        "Default: models/demo/neuroepithelium/regression_checkpoints."},
    seg_dir={"label": "Segmentation checkpoints", "mode": "d",
             "tooltip": "Folder of segmentation (centre-detection) checkpoints, same layout "
                        "as the regression folder. "
                        "Default: models/demo/neuroepithelium/segmentation_checkpoints."},
    model_sets={"label": "Model sets (e.g. 1-7, or 8)",
                "tooltip": "Which trained model set(s) to run: e.g. '8' for one model, '1-7' "
                           "for the 7-model ensemble. Several sets → per-frame consensus. "
                           "Leave blank (no number) to auto-detect the set(s) present in the "
                           "selected checkpoint folders. Default: 8."},
    frame_start={"label": "First frame",
                 "tooltip": "First frame to process (0-based). Default: 0."},
    frame_end={"label": "Last frame (-1 = end)",
               "tooltip": "Last frame to process (inclusive); -1 means the final frame. "
                          "Default: -1."},
    device={"choices": ["auto", "gpu", "cpu"], "label": "Compute device",
            "tooltip": "Where PyTorch inference runs (ignored by the keras backend, which is "
                       "TensorFlow-on-CPU): auto = GPU if a CUDA device is available else CPU; "
                       "gpu = force CUDA (errors if none); cpu = force CPU. Default: auto."},
    seg_threshold={"label": "Seg. threshold", "min": 0.0, "max": 1.0, "step": 0.05,
                   "tooltip": "Probability cutoff on the U-Net segmentation map (0–1); lower "
                              "= more / smaller detections. Default: 0.5."},
    eps={"label": "Consensus eps (px)",
         "tooltip": "Consensus clustering radius in pixels: detections from different models "
                    "within this distance merge into one division (ensemble only). "
                    "Default: 10."},
    min_models={"label": "Min models (consensus)",
                "tooltip": "Minimum number of models that must agree (within eps) to keep a "
                           "consensus detection (ensemble only). Default: 6."},
    angle_mode={"choices": ["auto", "degrees", "radians"], "label": "Angle mode",
                "tooltip": "How stored angle units are interpreted before consensus "
                           "(ensemble only): auto-detect, or force degrees/radians. "
                           "In-process angles are always degrees, so leave on auto/degrees. "
                           "Default: auto."},
    min_cluster_size={"label": "Min cluster size", "min": 2,
                      "tooltip": "HDBSCAN minimum cluster size when grouping detections into "
                                 "a consensus (ensemble only). Default: 2."},
    min_samples={"label": "Min samples", "min": 1,
                 "tooltip": "HDBSCAN/DBSCAN min_samples for consensus grouping (ensemble "
                            "only); higher = more conservative. Default: 1."},
    pbar={"visible": False, "max": 0, "label": "idle"},
)
def dare2d_widget(
    image: "napari.layers.Image",
    movie: Path = Path(""),
    backend: str = "pytorch",
    reg_dir: Path = _api.DEFAULT_REG_DIR,
    seg_dir: Path = _api.DEFAULT_SEG_DIR,
    model_sets: str = "8",
    frame_start: int = 0,
    frame_end: int = -1,
    device: str = "auto",
    seg_threshold: float = 0.5,
    eps: float = 10.0,
    min_models: int = 6,
    angle_mode: str = "auto",
    min_cluster_size: int = 2,
    min_samples: int = 1,
    pbar: ProgressBar = None,
):
    """Detect cell divisions in the selected (T, Y, X) stack and overlay them.

    One model set -> raw detections. Several sets -> per-frame consensus.
    ``backend`` picks Keras (TF) or the PyTorch port (GPU); results are the same.
    A click while a run is active stops it (the Run button toggles to **Stop DARE2D**).
    """
    # A click while a run is in progress means STOP: abort the worker and bail out.
    active = _DIV.get("worker")
    if active is not None:
        active.quit()
        _cb = _DIV.get("call_button")
        if _cb is not None:
            _cb.text = "Stopping…"
        return
    viewer = napari.current_viewer()
    # Input precedence: the selected Image layer; else the movie file -- reusing an already-
    # loaded layer of the same name so picking a movie never double-adds it (the movie picker
    # also loads it on selection; see _division_widget_init).
    if image is None and movie and Path(movie).is_file():
        import napari.layers as nl
        stem = Path(movie).stem
        image = next((ly for ly in viewer.layers
                      if isinstance(ly, nl.Image) and ly.name == stem), None)
        if image is None:
            image = viewer.add_image(_api.read_stack(movie), name=stem)
    if image is None:
        raise ValueError("Pick a movie file (.tif) or select an open Image layer first.")
    stack = np.asarray(image.data)
    if stack.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {stack.shape}")

    # Resolve inputs eagerly so bad paths/specs error before the worker starts.
    # A blank / number-free field means "auto": discover the sets present in the
    # selected checkpoint folders instead of requiring an explicit number/range.
    if any(ch.isdigit() for ch in str(model_sets)):
        sets = _api.parse_sets(model_sets)
    else:
        sets = _api.discover_sets(reg_dir, seg_dir, backend=backend)
    frames = _api.resolve_frames(stack.shape[0], frame_start, frame_end)
    n_frames = stack.shape[0]
    base_name = image.name
    # Tag the output layers with the run's date+time so successive runs are self-identifying.
    run_stamp = datetime.datetime.now().strftime("%d/%m/%y %H:%M")
    # Keras needs the .h5 checkpoints; the torch backend loads .pt by set number.
    reg_ckpts = seg_ckpts = None
    if backend == "keras":
        reg_ckpts, seg_ckpts = _api.find_checkpoints(reg_dir, seg_dir, sets)

    @thread_worker
    def run():
        if backend == "pytorch":
            torch_device = {"gpu": "cuda", "cpu": "cpu"}.get(device, "auto")
            build = _pytorch_builder(sets, reg_dir, seg_dir, device=torch_device)
        else:
            build = lambda i: _api.build_models(reg_ckpts[i], seg_ckpts[i])  # noqa: E731

        def cleanup(reg, seg):
            del reg, seg
            if backend == "keras":
                import tensorflow.keras.backend as K
                K.clear_session()  # free graph between sets (CPU/RAM hygiene)

        if len(sets) == 1:
            reg, seg = build(0)
            per_frame = {}
            for k, i in enumerate(frames):
                per_frame.update(_api.infer_stack(stack, reg, seg, frames=[i],
                                                  threshold=seg_threshold))
                yield (k + 1, len(frames))
            return _api.to_layer_data(per_frame, frame_base=0,
                                      name=f"{base_name} DARE2D ({backend}) {run_stamp}")

        # Ensemble: per-set detections -> consensus. Same primitives as
        # _api.run_ensemble, inlined so we can yield per-model progress.
        acc = defaultdict(list)
        for m in range(len(sets)):
            reg, seg = build(m)
            res = _api.infer_stack(stack, reg, seg, frames=frames, threshold=seg_threshold)
            for i, dets in res.items():
                for d in dets:
                    acc[i + 1].append(
                        (m + 1, float(d["x"]), float(d["y"]),
                         float(d["angle"]), float(d["length"]))
                    )
            cleanup(reg, seg)
            yield (m + 1, len(sets))
        cons = _api.consensus(
            dict(acc), n_frames=n_frames, eps=eps,
            min_models=min_models, num_models=len(sets), angle_mode=angle_mode,
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        )
        return _api.to_layer_data(cons, frame_base=1,
                                  name=f"{base_name} DARE2D consensus ({backend}) {run_stamp}")

    def _on_yield(v):
        done, total = v
        pbar.max = total
        pbar.value = done
        pbar.label = f"DARE2D {done}/{total}"

    def _on_return(layer_data):
        for data, meta, ltype in layer_data:
            viewer._add_layer_from_data(data, meta, ltype)
        out_w = _DIV.get("out")                    # propose a fresh date-stamped run folder
        if out_w is not None:
            out_w.value = _default_out_dir()
        for w_ in _DIV.get("save_widgets", ()):   # reveal the save-results section
            w_.visible = True

    cb = _DIV.get("call_button")

    def _on_start():
        pbar.visible = True
        if cb is not None:
            cb.text = "Stop DARE2D"

    def _on_finish():
        pbar.visible = False
        _DIV["worker"] = None
        if cb is not None:
            cb.text = "Run DARE2D"

    worker = run()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_on_return)
    worker.started.connect(_on_start)
    worker.finished.connect(_on_finish)
    _DIV["worker"] = worker
    worker.start()
    return worker


@magic_factory(
    call_button="Load annotations",
    folder={"mode": "d", "label": "Annotations folder"},
    show_links={"label": "Draw pair links"},
    point_size={"label": "Point size"},
)
def annotations_widget(
    folder: Path = _api.DEFAULT_ANNOT_DIR,
    show_links: bool = True,
    point_size: int = 12,
):
    """Overlay ground-truth annotations (paired daughter cells) from .npy files.

    Reads ``division_position*.npy`` in ``folder`` and adds a Points layer (every
    annotated cell) and a Vectors layer linking each daughter-cell pair. Fast
    (file reads only), so it runs synchronously.
    """
    viewer = napari.current_viewer()
    if viewer is None:
        raise RuntimeError("no active napari viewer")
    layer_data = _api.annotations_to_layer_data(
        folder, point_size=point_size, show_links=show_links
    )
    for data, meta, ltype in layer_data:
        viewer._add_layer_from_data(data, meta, ltype)


# ---------------------------------------------------------------------------
# Retraining widget (leave-one-out) with a backend toggle
# ---------------------------------------------------------------------------
# Spawns the training drivers in training/ as a subprocess (training is long/heavy;
# a subprocess keeps napari responsive, can use a different env/WSL, and is killable).
# Three backends -> three commands:
#   PyTorch (GPU)       : python training/torch/train.py         (native Windows GPU)
#   TensorFlow (CPU)    : python training/tf/train_split.py      (Windows, CPU)
#   TensorFlow (WSL GPU): wsl bash training/tf/wsl/run_train.sh  (TF 2.12 GPU in WSL2)
# Each writes models/<run>/<reg|seg>_checkpoints/...; the curated models/demo/<dataset>/ is never touched.

_TRAIN_DIR = _api.PROJECT_ROOT / "training"
_RUN = {"proc": None, "cancel": False}            # proc + cancel flag + the inline Stop button
_EPOCH_RE = re.compile(r"(?:\[epoch |Epoch )(\d+)/(\d+)")
_STEP_RE = re.compile(r"\[step\] (\d+)/(\d+) (\d+)/(\d+)")   # epoch/epochs step/steps
_PHASE_RE = re.compile(r"\[phase\] (.+)")
_PREP_RE = re.compile(r"\[prep\] (\S+):")
_EXP_MAP = {"both": ["regression2d", "segmentation2d"],
            "regression": ["regression2d"], "segmentation": ["segmentation2d"]}


def _echo(text):
    """Mirror a chunk of the training subprocess's output to the terminal napari was
    launched from, so the FULL log (progress bar, %, it/s, eta, per-epoch losses) is
    visible there — not only summarized in the GUI progress bar. Best-effort and never
    fatal: a logging hiccup must not interrupt a training run."""
    out = sys.stdout if sys.stdout is not None else sys.__stdout__
    if out is None:                       # e.g. pythonw / no console attached
        return
    try:
        out.write(text)
        out.flush()
    except Exception:
        pass


def _win_to_wsl(p):
    """C:\\a\\b -> /mnt/c/a/b (for invoking WSL on a Windows path)."""
    p = str(p)
    return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"


def _retrain_widget_init(widget):
    """Add a 'Download data' button (when the full Zenodo data is incomplete) and a 'Stop
    retraining' button (shown only while a run is active). The WSL-GPU backend shells out to
    ``wsl`` and is Windows-only, so it is hidden on other platforms."""
    # Steer users toward the notebooks, which are the recommended path for both regimes.
    note = Label(value="The preferred options for retraining and fine-tuning are the "
                       "corresponding notebooks available in the notebooks folder.")
    note.tooltip = ("See notebooks/Run_dare2d_Retraining.ipynb and "
                    "notebooks/Run_dare2d_Finetune.ipynb.")
    widget.insert(0, note)
    _add_download_section(widget, _data_complete)
    if sys.platform != "win32":
        widget.backend.choices = [c for c in widget.backend.choices if "WSL" not in c]
    # One progress bar per stage, hidden until that stage runs, so the segmentation bar
    # appears below the regression bar once the regression stage is under way.
    bars = {}
    for _stage in ("regression", "segmentation"):
        b = ProgressBar(max=0, label=f"{_stage} stage:")
        b.visible = False
        widget.append(b)
        bars[_stage] = b
    _RUN["bars"] = bars
    stop = PushButton(text="Stop retraining")
    stop.visible = False
    stop.tooltip = "Cancel the running retraining (terminates the training subprocess)."

    def _stop():
        _RUN["cancel"] = True
        p = _RUN.get("proc")
        if p is not None and p.poll() is None:
            p.terminate()

    stop.clicked.connect(_stop)
    widget.append(stop)
    _RUN["stop_btn"] = stop
    cb = getattr(widget, "_call_button", None)
    _RUN["call_button"] = cb               # hidden while a run is active (Stop shows instead)
    if cb is not None:
        cb.tooltip = ("Launch leave-one-out retraining as a subprocess; progress shows in "
                      "the bar and the terminal.")

    # --- Transfer-learning / fine-tune controls -------------------------------------------
    # The fine-tune knobs live under a collapsed "Advanced parameters" toggle, and the whole
    # fine-tune UI (base-model picker + the advanced toggle) shows only in fine-tune mode. In
    # that mode the scratch backend selector is hidden (fine-tune is PyTorch-only) and Model is
    # restricted to a single stage (a .pt is regression OR segmentation).
    ft_params = (widget.unfreeze_last, widget.bn_mode, widget.ft_lr, widget.discriminative,
                 widget.backbone_lr_mult, widget.weight_decay, widget.lr_schedule,
                 widget.warmup_epochs, widget.grad_clip, widget.batch_size,
                 widget.augment, widget.augment_strength, widget.patience)
    ft_toggle = PushButton(text="▸ Advanced parameters")
    ft_toggle.tooltip = ("Show/hide fine-tuning options: freeze depth, learning rates, schedule, "
                         "augmentation, early stopping.")
    widget.insert(list(widget).index(widget.unfreeze_last), ft_toggle)

    def _toggle_ft():
        show = not ft_params[0].visible
        for w_ in ft_params:
            w_.visible = show
        ft_toggle.text = "▾ Advanced parameters" if show else "▸ Advanced parameters"

    ft_toggle.clicked.connect(_toggle_ft)
    _ALL_MODELS = ["both", "regression", "segmentation"]

    def _apply_mode(*_):
        finetune = str(widget.mode.value).startswith("Transfer")
        widget.backend.visible = not finetune
        ft_toggle.visible = finetune
        if list(widget.model.choices) != _ALL_MODELS:    # both/regression/segmentation in both modes
            widget.model.choices = _ALL_MODELS
        if not finetune:                                 # collapse + hide all fine-tune controls
            for w_ in ft_params:
                w_.visible = False
            ft_toggle.text = "▸ Advanced parameters"
            widget.base_reg.visible = False
            widget.base_seg.visible = False
        else:                                            # show the base picker(s) for the stage(s)
            stage = str(widget.model.value)
            widget.base_reg.visible = stage in ("both", "regression")
            widget.base_seg.visible = stage in ("both", "segmentation")

    widget.mode.changed.connect(_apply_mode)
    widget.model.changed.connect(_apply_mode)            # base picker(s) depend on the chosen stage(s)
    _apply_mode()                                        # initial state = scratch


@magic_factory(
    widget_init=_retrain_widget_init,
    call_button="Start retraining",
    mode={"choices": ["Retrain from scratch", "Transfer learning / fine-tune"], "label": "Mode",
          "tooltip": "Retrain from scratch (existing) or fine-tune a pretrained .pt checkpoint "
                     "(PyTorch only)."},
    base_reg={"mode": "r", "label": "Regression model (.pt)", "filter": "*.pt",
              "tooltip": "Pretrained REGRESSION .pt to fine-tune (e.g. torch_reg_set_8.pt or "
                         "regression_checkpoints/.../best.pt). Shown for Model = regression or both."},
    base_seg={"mode": "r", "label": "Segmentation model (.pt)", "filter": "*.pt",
              "tooltip": "Pretrained SEGMENTATION .pt to fine-tune (e.g. torch_seg_set_8.pt or "
                         "segmentation_checkpoints/.../best.pt). Shown for Model = segmentation or both."},
    test_set={"choices": [1, 2, 3, 4, 5, 6, 7, 8], "label": "Test set (held out)",
              "tooltip": "The set held out for validation (leave-one-out); the model "
                         "trains on the others and is tested on this one."},
    train_sets={"label": "Train sets (blank = complement)",
                "tooltip": "Comma-separated sets to train on (e.g. '1,2,3'). Blank = every "
                           "set except the held-out test set."},
    model={"choices": ["both", "regression", "segmentation"],
           "tooltip": "Which stage(s) to retrain: both, just the regression head, or just "
                      "the segmentation (centre-detection) U-Net."},
    backend={"choices": ["PyTorch (GPU)", "TensorFlow (CPU)", "TensorFlow (WSL GPU)"],
             "label": "Backend",
             "tooltip": "PyTorch (GPU) = native-Windows CUDA. TensorFlow (CPU) = native "
                        "Windows (slow). TensorFlow (WSL GPU) = TF 2.12 GPU inside WSL2."},
    raw_dir={"mode": "d", "label": "Data folder (raw sets)",
             "tooltip": "Folder of raw sets (set_1…set_8), each with the movie .tif + "
                        "division_position*.npy ground truth."},
    run_name={"label": "Run name (blank = date)",
              "tooltip": "Output subfolder under models/<run>/; blank = today's date. "
                         "The curated models/demo/<dataset>/ is never overwritten."},
    epochs={"tooltip": "Number of training epochs per stage."},
    steps={"tooltip": "Optimizer steps per epoch."},
    crop={"tooltip": "Crop size in pixels that each frame is tiled into for training."},
    unfreeze_last={"label": "Unfreeze last N blocks",
                   "tooltip": "Backbone is frozen; unfreeze the last N backbone blocks. "
                              "Default: 0 (head only)."},
    bn_mode={"choices": ["frozen", "adapt"], "label": "Frozen-backbone BN",
             "tooltip": "Normalization of the frozen backbone. 'frozen' (default): BatchNorm "
                        "running stats fixed (BN .eval()) - a true freeze. 'adapt': BatchNorm "
                        "re-estimates stats on the new data (weights stay frozen) - for a larger, "
                        "distribution-shifted fine-tune set."},
    ft_lr={"label": "Fine-tune LR",
           "tooltip": "Learning rate for the (unfrozen) head. Default: 1e-4 (lower than scratch)."},
    discriminative={"label": "Discriminative LR",
                    "tooltip": "Use a lower LR for the unfrozen backbone than the head. Default: on."},
    backbone_lr_mult={"label": "Backbone LR x",
                      "tooltip": "Unfrozen-backbone LR = Fine-tune LR x this. Default: 0.1."},
    weight_decay={"label": "Weight decay", "tooltip": "AdamW weight decay. Default: 1e-4."},
    lr_schedule={"choices": ["cosine", "constant"], "label": "LR schedule",
                 "tooltip": "Linear warmup then cosine decay (default) or constant. Default: cosine."},
    warmup_epochs={"label": "Warmup epochs",
                   "tooltip": "Linear LR warmup epochs before the schedule. Default: 1."},
    grad_clip={"label": "Grad clip (max-norm)",
               "tooltip": "Gradient max-norm clipping; 0 = off. Default: 1.0."},
    batch_size={"tooltip": "Mini-batch size (fine-tune). Default: 32."},
    augment={"label": "Augmentation",
             "tooltip": "Apply the training-data augmentations. Default: on."},
    augment_strength={"label": "Augment strength",
                      "tooltip": "Scales augmentation probabilities (0-1). Default: 1.0."},
    patience={"label": "Early-stop patience",
              "tooltip": "Stop if val loss doesn't improve for N epochs; 0 = off. Default: 0."},
)
def retrain_widget(
    mode: str = "Retrain from scratch",
    model: str = "both",
    base_reg: Path = Path(),
    base_seg: Path = Path(),
    test_set: int = 8,
    train_sets: str = "",
    backend: str = "PyTorch (GPU)",
    raw_dir: Path = _api.DATA_DIR,
    run_name: str = "",
    epochs: int = 50,
    steps: int = 1000,
    crop: int = 256,
    unfreeze_last: int = 0,
    bn_mode: str = "frozen",
    ft_lr: str = "1e-4",
    discriminative: bool = True,
    backbone_lr_mult: str = "0.1",
    weight_decay: str = "1e-4",
    lr_schedule: str = "cosine",
    warmup_epochs: int = 1,
    grad_clip: str = "1.0",
    batch_size: int = 32,
    augment: bool = True,
    augment_strength: str = "1.0",
    patience: int = 0,
):
    """Retrain DARE2D on a subset of sets, testing on the held-out ``test_set``.

    Output goes to ``models/<run>/`` (dated; never overwrites an existing best.h5).
    A **Stop retraining** button appears while a run is active. Progress/logs also
    print to the terminal.
    """
    if _RUN["proc"] is not None and _RUN["proc"].poll() is None:
        raise RuntimeError("a retraining is already running (use the Stop button first)")
    finetune = str(mode).startswith("Transfer")
    exps = _EXP_MAP[model]
    # each fine-tune stage uses its own base .pt: regression -> base_reg, segmentation -> base_seg
    # (Model = both runs both stages, each from its own checkpoint).
    _base_for = {"regression2d": Path(str(base_reg)), "segmentation2d": Path(str(base_seg))}
    if finetune:                                        # validate up-front: clear message, fail fast
        for exp in exps:
            bm = _base_for[exp]
            stg = "regression" if exp == "regression2d" else "segmentation"
            if bm.suffix.lower() != ".pt":
                raise RuntimeError(
                    f"Fine-tuning the {stg} stage needs a .pt in the '{stg.capitalize()} model "
                    "(.pt)' field (curated models ship .pt beside .h5; convert your own with "
                    "dare2d-torch/convert_to_torch.py).")
            if not bm.exists():
                raise RuntimeError(f"{stg} base model not found: {bm}")
    rn = run_name.strip() or datetime.date.today().isoformat()
    _RUN["cancel"] = False

    def _cmd(exp):
        common = ["--experiment", exp, "--test-set", str(test_set), "--run-name", rn,
                  "--epochs", str(epochs), "--steps", str(steps), "--crop", str(crop),
                  "--raw-root", str(raw_dir)]
        if train_sets.strip():
            common += ["--train-sets", train_sets.strip()]
        if finetune:                                   # fine-tune == PyTorch only; per-stage base .pt
            ft = ["--base-model", str(_base_for[exp]),
                  "--unfreeze-last", str(unfreeze_last), "--bn-mode", str(bn_mode),
                  "--ft-lr", str(ft_lr),
                  "--backbone-lr-mult", str(backbone_lr_mult), "--weight-decay", str(weight_decay),
                  "--lr-schedule", str(lr_schedule), "--warmup-epochs", str(warmup_epochs),
                  "--grad-clip", str(grad_clip), "--batch-size", str(batch_size),
                  "--augment-strength", str(augment_strength), "--patience", str(patience)]
            if not discriminative:
                ft += ["--no-discriminative"]
            if not augment:
                ft += ["--no-augment"]
            return [sys.executable, str(_TRAIN_DIR / "torch" / "finetune.py"), *common, *ft]
        if backend.startswith("PyTorch"):
            return [sys.executable, str(_TRAIN_DIR / "torch" / "train.py"), *common]
        if "WSL" in backend:
            if sys.platform != "win32":
                raise RuntimeError("The 'TensorFlow (WSL GPU)' backend needs Windows + WSL2; "
                                   "use 'PyTorch (GPU)' or 'TensorFlow (CPU)' on this OS.")
            sh = _win_to_wsl(_TRAIN_DIR / "tf" / "wsl" / "run_train.sh")
            return ["wsl", "-d", "Ubuntu", "bash", sh, *common]
        return [sys.executable, str(_TRAIN_DIR / "tf" / "train_split.py"), *common]

    # PYTHONIOENCODING=utf-8 makes the child encode stdout as UTF-8 to match the pipe's
    # encoding="utf-8" below; without it the child uses cp1252 on Windows and any non-ASCII
    # (…, —, progress glyphs) decodes to mojibake in the teed terminal log.
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", SM_FRAMEWORK="tf.keras",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

    @thread_worker
    def run():
        for exp in exps:
            if _RUN["cancel"]:
                break
            stage = exp.replace("2d", "")     # "regression" / "segmentation"
            _echo(f"\n========== {stage} stage - retraining ({backend}) ==========\n")
            # decode as UTF-8 w/ replacement: training output has non-cp1252 bytes
            # (progress bars / warnings) that the Windows default codec rejects.
            proc = subprocess.Popen(_cmd(exp), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, encoding="utf-8",
                                    errors="replace", bufsize=1,
                                    env=env, cwd=str(_api.PROJECT_ROOT))
            _RUN["proc"] = proc
            for line in proc.stdout:
                if _RUN["cancel"]:
                    proc.terminate()
                    break
                _echo(line)                   # tee: full training log -> terminal
                mph = _PHASE_RE.search(line)
                if mph:
                    # "training…" -> flip THIS stage's bar to a determinate 0% (so it stops
                    # spinning the moment training begins); prep phases stay "busy".
                    if mph.group(1).strip().lower().startswith("training"):
                        yield (stage, "pct", 0.0, None)
                    else:
                        yield (stage, "busy", "preprocessing…", None)
                    continue
                mpr = _PREP_RE.search(line)
                if mpr:
                    yield (stage, "busy", f"preprocessing {mpr.group(1)}…", None)
                    continue
                # within-stage percent (0..100) from a step heartbeat, else an epoch line;
                # the ETA is taken straight from the terminal's [step] line.
                within = eta = None
                ms = _STEP_RE.search(line)
                if ms:
                    ep, tot, k, kn = (int(g) for g in ms.groups())
                    within = ((ep - 1) * kn + k) / (tot * kn) * 100.0
                    meta = re.search(r"eta (\S+)", line)
                    eta = meta.group(1) if meta else None
                else:
                    me = _EPOCH_RE.search(line)
                    if me:
                        ep, tot = int(me.group(1)), int(me.group(2))
                        within = ep / tot * 100.0
                if within is not None:
                    yield (stage, "pct", within, eta)
            rc = proc.wait()
            if _RUN["cancel"]:
                break
            if rc != 0:
                raise RuntimeError(f"{exp} retraining failed (exit {rc}); see terminal")
        return rn

    bars = _RUN.get("bars", {})
    _stop_btn = _RUN.get("stop_btn")
    _call_btn = _RUN.get("call_button")

    def _set_running(running):
        # while a run is active: show 'Stop retraining', hide 'Start retraining'
        # (and the reverse once it ends, so you can launch the next run).
        if _stop_btn is not None:
            _stop_btn.visible = running
        if _call_btn is not None:
            _call_btn.visible = not running

    def _on_yield(v):
        stage, kind, val, eta = v
        bar = bars.get(stage)
        if bar is None:
            return
        bar.visible = True
        if kind == "busy":                      # indeterminate (animated) bar
            bar.max = 0
            bar.label = f"{stage} stage: {val}"
        else:                                   # determinate within-stage percent
            bar.max = 100
            bar.value = max(0, min(100, int(round(val))))
            bar.label = f"{stage} stage:" + (f"  eta {eta}" if eta else "")

    def _on_start():
        for stage, bar in bars.items():         # fresh run: reset every bar
            bar.visible = False
            bar.max, bar.value = 0, 0
            bar.label = f"{stage} stage:"
        if exps:                                # show the first stage busy immediately
            first = exps[0].replace("2d", "")
            bar = bars.get(first)
            if bar is not None:
                bar.visible = True
                bar.max = 0
                bar.label = f"{first} stage: starting… (importing backend)"
        _set_running(True)

    def _done(_=None):
        _RUN["proc"] = None
        if _RUN["cancel"]:
            for bar in bars.values():
                if bar.visible:
                    bar.label = bar.label.split("  eta")[0] + "  (stopped)"

    def _on_error(_e):
        _RUN["proc"] = None
        for bar in bars.values():
            if bar.visible:
                bar.label = "error (see terminal)"

    worker = run()
    worker.yielded.connect(_on_yield)
    worker.returned.connect(_done)
    worker.errored.connect(_on_error)
    worker.started.connect(_on_start)
    worker.finished.connect(lambda: _set_running(False))
    worker.start()
    return worker
