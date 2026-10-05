# %% [code] {"jupyter":{"outputs_hidden":false}}
# MarklessMan — Stage-1 YOLO26n export (.pt -> .onnx) on Kaggle. SELF-CONTAINED:
# paste every cell into a Kaggle notebook and Run All. Exports nothing locally.
#
# Input:  shinyunaa/marklessman-det-model (contains best.pt, YOLO26n watermark det)
# Output: /kaggle/working/onnx/best.onnx (+ best.pt copy + onnx_parity.json)
#
# CONFIG — edit these, then Run All.
WEIGHTS_SLUG = "shinyunaa/marklessman-det-model"  # fallback download if no attachment
WEIGHTS_FILE = "best.pt"                          # filename inside the dataset
IMGSZ = 640                                       # must match training imgsz
OPSET = None                                      # None -> latest supported; pin e.g. 17 for older runtimes
SIMPLIFY = True                                   # onnxslim graph simplify (default True)
DYNAMIC = False                                   # static 640x640 export (recommended for deployment)
CONF_PARITY = 0.25                                # conf threshold for torch-vs-onnx parity check
OUT_DIR = "/kaggle/working/onnx"                  # everything here is kept as notebook output

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Deps. YOLO26 needs a recent ultralytics; onnx/onnxruntime for export+parity,
# onnxslim backs simplify=True.
import subprocess
import sys
from pathlib import Path


def ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name],
                       check=True)


for _imp, _pip in (("ultralytics", "ultralytics"), ("onnx", "onnx"),
                   ("onnxruntime", "onnxruntime"), ("onnxslim", "onnxslim")):
    ensure_pkg(_imp, _pip)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Resolve best.pt: prefer the attached dataset under /kaggle/input (no download).
# Fallback to kagglehub download only if nothing is attached.
import shutil

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)


def resolve_weights() -> Path:
    hits = [p for p in Path("/kaggle/input").rglob(WEIGHTS_FILE) if p.is_file()]
    if hits:
        print(f"using attached input: {hits[0]}")
        return hits[0]
    print("nothing under /kaggle/input — downloading via kagglehub")
    import kagglehub
    _ds = Path(kagglehub.dataset_download(WEIGHTS_SLUG))
    cands = sorted(p for p in _ds.rglob(WEIGHTS_FILE) if p.is_file())
    assert cands, f"no {WEIGHTS_FILE} found — check WEIGHTS_SLUG / attachment"
    return cands[0]


_src = resolve_weights()
work_pt = out_dir / WEIGHTS_FILE
if _src.resolve() != work_pt.resolve():
    shutil.copy(_src, work_pt)
print(f"weights -> {work_pt} ({work_pt.stat().st_size / 1e6:.1f} MB)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Export best.pt -> best.onnx. Static 640, FP32, simplify.
# Ultralytics writes best.onnx next to best.pt, so exporting the /kaggle/working
# copy keeps the artifact in OUT_DIR directly.
import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"gpus={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    print(f"  gpu0: {torch.cuda.get_device_name(0)}")
# NOTE: export runs fine on CPU; GPU only speeds it up. No assert here.

from ultralytics import YOLO

model = YOLO(work_pt.as_posix())
print(f"loaded: task={model.task} names={model.names}")
export_kwargs = dict(format="onnx", imgsz=IMGSZ, simplify=SIMPLIFY,
                     dynamic=DYNAMIC, verbose=True)
if OPSET is not None:
    export_kwargs["opset"] = OPSET
onnx_path = Path(model.export(**export_kwargs))
print(f"exported -> {onnx_path}")

# Normalize location: keep best.onnx inside OUT_DIR even if exporter wrote elsewhere.
want = out_dir / (work_pt.stem + ".onnx")
if onnx_path.resolve() != want.resolve():
    shutil.copy(onnx_path, want)
    onnx_path = want
assert onnx_path.exists(), f"missing {onnx_path}"
print(f"onnx: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Verify the ONNX graph: checker + inputs/outputs + ultralytics metadata.
import onnx

m = onnx.load(onnx_path.as_posix())
onnx.checker.check_model(m)
print("onnx.checker: OK")
for i in m.graph.input:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
    print(f"  input: {i.name} {dims}")
for o in m.graph.output:
    dims = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
    print(f"  output: {o.name} {dims}")
print(f"  opset: {[o.version for o in m.opset_import]}")
for p in m.metadata_props:
    print(f"  meta {p.key}={p.value}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Parity check: torch .pt vs exported .onnx on one synthetic 640x640 image.
# No marklessman-v1 input needed — dummy image only checks the graphs run and
# agree, not detection quality.
import json
import time

import numpy as np
from PIL import Image

dummy = (np.random.default_rng(7).integers(0, 255, (IMGSZ, IMGSZ, 3))).astype(np.uint8)
dummy[200:400, 200:400] = 128  # flat gray square: deterministic content
probe = out_dir / "parity_probe.jpg"
Image.fromarray(dummy).save(probe, quality=92)
print(f"probe -> {probe}")

pt_model = YOLO(work_pt.as_posix())
ort_model = YOLO(onnx_path.as_posix())  # ultralytics wraps onnxruntime here

t0 = time.perf_counter()
r_pt = pt_model.predict(probe.as_posix(), conf=CONF_PARITY, verbose=False)[0]
t_pt = time.perf_counter() - t0

t0 = time.perf_counter()
r_ort = ort_model.predict(probe.as_posix(), conf=CONF_PARITY, verbose=False)[0]
t_ort = time.perf_counter() - t0

b_pt = r_pt.boxes.xyxy.cpu().numpy() if r_pt.boxes is not None else np.zeros((0, 4))
b_ort = r_ort.boxes.xyxy.cpu().numpy() if r_ort.boxes is not None else np.zeros((0, 4))
print(f"pt boxes: {len(b_pt)} ({t_pt * 1e3:.0f} ms)  onnx boxes: {len(b_ort)} ({t_ort * 1e3:.0f} ms)")

snap = {
    "weights": work_pt.as_posix(),
    "onnx": onnx_path.as_posix(),
    "pt_size_mb": round(work_pt.stat().st_size / 1e6, 2),
    "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
    "imgsz": IMGSZ, "opset": OPSET, "simplify": SIMPLIFY, "dynamic": DYNAMIC,
    "conf_parity": CONF_PARITY,
    "pt_boxes": int(len(b_pt)), "onnx_boxes": int(len(b_ort)),
    "pt_ms": round(t_pt * 1e3, 1), "onnx_ms": round(t_ort * 1e3, 1),
}
(out_dir / "onnx_parity.json").write_text(json.dumps(snap, indent=2))
print(json.dumps(snap, indent=2))

# Raw onnxruntime session: proves the file loads outside ultralytics too.
import onnxruntime as ort

sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print(f"ort providers: {sess.get_providers()}  input: {inp.name} {inp.shape} {inp.type}")
arr = np.array(Image.open(probe).convert("RGB").resize((IMGSZ, IMGSZ)),
               dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
outs = sess.run(None, {inp.name: arr})
print(f"ort raw outputs: {[o.shape for o in outs]}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Done. Everything under /kaggle/working is kept via "Save & Run All".
# Use the artifact like this (also works for split_and_detect.py / inference.py):
#   from ultralytics import YOLO
#   model = YOLO("/kaggle/working/onnx/best.onnx")  # or ./weights/best.onnx locally
#   r = model.predict("page.jpg", conf=0.25, verbose=False)[0]
print(f"artifacts -> {out_dir}")
for p in sorted(out_dir.iterdir()):
    print(f"  {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
