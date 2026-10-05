# %% [code] {"jupyter":{"outputs_hidden":false}}
# MarklessMan — Stage-2 SLBR export (FP32 .pth -> .onnx) + weight-only INT8 quant.
# SELF-CONTAINED: paste every cell into a Kaggle notebook and Run All.
# Export runs on GPU when visible (T4 x2); ORT parity/bench stays CPU.
# No calibration data needed.
#
# Input:  shinyunaa/marklessman-slbr (finetuned model_best.pth, epoch 10)
# Output: /kaggle/working/slbr_onnx/{slbr_256_fp32.onnx, slbr_256_int8.onnx,
#         slbr_dyn_fp32.onnx, slbr_dyn_int8.onnx} (+ parity.json, comparison.png)
#
# Method: clone bcmi/SLBR-Visible-Watermark-Removal at runtime, load the
# finetuned checkpoint via src.models (same flags as run_slbr.py), wrap the
# net + mask-blend into a single-output graph (cleaned [1,3,H,W] 0-1), export
# fixed-256 (s256 path) + dynamic-square (direct path), then apply the same
# weight-only per-channel asymmetric UINT8 as ManhwaOCR/models/quant.py
# (Conv weights -> UINT8 + DequantizeLinear(axis=0); BN/mask/compute stay FP32).
#
# NOTE: dynamic means SQUARE-only (H==W, multiple of 32 after pad). This matches
# run_slbr.py which always feeds native squares (square_crop_geo). Non-square
# HW (e.g. 320x480) fails inside refinement CFFBlocks even in torch
# (F.interpolate size reversal bug) — never occurs in the real pipeline.
#
# CONFIG — edit these, then Run All.
SLBR_CKPT_SLUG = "shinyunaa/marklessman-slbr"  # fallback download if no attachment
CKPT_HINT = "model_best"                       # file containing this substring
REPO_URL = "https://github.com/bcmi/SLBR-Visible-Watermark-Removal.git"
REPO_DIR = "/kaggle/working/slbr_repo"
OUT_DIR = "/kaggle/working/slbr_onnx"          # everything here is kept as output
OPSET = 18                                     # exporter floor is 18 (Resize has no v17 adapter); ORT 1.24-safe

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Deps. Kaggle already ships torch 2.x; we add only the small extras.
# onnx/onnxruntime for export+parity+quant (pure-python graph rewrite, CPU).
import subprocess
import sys
from pathlib import Path


def ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name],
                       check=True)


for _imp, _pip in (("progress", "progress"), ("tensorboardX", "tensorboardX"),
                   ("onnx", "onnx"), ("onnxruntime", "onnxruntime"),
                   ("onnxscript", "onnxscript")):
    ensure_pkg(_imp, _pip)

import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
print("Export uses GPU when visible; ORT parity/bench stays CPU.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Resolve finetuned ckpt: prefer attached input under /kaggle/input (no download).
# Fallback to kagglehub download. Accepts model_best.pth OR model_best.pth.tar
# (torch.load handles both; the .zip container IS the checkpoint — see below).
import shutil

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)


def resolve_ckpt() -> Path:
    hits = [p for p in Path("/kaggle/input").rglob("*")
            if p.is_file() and CKPT_HINT in p.name
            and p.suffix in (".pth", ".tar")
            or (p.is_file() and CKPT_HINT in p.name and p.name.endswith(".pth.zip"))]
    # rglob above is loose; tighten: name contains hint AND endswith .pth/.pth.tar/.pth.zip
    hits = [p for p in Path("/kaggle/input").rglob("*")
            if p.is_file() and CKPT_HINT in p.name
            and (p.name.endswith(".pth") or p.name.endswith(".pth.tar")
                 or p.name.endswith(".pth.zip"))]
    if hits:
        print(f"using attached input: {hits[0]}")
        return hits[0]
    print("nothing under /kaggle/input — downloading via kagglehub")
    import kagglehub
    _ds = Path(kagglehub.dataset_download(SLBR_CKPT_SLUG))
    cands = sorted(p for p in _ds.rglob("*")
                   if p.is_file() and CKPT_HINT in p.name
                   and (p.name.endswith(".pth") or p.name.endswith(".pth.tar")
                        or p.name.endswith(".pth.zip")))
    assert cands, f"no {CKPT_HINT}* checkpoint found — check SLBR_CKPT_SLUG / attachment"
    return cands[0]


_src = resolve_ckpt()
# NOTE: weights/model_best.pth.zip IS the torch checkpoint itself (zip-serialization),
# not an outer container — torch.load reads it directly. Normalize to work_ckpt
# so the rest of the notebook never cares about the extension.
work_ckpt = out_dir / "model_best.pth"
if _src.resolve() != work_ckpt.resolve():
    shutil.copy(_src, work_ckpt)
print(f"ckpt -> {work_ckpt} ({work_ckpt.stat().st_size / 1e6:.1f} MB)")

_ck = torch.load(work_ckpt.as_posix(), map_location="cpu", weights_only=False)
print(f"ckpt keys: {list(_ck.keys())}  epoch={_ck.get('epoch')}  "
      f"nets={_ck.get('nets')}  best={_ck.get('best_acc')}  "
      f"state_dict={len(_ck.get('state_dict', {}))}")
del _ck

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Clone SLBR repo (runtime dependency) + in-process skimage shim (repo imports
# removed compare_psnr/compare_ssim).
# Export uses GPU when visible (e.g. T4 x2): SLBR_CPU=0 puts the model on
# cuda:0, and the multi_gpu() no-op patch below keeps it single-GPU so keys
# stay encoder.* (DataParallel would rename to encoder.module.* and break
# the strict load of this single-GPU checkpoint). Falls back to CPU with no GPU.
import os

os.environ["SLBR_CPU"] = "0"  # cuda when available, cpu otherwise

if not (Path(REPO_DIR) / "train.py").exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print("repo cached")
print(sorted(p.name for p in Path(REPO_DIR).iterdir()))

import skimage.measure as _skm
import skimage.metrics as _skmet

for _n, _f in (("compare_psnr", _skmet.peak_signal_noise_ratio),
               ("compare_ssim", _skmet.structural_similarity)):
    if not hasattr(_skm, _n):
        setattr(_skm, _n, _f)

# CPU-only Kaggle has 0 GPUs but the finetuned ckpt holds CUDA storages, and
# the upstream resume() calls bare torch.load(resume_path) (no map_location,
# and torch>=2.6 defaults weights_only=True which rejects optimizer state).
# Patch on disk so every import/subprocess picks it up.
_bp = Path(REPO_DIR) / "src" / "models" / "BasicModel.py"
_bt = _bp.read_text()
_old = "current_checkpoint = torch.load(resume_path)"
_new = "current_checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)"
if _new in _bt:
    print("resume map_location ok (cached)")
elif _old in _bt:
    _bp.write_text(_bt.replace(_old, _new))
    print("patched BasicModel.resume -> torch.load(map_location='cpu', weights_only=False)")
else:
    raise AssertionError("BasicModel.resume pattern changed — inspect src/models/BasicModel.py")

# Keep the model single-GPU even with 2 GPUs visible: DataParallel would
# rename keys to encoder.module.* and break the strict load of this
# single-GPU checkpoint. Model still sits on cuda:0 when available.
_rp = Path(REPO_DIR) / "src" / "networks" / "resunet.py"
_rt = _rp.read_text()
_r_old = """    def multi_gpu(self):
        self.encoder = nn.DataParallel(self.encoder, device_ids=range(torch.cuda.device_count()))
        self.shared_decoder = nn.DataParallel(self.shared_decoder, device_ids=range(torch.cuda.device_count()))
        self.coarse_decoder = nn.DataParallel(self.coarse_decoder, device_ids=range(torch.cuda.device_count()))
        if self.refinement is not None:
            self.refinement = nn.DataParallel(self.refinement, device_ids=range(torch.cuda.device_count()))
        return"""
_r_new = """    def multi_gpu(self):
        # Export: no-op. Single-GPU (cuda:0 when available) keeps encoder.*
        # keys matching the checkpoint.
        return"""
if "Export: no-op" in _rt:
    print("multi_gpu no-op ok (cached)")
elif _r_old in _rt:
    _rp.write_text(_rt.replace(_r_old, _r_new))
    print("patched resunet.multi_gpu -> no-op (single-GPU export)")
else:
    raise AssertionError("resunet.multi_gpu pattern changed — inspect src/networks/resunet.py")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Load finetuned SLBR via src.models (SAME flags as run_slbr.py).
# Wraps net + mask-blend into a single-output module for ONNX:
#   cleaned = denorm(im*mask + norm(x)*(1-mask)), [1,3,H,W] 0-1 float.
import argparse

import torch.nn as nn
from options import Options
import src.models as _models

ORT_CPU = ["CPUExecutionProvider"]  # never CUDA — this notebook is CPU-only


def build_machine(ckpt: Path):
    p = Options().init(argparse.ArgumentParser())
    args, _ = p.parse_known_args(args=[
        "--nets", "slbr", "--models", "slbr",
        "--mask_mode", "res", "--k_center", "2", "--k_refine", "3",
        "--k_skip_stage", "3", "--use_refine",
        "--input-size", "256", "--crop_size", "256",
        "--lr", "1e-4", "--beta1", "0.5", "--beta2", "0.999",
        "--checkpoint", (out_dir / "ckpt").as_posix(), "--name", "export",
        "--evaluate", "--resume", ckpt.as_posix()])
    machine = _models.__dict__[args.models](datasets=(None, None), args=args)
    machine.model.eval()
    return machine


class SlbrClean(nn.Module):
    """Single-input single-output export wrapper (0-1 in, 0-1 out)."""

    def __init__(self, machine):
        super().__init__()
        self.net = machine.model
        self._norm = machine.norm
        self._denorm = machine.denorm

    def forward(self, x):
        outputs = self.net(self._norm(x))
        imoutput, immask_all, _ = outputs
        im = imoutput[0]
        mk = immask_all[0]
        return self._denorm(im * mk + self._norm(x) * (1 - mk))


machine = build_machine(work_ckpt)
wrapper = SlbrClean(machine).eval()
print(f"machine device={machine.device}  params on "
      f"{'GPU' if machine.device.type == 'cuda' else 'CPU'}")
with torch.no_grad():
    _probe = wrapper(torch.rand(1, 3, 256, 256).to(machine.device))
    print(f"probe out: {tuple(_probe.shape)} range "
          f"[{float(_probe.min()):.4f},{float(_probe.max()):.4f}]")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Export FP32: fixed-256 (s256 path) + dynamic-square (direct path, H==W only).
# Opset 17, classic exporter, constant folding. Checker + ORT CPU session after.
import onnx

FP32_256 = out_dir / "slbr_256_fp32.onnx"
FP32_DYN = out_dir / "slbr_dyn_fp32.onnx"

with torch.no_grad():
    _d = torch.rand(1, 3, 256, 256).to(machine.device)
    torch.onnx.export(wrapper, _d, FP32_256.as_posix(),
                      input_names=["input"], output_names=["cleaned"],
                      opset_version=OPSET, do_constant_folding=True,
                      dynamic_axes=None)
    print(f"fixed-256 -> {FP32_256} ({FP32_256.stat().st_size / 1e6:.1f} MB)")
    torch.onnx.export(wrapper, _d, FP32_DYN.as_posix(),
                      input_names=["input"], output_names=["cleaned"],
                      opset_version=OPSET, do_constant_folding=True,
                      dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                                    "cleaned": {0: "N", 2: "H", 3: "W"}})
    print(f"dynamic-square -> {FP32_DYN} ({FP32_DYN.stat().st_size / 1e6:.1f} MB)")

for _p in (FP32_256, FP32_DYN):
    _m = onnx.load(_p.as_posix())
    onnx.checker.check_model(_m)
    print(f"{_p.name}: checker OK, {len(_m.graph.node)} nodes, "
          f"opset {[o.version for o in _m.opset_import]}")
    for _i in _m.graph.input:
        _dims = [d.dim_value if d.dim_value else d.dim_param
                 for d in _i.type.tensor_type.shape.dim]
        print(f"  input { _i.name} {_dims}")

# Dynamo writes the 85MB weights next door (*.onnx.data, static-link style).
# Consolidate to single-file artifacts like the local export.
for _p in (FP32_256, FP32_DYN):
    _dp = Path(_p.as_posix() + ".data")
    if _dp.exists():
        print(f"consolidating {_dp.name} ({_dp.stat().st_size / 1e6:.1f} MB) ...")
        _m = onnx.load(_p.as_posix(), load_external_data=True)
        onnx.save(_m, _p.as_posix())
        try:
            _dp.unlink()
        except Exception:
            pass
        print(f"  {_p.name} now single file ({_p.stat().st_size / 1e6:.1f} MB)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Parity: torch wrapper vs ORT FP32 on synthetic 256 (CPU only).
# Dynamic variant checked on square sizes only (256/384/512 — never rectangular).
import numpy as np
import onnxruntime as ort

with torch.no_grad():
    _x = torch.rand(1, 3, 256, 256).to(machine.device)
    _ref = wrapper(_x).cpu().numpy()


def _ort_run(path: Path, arr: np.ndarray) -> np.ndarray:
    sess = ort.InferenceSession(path.as_posix(), providers=ORT_CPU)
    return sess.run(None, {sess.get_inputs()[0].name: arr})[0]


_o256 = _ort_run(FP32_256, _x.cpu().numpy())
print(f"[256] torch-vs-ort max={abs(_o256 - _ref).max():.2e} "
      f"mean={abs(_o256 - _ref).mean():.2e}")

for _s in (256, 384, 512):
    _a = np.random.rand(1, 3, _s, _s).astype(np.float32)
    _o = _ort_run(FP32_DYN, _a)
    print(f"[dyn {_s}x{_s}] ort out {tuple(_o.shape)} mean={_o.mean():.4f}")
assert _o256.shape == (1, 3, 256, 256)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Weight-only per-channel asymmetric UINT8 — verbatim port of
# ManhwaOCR/models/quant.py::quantize_weights_per_channel_uint8.
# Only Conv *weights* are touched (small, well-behaved floats). Each output
# channel c gets scale[c]=(max-min)/255, zp[c]=clip(round(-min/scale),0,255),
# stored UINT8 + DequantizeLinear(axis=0). BN/mask/compute stay FP32.
# No activation quant, no calibration data, deterministic.
from onnx import helper, numpy_helper, TensorProto, shape_inference


def quantize_weights_per_channel_uint8(model):
    init_by_name = {init.name: init for init in model.graph.initializer}
    conv_nodes = [n for n in model.graph.node if n.op_type == "Conv"]
    print(f"Found {len(conv_nodes)} Conv nodes to quantize (weight-only)")
    new_scale_inits, new_zp_inits, dq_nodes = [], [], []
    quantized_count, skipped, shared = 0, 0, 0
    _quantized = {}  # weight_name -> dequant output (SLBR shares some inits)
    for conv in conv_nodes:
        if len(conv.input) < 2:
            skipped += 1
            continue
        weight_name = conv.input[1]
        if weight_name in _quantized:
            # Shared weight: UINT8 + DQ already exist — rewire to the same
            # dequant. (Without this the Conv keeps the raw UINT8 input,
            # which ORT rejects: "tensor(uint8) of Conv is invalid".)
            conv.input[1] = _quantized[weight_name]
            quantized_count += 1
            shared += 1
            continue
        if weight_name not in init_by_name:
            skipped += 1
            continue
        init = init_by_name[weight_name]
        if init.data_type != TensorProto.FLOAT:
            skipped += 1
            continue
        w_arr = numpy_helper.to_array(init)
        if w_arr.ndim < 1:
            skipped += 1
            continue
        oc = w_arr.shape[0]
        w_reshaped = w_arr.reshape(oc, -1)
        scales = np.empty(oc, dtype=np.float32)
        zps = np.empty(oc, dtype=np.uint8)
        q_arr = np.empty_like(w_arr, dtype=np.uint8)
        for c in range(oc):
            w_c = w_reshaped[c]
            min_c, max_c = float(w_c.min()), float(w_c.max())
            scale_c = (max_c - min_c) / 255.0
            if scale_c < 1e-8:
                scale_c = 1e-8
            zp_c = int(np.clip(int(np.round(-min_c / scale_c)), 0, 255))
            scales[c] = scale_c
            zps[c] = zp_c
            q_arr[c] = np.clip(np.round(w_arr[c] / scale_c + zp_c), 0, 255).astype(np.uint8)
        q_init = numpy_helper.from_array(q_arr, name=weight_name)
        for idx, it in enumerate(model.graph.initializer):
            if it.name == weight_name:
                model.graph.initializer[idx].CopyFrom(q_init)
                break
        scale_name, zp_name = weight_name + "__scale", weight_name + "__zero_point"
        dequant_output = weight_name + "__dequant"
        new_scale_inits.append(numpy_helper.from_array(scales, name=scale_name))
        new_zp_inits.append(numpy_helper.from_array(zps, name=zp_name))
        dq_nodes.append(helper.make_node(
            "DequantizeLinear", inputs=[weight_name, scale_name, zp_name],
            outputs=[dequant_output], name=weight_name + "__dq", axis=0))
        conv.input[1] = dequant_output
        _quantized[weight_name] = dequant_output
        quantized_count += 1
    model.graph.initializer.extend(new_scale_inits)
    model.graph.initializer.extend(new_zp_inits)
    existing_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(dq_nodes)
    model.graph.node.extend(existing_nodes)
    return model, {"quantized": quantized_count, "skipped": skipped,
                   "shared": shared, "dq_nodes": len(dq_nodes)}


INT8_256 = out_dir / "slbr_256_int8.onnx"
INT8_DYN = out_dir / "slbr_dyn_int8.onnx"

for _fp32, _int8 in ((FP32_256, INT8_256), (FP32_DYN, INT8_DYN)):
    print(f"Quantizing {_fp32.name} -> {_int8.name} ...")
    _qm = onnx.load(_fp32.as_posix())
    _qm, _st = quantize_weights_per_channel_uint8(_qm)
    print(f"  quantized {_st['quantized']} Conv ({_st.get('shared', 0)} shared rewired), "
          f"skipped {_st['skipped']}, DQ={_st['dq_nodes']}")
    onnx.checker.check_model(_qm)
    print("  checker: OK")
    # ORT-load gate: fail fast here (not in the quality cell) if a Conv still
    # eats a raw UINT8 weight (the shared-weight bug class).
    _sess = ort.InferenceSession(_qm.SerializeToString(), providers=ORT_CPU)
    print(f"  ORT load OK: {[i.name + str(list(i.shape)) for i in _sess.get_inputs()]}")
    del _sess
    try:
        shape_inference.infer_shapes(_qm)
        print("  shape_inference: OK")
    except Exception as _e:
        print(f"  shape_inference warning: {_e}")
    onnx.save(_qm, _int8.as_posix())
    print(f"  saved {_int8.name} ({_int8.stat().st_size / 1e6:.2f} MB)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Size report + weight-dtype verification.
from collections import Counter

_fp32_sz = FP32_256.stat().st_size / 1e6
_i8_sz = INT8_256.stat().st_size / 1e6
print("=" * 58)
print(f"FP32 fixed-256      : {_fp32_sz:7.2f} MB  [slbr_256_fp32.onnx]")
print(f"INT8 fixed-256      : {_i8_sz:7.2f} MB  [slbr_256_int8.onnx]  "
      f"{(1 - _i8_sz / _fp32_sz) * 100:5.1f}% smaller  {_fp32_sz / _i8_sz:.2f}x")
print(f"FP32 dynamic-square : {FP32_DYN.stat().st_size / 1e6:7.2f} MB")
print(f"INT8 dynamic-square : {INT8_DYN.stat().st_size / 1e6:7.2f} MB")
print("=" * 58)

_m = onnx.load(INT8_256.as_posix())
print(f"DQ nodes: {sum(1 for n in _m.graph.node if n.op_type == 'DequantizeLinear')} "
      f"(one per Conv)  ops={dict(Counter(n.op_type for n in _m.graph.node))}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Quality check FP32 vs INT8 (CPU): synthetic 256 + optional real test crops.
# If marklessman-clean (native CLWD test/) is attached, samples 4 real test
# crops; else synthetic.
import cv2
import glob as _glob

CALIB = None
for _cand in [p for p in Path("/kaggle/input").rglob("Watermarked_image") if p.is_dir()
              and p.parent.name == "test"]:
    CALIB = _cand.parent
    break
print(f"test crops source: {CALIB if CALIB else '(synthetic only)'}")


def _fp32_int8_diff(arr: np.ndarray, fp32_path: Path, int8_path: Path):
    _f = _ort_run(fp32_path, arr)[0]
    _q = _ort_run(int8_path, arr)[0]
    _d = np.abs(_f.astype(np.float32) - _q.astype(np.float32)) * 255.0
    return (float(_d.max()), float(_d.mean()),
            int((_d.max(axis=0) > 50).sum()), _f, _q)


_tests = []
if CALIB is not None:
    _ids = sorted((CALIB / "Watermarked_image").glob("*.jpg"))[:4]
    for _jp in _ids:
        _cid = _jp.stem
        _im = cv2.imread(_jp.as_posix())
        if _im is None:
            continue
        _im = cv2.resize(_im, (256, 256))
        _t = cv2.cvtColor(_im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        _tests.append((f"test:{_cid}", _t))
else:
    _rng = np.random.default_rng(7)
    for _k in range(2):
        _tests.append((f"synth{_k}", _rng.random((1, 3, 256, 256)).astype(np.float32)))

for _name, _arr in _tests:
    _mx, _mn, _bad, _, _ = _fp32_int8_diff(_arr, FP32_256, INT8_256)
    print(f"[{_name}] INT8 vs FP32: max={_mx:.1f}/255 mean={_mn:.2f} px>50:{_bad}")

# Save one side-by-side for the eye test (first test only).
import matplotlib.pyplot as plt

_nm, _ta = _tests[0]
_, _, _, _of, _oq = _fp32_int8_diff(_ta, FP32_256, INT8_256)
_fig, _ax = plt.subplots(1, 3, figsize=(12, 4))
_ax[0].imshow((_ta[0].transpose(1, 2, 0) * 255).astype(np.uint8))
_ax[0].set_title(f"input {_nm}")
_ax[1].imshow((_of.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
_ax[1].set_title("FP32")
_ax[2].imshow((_oq.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
_ax[2].set_title("INT8 w-only")
for _a in _ax:
    _a.axis("off")
_fig.tight_layout()
_fig.savefig((out_dir / "comparison.png").as_posix(), dpi=120)
print(f"saved {(out_dir / 'comparison.png').as_posix()}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Latency bench, CPU only (fixed-256). Weight-only INT8 is memory-bound:
# expect size win for sure, modest speed win on x86 (ORT dequant overhead).
import time


def bench(path: Path, arr: np.ndarray, warmup=5, runs=20) -> float:
    sess = ort.InferenceSession(path.as_posix(), providers=ORT_CPU)
    _in = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {_in: arr})
    _t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {_in: arr})
    return (time.perf_counter() - _t0) / runs


_probe = np.random.rand(1, 3, 256, 256).astype(np.float32)
_lat_f = bench(FP32_256, _probe)
_lat_q = bench(INT8_256, _probe)
print(f"FP32 256 avg (CPU): {_lat_f * 1e3:.1f} ms")
print(f"INT8 256 avg (CPU): {_lat_q * 1e3:.1f} ms  ({_lat_f / _lat_q:.2f}x)")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Done. Everything under /kaggle/working is kept via "Save & Run All".
# Local use (mirrors run_slbr.py, but ONNX CPU — no torch needed at infer):
#   import onnxruntime as ort, cv2, numpy as np
#   sess = ort.InferenceSession("slbr_256_int8.onnx", providers=["CPUExecutionProvider"])
#   # s256 path: resize square crop -> 256, /255.0, NCHW, run, cubic back up
#   # direct path: use slbr_dyn_int8.onnx on SQUARE crops only (H==W, pad to 32)
import json

snap = {
    "ckpt": work_ckpt.as_posix(),
    "opset": OPSET,
    "fp32_256_mb": round(FP32_256.stat().st_size / 1e6, 2),
    "int8_256_mb": round(INT8_256.stat().st_size / 1e6, 2),
    "fp32_dyn_mb": round(FP32_DYN.stat().st_size / 1e6, 2),
    "int8_dyn_mb": round(INT8_DYN.stat().st_size / 1e6, 2),
    "providers": ORT_CPU,
}
(out_dir / "parity.json").write_text(json.dumps(snap, indent=2))
print(json.dumps(snap, indent=2))
print(f"artifacts -> {out_dir}")
for p in sorted(out_dir.iterdir()):
    if p.is_file():
        print(f"  {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
