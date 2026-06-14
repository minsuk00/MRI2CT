"""Verify compile_mode="model" compiles EVERY network individually, and that the
compiled training step actually runs.

Networks expected compiled in "model" mode:
  - amix : feature extractor (v1_4) + translator + teacher + perceptual extractor
  - unet : translator + teacher + perceptual extractor

In "none"/None mode NOTHING should be compiled; in "full" mode the whole step is
compiled and the sub-models stay raw.

Fast by design: builds models via _setup_models()/_setup_loss() directly (bypasses
data discovery + wandb via __new__), then runs one synthetic step.

Usage:
    cd /home/minsukc/MRI2CT
    micromamba run -n mrct python tests/test_compile_mode.py
"""
import copy
import os
import sys
import traceback

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

from common.config import DEFAULT_CONFIG, Config  # noqa: E402

results = {}


def check(name, cond, detail=""):
    ok = bool(cond)
    results[name] = ok
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}" + (f": {detail}" if detail else ""))
    return ok


def is_compiled(m):
    """True iff m is a torch.compile OptimizedModule wrapper."""
    return m is not None and (hasattr(m, "_orig_mod") or m.__class__.__name__ == "OptimizedModule")


def _make(cls, overrides):
    """Build a trainer instance WITHOUT running __init__ (no data/wandb)."""
    cfg = Config({**copy.deepcopy(DEFAULT_CONFIG), **overrides})
    t = cls.__new__(cls)
    t.cfg = cfg
    t.device = torch.device("cuda")
    t.prefix = "test"
    return t


def build_amix(compile_mode, perceptual_w=0.0):
    from amix.trainer import Trainer
    t = _make(Trainer, {
        "compile_mode": compile_mode,
        "anatomix_weights": "v1_4",
        "dice_w": 0.1, "validate_dice": True,
        "perceptual_w": perceptual_w,
        "finetune_feat_extractor": False,
    })
    t._setup_models()
    t._setup_loss()
    return t


def build_unet(compile_mode, perceptual_w=0.0):
    from unet_baseline.train import BaselineTrainer
    t = _make(BaselineTrainer, {
        "compile_mode": compile_mode,
        "dice_w": 0.1, "validate_dice": True,
        "perceptual_w": perceptual_w,
        "input_nc": 1, "output_nc": 1, "ngf": 16, "num_downs": 4, "norm": "batch",
        "model_type": "unet_baseline",
    })
    t._setup_models()
    t._setup_loss()
    return t


# ── compiled-ness wiring (instant: torch.compile is lazy) ──────────────────
def test_amix_model_compiles_all():
    print("\n[amix / compile_mode='model'] all networks compiled")
    t = build_amix("model", perceptual_w=0.1)
    check("amix model: feature extractor compiled", is_compiled(t.feat_extractor))
    check("amix model: translator compiled", is_compiled(t.model))
    check("amix model: teacher compiled", is_compiled(t.teacher_model))
    check("amix model: perceptual extractor compiled", is_compiled(t.loss_fn.perceptual.extractor))
    check("amix model: step NOT separately compiled (runs eager)", not is_compiled(t.train_step))


def test_amix_none_compiles_nothing():
    print("\n[amix / compile_mode=None] nothing compiled")
    t = build_amix(None, perceptual_w=0.1)
    check("amix none: feature extractor NOT compiled", not is_compiled(t.feat_extractor))
    check("amix none: translator NOT compiled", not is_compiled(t.model))
    check("amix none: teacher NOT compiled", not is_compiled(t.teacher_model))
    check("amix none: perceptual extractor NOT compiled", not is_compiled(t.loss_fn.perceptual.extractor))


def test_unet_model_compiles_all():
    print("\n[unet / compile_mode='model'] all networks compiled")
    t = build_unet("model", perceptual_w=0.1)
    check("unet model: translator compiled", is_compiled(t.model))
    check("unet model: teacher compiled", is_compiled(t.teacher_model))
    check("unet model: perceptual extractor compiled", is_compiled(t.loss_fn.perceptual.extractor))


def test_unet_none_compiles_nothing():
    print("\n[unet / compile_mode=None] nothing compiled")
    t = build_unet(None, perceptual_w=0.1)
    check("unet none: translator NOT compiled", not is_compiled(t.model))
    check("unet none: teacher NOT compiled", not is_compiled(t.teacher_model))
    check("unet none: perceptual extractor NOT compiled", not is_compiled(t.loss_fn.perceptual.extractor))


# ── functional: the compiled step actually runs ────────────────────────────
def _run_step(t, label, p=64):
    """One synthetic train_step; assert finite loss + translator gradients exist."""
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        mri = torch.rand(1, 1, p, p, p, device="cuda", generator=g)
        ct = torch.rand(1, 1, p, p, p, device="cuda", generator=g)
        seg = torch.randint(0, 12, (1, 1, p, p, p), device="cuda", generator=g)
        out = t.train_step(mri, ct, seg)
        loss = out[1]
        check(f"{label}: step runs, loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
        raw = getattr(t.model, "_orig_mod", t.model)
        has_grad = any(p_.grad is not None for p_ in raw.parameters())
        check(f"{label}: translator received gradients", has_grad)
    except Exception as e:
        check(f"{label}: step runs without error", False, str(e)[:200])
        traceback.print_exc()


def test_amix_step_model():
    print("\n[amix / compile_mode='model'] functional step (triggers real compile)")
    t = build_amix("model", perceptual_w=0.1)
    _run_step(t, "amix model step")


def test_unet_step_model():
    print("\n[unet / compile_mode='model'] functional step (triggers real compile)")
    t = build_unet("model", perceptual_w=0.1)
    _run_step(t, "unet model step")


def main():
    if not torch.cuda.is_available():
        print("CUDA required for this test."); sys.exit(1)
    test_amix_model_compiles_all()
    test_amix_none_compiles_nothing()
    test_unet_model_compiles_all()
    test_unet_none_compiles_nothing()
    test_amix_step_model()
    test_unet_step_model()

    print("\n" + "=" * 56 + "\nSUMMARY")
    passed = sum(v for v in results.values())
    failed = sum(1 for v in results.values() if not v)
    for n, v in results.items():
        print(f"  {'✅' if v else '❌'}  {n}")
    print(f"\nPassed: {passed} | Failed: {failed}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
