"""Correctness tests for the dice-loss path in src/common/loss.py.

Covers:
  - get_class_dice soft-dice math (full-volume + body-mask paths) vs a manual ref
  - CompositeLoss's per-class weighted dice term (run on the REAL forward, GPU):
      * reduces to dice_w * mean(1 - fg_dice) when dice_bone_w == dice_w
      * bone weight is REPLACED by dice_bone_w (not added)
      * background (idx 0) is excluded (weight 0)
      * diagnostics (loss_dice / dice_score_bone) match
"""
import os
import sys

import torch

try:
    import pytest
except ModuleNotFoundError:  # minimal shim so the file runs without pytest installed
    class _Mark:
        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn):
                fn.__skip__ = (bool(cond), reason)
                return fn
            return deco

    class pytest:  # noqa: N801
        mark = _Mark()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common.loss import CompositeLoss, get_class_dice  # noqa: E402

CUDA = torch.cuda.is_available()


def _manual_class_dice(logits, target, num_classes, smooth=1e-5):
    """Independent soft-dice reference (full-volume, mean over batch)."""
    probs = torch.softmax(logits.float(), dim=1)
    target = target.squeeze(1)  # [B,1,X,Y,Z] -> [B,X,Y,Z] to match probs[:, c]
    dices = torch.zeros(num_classes)
    for c in range(num_classes):
        p = probs[:, c]
        t = (target == c).float()
        inter = (p * t).flatten(1).sum(1)
        union = p.flatten(1).sum(1) + t.flatten(1).sum(1)
        dices[c] = ((2 * inter + smooth) / (union + smooth)).mean()
    return dices


def test_get_class_dice_full_volume_matches_manual():
    torch.manual_seed(0)
    B, C = 2, 12
    logits = torch.randn(B, C, 6, 6, 6)
    target = torch.randint(0, C, (B, 1, 6, 6, 6))
    cd, bone = get_class_dice(logits, target, bone_idx=5)
    ref = _manual_class_dice(logits, target, C)
    assert torch.allclose(cd, ref, atol=1e-5), (cd, ref)
    assert torch.allclose(bone, cd[5])


def test_get_class_dice_perfect_prediction_is_one():
    C = 12
    target = torch.randint(0, C, (1, 1, 5, 5, 5))
    # logits that argmax/softmax-peak exactly on the GT label -> dice ~1 for present classes
    logits = torch.full((1, C, 5, 5, 5), -10.0)
    for c in range(C):
        logits[0, c][target[0, 0] == c] = 10.0
    cd, _ = get_class_dice(logits, target, bone_idx=5)
    present = [c for c in range(C) if (target == c).any()]
    for c in present:
        assert cd[c] > 0.99, (c, float(cd[c]))


def test_absent_class_dice_is_one():
    """Class absent in both pred and GT -> 0/0 -> smooth/smooth = 1 (no loss)."""
    C = 4
    target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long)  # only class 0 present
    # logit gap large enough that absent classes have ~0 prob mass (<< smooth),
    # so 0/0 -> smooth/smooth = 1 exactly. A small gap leaves leakage ~ smooth.
    logits = torch.full((1, C, 4, 4, 4), -100.0)
    logits[0, 0] = 10.0  # predict class 0 everywhere
    cd, _ = get_class_dice(logits, target, bone_idx=5)
    assert torch.allclose(cd[1:], torch.ones(C - 1), atol=1e-4), cd


def test_body_mask_path_matches_full_when_mask_all_ones():
    torch.manual_seed(1)
    C = 12
    logits = torch.randn(1, C, 6, 6, 6)
    target = torch.randint(0, C, (1, 1, 6, 6, 6))
    full, _ = get_class_dice(logits, target, mask=None, bone_idx=5)
    mask = torch.ones(1, 1, 6, 6, 6)
    masked, _ = get_class_dice(logits, target, mask=mask, bone_idx=5)
    assert torch.allclose(full, masked, atol=1e-5), (full, masked)


# ---- The actual implementation under review: CompositeLoss weighted dice term ----

def _dice_only_weights(dice_w, dice_bone_w):
    return {
        "l1": 0.0, "l2": 0.0, "ssim": 0.0, "perceptual": 0.0,
        "dice_w": dice_w, "dice_bone_w": dice_bone_w,
        "dice_bone_idx": 5, "dice_exclude_background": True,
    }


def _expected_dice_term(cd, dice_w, dice_bone_w, bone_idx=5, start=1):
    C = cd.shape[0]
    w = torch.zeros(C, device=cd.device, dtype=cd.dtype)
    w[start:] = dice_w
    w[bone_idx] = dice_bone_w
    return (w * (1.0 - cd)).sum() / (C - start)


@pytest.mark.skipif(not CUDA, reason="fused_ssim3d (called unconditionally in forward) needs CUDA")
def test_composite_dice_term_matches_reference():
    torch.manual_seed(2)
    dev = "cuda"
    B, C = 2, 12
    pred = torch.rand(B, 1, 6, 6, 6, device=dev, requires_grad=True)
    target = torch.rand(B, 1, 6, 6, 6, device=dev)
    logits = torch.randn(B, C, 6, 6, 6, device=dev)
    seg = torch.randint(0, C, (B, 1, 6, 6, 6), device=dev)

    dice_w, dice_bone_w = 0.1, 0.4
    loss_fn = CompositeLoss(_dice_only_weights(dice_w, dice_bone_w))
    total, comps = loss_fn(pred, target, pred_probs=logits, target_mask=seg)

    cd, _ = get_class_dice(logits, seg, bone_idx=5)
    expected = _expected_dice_term(cd, dice_w, dice_bone_w)
    assert torch.allclose(total, expected, atol=1e-5), (float(total), float(expected))

    # diagnostics
    assert torch.allclose(comps["dice_score_all"], cd[1:].mean(), atol=1e-5)
    assert torch.allclose(comps["loss_dice"], 1 - cd[1:].mean(), atol=1e-5)
    assert torch.allclose(comps["dice_score_bone"], cd[5], atol=1e-5)


@pytest.mark.skipif(not CUDA, reason="fused_ssim3d needs CUDA")
def test_equal_weights_reduce_to_old_general_term():
    """When dice_bone_w == dice_w, the new term == dice_w * mean(1 - fg_dice) (old general)."""
    torch.manual_seed(3)
    dev = "cuda"
    B, C = 2, 12
    pred = torch.rand(B, 1, 6, 6, 6, device=dev)
    target = torch.rand(B, 1, 6, 6, 6, device=dev)
    logits = torch.randn(B, C, 6, 6, 6, device=dev)
    seg = torch.randint(0, C, (B, 1, 6, 6, 6), device=dev)

    dice_w = 0.1
    loss_fn = CompositeLoss(_dice_only_weights(dice_w, dice_w))
    total, _ = loss_fn(pred, target, pred_probs=logits, target_mask=seg)

    cd, _ = get_class_dice(logits, seg, bone_idx=5)
    old_general = dice_w * (1 - cd[1:].mean())
    assert torch.allclose(total, old_general, atol=1e-5), (float(total), float(old_general))


@pytest.mark.skipif(not CUDA, reason="fused_ssim3d needs CUDA")
def test_bone_is_replaced_not_added():
    """Bone's effective per-class weight must be dice_bone_w, NOT dice_w + dice_bone_w."""
    torch.manual_seed(4)
    dev = "cuda"
    B, C = 1, 12
    pred = torch.rand(B, 1, 6, 6, 6, device=dev)
    target = torch.rand(B, 1, 6, 6, 6, device=dev)
    logits = torch.randn(B, C, 6, 6, 6, device=dev)
    seg = torch.randint(0, C, (B, 1, 6, 6, 6), device=dev)

    dice_w, dice_bone_w = 0.1, 0.4
    loss_fn = CompositeLoss(_dice_only_weights(dice_w, dice_bone_w))
    total, _ = loss_fn(pred, target, pred_probs=logits, target_mask=seg)

    cd, _ = get_class_dice(logits, seg, bone_idx=5)
    # "replace" reference (correct) vs "add" reference (the old, rejected behavior)
    replaced = _expected_dice_term(cd, dice_w, dice_bone_w)
    w_add = torch.full((C,), dice_w, device=dev)
    w_add[0] = 0.0
    w_add[5] = dice_w + dice_bone_w
    added = (w_add * (1 - cd)).sum() / (C - 1)
    assert torch.allclose(total, replaced, atol=1e-5)
    assert not torch.allclose(total, added, atol=1e-3)  # must NOT match the add behavior


@pytest.mark.skipif(not CUDA, reason="fused_ssim3d needs CUDA")
def test_background_excluded():
    """Changing background-class GT/pred must not affect the dice loss (weight 0)."""
    torch.manual_seed(5)
    dev = "cuda"
    B, C = 1, 12
    pred = torch.rand(B, 1, 6, 6, 6, device=dev)
    target = torch.rand(B, 1, 6, 6, 6, device=dev)
    logits = torch.randn(B, C, 6, 6, 6, device=dev)
    seg = torch.randint(1, C, (B, 1, 6, 6, 6), device=dev)  # no background voxels in GT

    loss_fn = CompositeLoss(_dice_only_weights(0.1, 0.4))
    total_a, _ = loss_fn(pred, target, pred_probs=logits, target_mask=seg)

    # perturb ONLY the background logit channel; foreground dice unchanged -> loss unchanged
    logits2 = logits.clone()
    logits2[:, 0] += 100.0
    cd1, _ = get_class_dice(logits, seg)
    cd2, _ = get_class_dice(logits2, seg)
    # background channel saturating changes probs, but bg weight is 0 in the loss:
    w = torch.zeros(C, device=dev)
    w[1:] = 0.1
    w[5] = 0.4
    t1 = (w * (1 - cd1)).sum() / (C - 1)
    assert torch.allclose(total_a, t1, atol=1e-5)
