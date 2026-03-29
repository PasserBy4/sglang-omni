# SPDX-License-Identifier: Apache-2.0
"""Degrade the semantic codebook in codec.pth via precision round-trip (FP8 or INT4).

This script is used to validate small-sample representativeness: a benchmark
run on the degraded model should show higher WER/CER, and we check whether
a small sample (e.g. N=10) detects the same degradation as the full dataset.

Usage
-----
# Step 1 – inspect codec.pth key names (no output written):
    python scripts/degrade_codec.py --model-path fishaudio/s2-pro --inspect

# Step 2 – create degraded checkpoint in /tmp/s2pro-degraded:
    python scripts/degrade_codec.py \
        --model-path fishaudio/s2-pro \
        --output-dir /tmp/s2pro-degraded

# Step 3 – start the degraded server on a separate port:
    python -m sglang_omni.cli.cli serve \
        --model-path /tmp/s2pro-degraded \
        --config examples/configs/s2pro_tts.yaml \
        --port 8001
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

def fp8_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    """FP8 (e4m3fn) round-trip: BF16 → FP8 → BF16.

    torch.float8_e4m3fn: 4 exponent bits, 3 mantissa bits, max=448.
    Typical relative error per element: ~6 % (vs ~14 % for INT4).
    Requires PyTorch >= 2.1.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn not available. Requires PyTorch >= 2.1. "
            "Use --quant int4 as a fallback."
        )
    orig_dtype = tensor.dtype
    return tensor.to(torch.float8_e4m3fn).to(orig_dtype)


def int4_roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric INT4 round-trip: BF16 → INT4 → BF16.

    INT4 range: [-7, 7]  (15 levels, symmetric so zero is exact)
    Typical relative error per element: ~14 %
    """
    orig_dtype = tensor.dtype
    w = tensor.float()
    amax = w.abs().max()
    if amax == 0:
        return tensor
    scale = amax / 7.0
    w_int4 = (w / scale).round().clamp(-7, 7)
    return (w_int4 * scale).to(orig_dtype)


QUANT_FNS = {
    "fp8": fp8_roundtrip,
    "int4": int4_roundtrip,
}


def _degrade_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_key: str,
    quant_fn,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Apply *quant_fn* round-trip to *target_key* in *state_dict*.

    Returns the modified state dict and a small stats dict for logging.
    """
    if target_key not in state_dict:
        raise KeyError(
            f"Key '{target_key}' not found in codec.pth.\n"
            f"Run with --inspect to list available keys."
        )

    original = state_dict[target_key]
    degraded = quant_fn(original)

    abs_err = (degraded.float() - original.float()).abs()
    stats = {
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_err_pct": (abs_err / (original.float().abs() + 1e-9)).mean().item() * 100,
    }

    state_dict = dict(state_dict)   # shallow copy so we don't mutate caller's dict
    state_dict[target_key] = degraded
    return state_dict, stats


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_dir():
        return p
    log.info("Downloading model from HuggingFace: %s", model_path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path))


def _find_semantic_codebook_key(state_dict: dict[str, torch.Tensor]) -> str | None:
    """Heuristic: find the semantic quantizer codebook weight key."""
    candidates = [k for k in state_dict if "semantic" in k and "codebook" in k and "weight" in k]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Prefer the one that looks like an embedding (2-D, dim-8 or dim-16)
        for k in candidates:
            t = state_dict[k]
            if t.ndim == 2 and t.shape[-1] <= 32:
                return k
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Main actions
# ---------------------------------------------------------------------------

def inspect(checkpoint_dir: Path) -> None:
    codec_path = checkpoint_dir / "codec.pth"
    if not codec_path.exists():
        raise FileNotFoundError(f"codec.pth not found in {checkpoint_dir}")

    state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)

    print(f"\n=== codec.pth keys ({len(state_dict)} total) ===\n")
    for k, v in sorted(state_dict.items()):
        print(f"  {k:<70}  shape={list(v.shape)}  dtype={v.dtype}")

    candidate = _find_semantic_codebook_key(state_dict)
    if candidate:
        print(f"\n>>> Auto-detected target key: {candidate}")
        t = state_dict[candidate]
        print(f"    shape={list(t.shape)}  dtype={t.dtype}  "
              f"abs_max={t.float().abs().max():.4f}")
    else:
        print("\n>>> No clear semantic-codebook key found; set --target-key manually.")


def create_degraded_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    target_key: str | None,
    quant: str = "fp8",
) -> None:
    codec_path = checkpoint_dir / "codec.pth"
    if not codec_path.exists():
        raise FileNotFoundError(f"codec.pth not found in {checkpoint_dir}")

    log.info("Loading codec.pth from %s", codec_path)
    state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)

    if target_key is None:
        target_key = _find_semantic_codebook_key(state_dict)
        if target_key is None:
            raise RuntimeError(
                "Cannot auto-detect target key. "
                "Run with --inspect and pass --target-key explicitly."
            )
        log.info("Auto-detected target key: %s", target_key)

    quant_fn = QUANT_FNS[quant]
    log.info("Applying %s round-trip to: %s", quant.upper(), target_key)
    degraded_sd, stats = _degrade_state_dict(state_dict, target_key, quant_fn)

    log.info(
        "Quantisation stats — shape=%s dtype=%s "
        "max_abs_err=%.4f mean_abs_err=%.6f rel_err=%.2f%%",
        stats["shape"], stats["dtype"],
        stats["max_abs_err"], stats["mean_abs_err"], stats["rel_err_pct"],
    )

    if output_dir.exists():
        log.warning("Output dir already exists, removing: %s", output_dir)
        shutil.rmtree(output_dir)

    log.info("Copying checkpoint to %s", output_dir)
    shutil.copytree(checkpoint_dir, output_dir, symlinks=False)

    degraded_codec_path = output_dir / "codec.pth"
    if degraded_codec_path.exists() or degraded_codec_path.is_symlink():
        degraded_codec_path.unlink()
    log.info("Saving degraded codec.pth to %s", degraded_codec_path)
    torch.save(degraded_sd, degraded_codec_path)

    log.info("Done. Degraded checkpoint written to: %s", output_dir)
    log.info(
        "Start degraded server with:\n"
        "  python -m sglang_omni.cli.cli serve \\\n"
        "    --model-path %s \\\n"
        "    --config examples/configs/s2pro_tts.yaml \\\n"
        "    --port 8001",
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Degrade S2-Pro codec.pth for representativeness validation."
    )
    p.add_argument("--model-path", default="fishaudio/s2-pro",
                   help="HuggingFace repo ID or local checkpoint directory.")
    p.add_argument("--output-dir", default="/tmp/s2pro-degraded",
                   help="Where to write the degraded checkpoint (default: /tmp/s2pro-degraded).")
    p.add_argument("--target-key", default=None,
                   help="State-dict key to degrade. Auto-detected if omitted.")
    p.add_argument("--quant", choices=["fp8", "int4"], default="fp8",
                   help="Quantisation scheme for the round-trip (default: fp8).")
    p.add_argument("--inspect", action="store_true",
                   help="Print all codec.pth keys and exit (no files written).")
    args = p.parse_args()

    checkpoint_dir = _resolve_checkpoint(args.model_path)

    if args.inspect:
        inspect(checkpoint_dir)
        return

    create_degraded_checkpoint(
        checkpoint_dir=checkpoint_dir,
        output_dir=Path(args.output_dir),
        target_key=args.target_key,
        quant=args.quant,
    )


if __name__ == "__main__":
    main()
