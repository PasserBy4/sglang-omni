# SPDX-License-Identifier: Apache-2.0
"""SeedTTS dataset loader for seed-tts-eval meta.lst files."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class SampleInput:
    sample_id: str
    ref_text: str
    ref_audio: str
    target_text: str


def load_seedtts_samples(
    path: str,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[SampleInput]:
    """Parse a seed-tts-eval meta.lst file.

    Format per line: ``id|ref_text|ref_audio_path|target_text``

    Args:
        path: Path to the meta.lst file.
        max_samples: Maximum number of samples to return. ``None`` means all.
        shuffle: If True, randomly sample instead of taking the first N.
        seed: Random seed for reproducible sampling (only used when *shuffle* is True).
    """
    base_dir = os.path.dirname(path)
    samples: list[SampleInput] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            samples.append(
                SampleInput(
                    sample_id=parts[0],
                    ref_text=parts[1],
                    ref_audio=os.path.join(base_dir, parts[2]),
                    target_text=parts[3],
                )
            )
            if max_samples and not shuffle and len(samples) >= max_samples:
                break

    if shuffle and max_samples and len(samples) > max_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, max_samples)

    return samples
