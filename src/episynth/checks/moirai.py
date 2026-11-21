r"""Verification of the Moirai port with the original uni2ts on one checkpoint and one input.

    python -m episynth.checks.moirai --checkpoint <catalogue with model.safetensors and config.json>
        --uni2ts-python <interpreter with uni2ts 2.0.0> [-out result.json]

Builds windows of three generators for patch sizes 8, 32 and 64, considers the original as a
separate interpreter (`_uni2ts_forward.py`) and transfer in the current, compares rationing, weight
logs, component parameters and log density. Prints the greatest absolute discrepancies; the log
density is expected rounding error level float32 (different attention cores in different versions of
torch).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from episynth.generators import GENERATORS
from episynth.model import MoiraiModule, load_uni2ts_checkpoint
from episynth.model.moirai import small
from episynth.windows import WindowSpec, make_windows

_HELPER = Path(__file__).with_name("_uni2ts_forward.py")


def compare(ckpt: Path, uni2ts_python: str, patch_size: int, n: int = 6) -> dict[str, float]:
    """The greatest difference between the port and the original for the same patch size."""
    spec = WindowSpec(patch_size=patch_size, context_lengths=(200, 512), dequantize_width=0.0)
    rows, seeds = [], np.arange(n)
    for name in ("sde", "waveform", "ets"):
        rows.append(GENERATORS[name](length=1024).generate(seeds))
    series = np.concatenate(rows)
    batch = make_windows(series, np.arange(len(series)), spec)
    model = MoiraiModule(small())
    load_uni2ts_checkpoint(model, str(ckpt / "model.safetensors"))
    model.eval()
    with torch.no_grad():
        mix = model(**batch.model_inputs(), patch_size=patch_size)
        ours = {
            "loc": mix.loc.numpy(),
            "scale": mix.scale.numpy(),
            "weights_logits": torch.log_softmax(mix.logits, -1).numpy(),
            "log_prob": mix.log_prob(batch.target).numpy(),
        }
        for i, p in enumerate(mix.params):
            for key, val in p.items():
                ours[f"comp{i}_{key}"] = val.numpy()
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "inputs.npz", Path(tmp) / "out.npz"
        np.savez(inp, patch_size=patch_size, **{k: v.numpy() for k, v in batch.model_inputs().items()})
        subprocess.run([uni2ts_python, "-I", str(_HELPER), str(ckpt), str(inp), str(out)], check=True)
        ref = dict(np.load(out))
    real = batch.sample_id.numpy().astype(bool)
    diffs = {}
    for key, theirs in ref.items():
        if key not in ours:
            continue
        mine = ours[key]
        if mine.ndim == 3 and mine.shape[-1] >= patch_size and key != "loc" and key != "scale":
            mine, theirs = mine[..., :patch_size], theirs[..., :patch_size]
        if mine.ndim == 4:
            theirs = theirs[..., : mine.shape[-2], :]
        m = real[..., None] if mine.ndim == 3 else real[..., None, None]
        diffs[key] = float(np.abs(np.where(m, mine - theirs, 0.0)).max())
    return diffs


def main() -> None:
    """Analysis of arguments, reconciliation of the size of the patch, printing and recording the result."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--uni2ts-python", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = {}
    for patch_size in (8, 32, 64):
        result[patch_size] = compare(args.checkpoint, args.uni2ts_python, patch_size)
        worst = max(result[patch_size].items(), key=lambda kv: kv[1])
        print(f"patch {patch_size:3d}: log_prob max abs diff {result[patch_size]['log_prob']:.2e}; worst {worst[0]} {worst[1]:.2e}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
