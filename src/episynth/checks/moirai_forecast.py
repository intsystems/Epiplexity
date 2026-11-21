r"""Verification of Moirai forecasts in the protocol of the GIFT-Eval notebook with `MoiraiForecast` from uni2ts on the windows of the dataset.

    python -m episynth.checks.moirai_forecast --data <GiftEval image> --checkpoint <checkpoint directory>
        --uni2ts-python <interpreter with uni2ts> [-dataset us_births/D] [-n-windows 4]

Takes the first windows of the dataset, considers the forecast medians our forecaster and the
original (a separate interpreter), and prints the MAE of both against the truth and the average
median difference. The difference should be at the level of sampling noise.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from episynth.evaluation.forecasters import MoiraiForecaster
from episynth.evaluation.gift_eval import GiftDataset
from episynth.model import MoiraiModule, load_uni2ts_checkpoint
from episynth.model.moirai import small

_HELPER = Path(__file__).with_name("_uni2ts_predict.py")


def main() -> None:
    """Argument analysis, both forecasts and comparison seal."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--uni2ts-python", required=True)
    parser.add_argument("--dataset", default="us_births/D")
    parser.add_argument("--n-windows", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=4000)
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    ds = GiftDataset(args.data, args.dataset)
    inst = list(ds.test_instances())[: args.n_windows]
    model = MoiraiModule(small())
    load_uni2ts_checkpoint(model, str(args.checkpoint / "model.safetensors"))
    torch.manual_seed(0)
    fc = MoiraiForecaster(model, args.patch_size, "notebook", args.context, n_samples=args.n_samples, batch_size=len(inst))
    ours = np.median(fc.predict([x.history for x in inst], ds.prediction_length).samples, axis=0)
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "windows.npz", Path(tmp) / "medians.npy"
        np.savez(inp, h=ds.prediction_length, **{f"h{i}": x.history for i, x in enumerate(inst)})
        subprocess.run(
            [args.uni2ts_python, "-I", str(_HELPER), str(args.checkpoint), str(inp), str(out), str(args.patch_size), str(args.context), str(args.n_samples)],
            check=True,
            capture_output=True,
        )
        theirs = np.load(out)
    for i, x in enumerate(inst):
        valid = np.isfinite(x.label)
        print(
            f"Window {i}: our MAE {np.abs(ours[i] - x.label)[valid].mean():.4g}, uni2ts {np.abs(theirs[i] - x.label)[valid].mean():.4g}; "
            f"mean difference between medians {np.abs(ours[i] - theirs[i]).mean():.4g}, ground-truth scale {np.abs(x.label[valid]).mean():.4g}"
        )


if __name__ == "__main__":
    main()
