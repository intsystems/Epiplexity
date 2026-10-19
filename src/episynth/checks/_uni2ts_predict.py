"""Прогноз `MoiraiForecast` из uni2ts на сохранённых окнах.

    python _uni2ts_predict.py <каталог чекпойнта> <windows.npz> <out.npy> <patch> <context> <n_samples>

Запускается интерпретатором с uni2ts 2.0.0 и gluonts. Входы: массивы `h<i>` (истории) и
скаляр `h` (горизонт). Выход: медианы прогнозов `(B, h)`.
"""

import sys

import numpy as np
import pandas as pd
import torch


def main(ckpt_dir: str, windows: str, out: str, patch: int, context: int, n_samples: int) -> None:
    """Считает медианы прогнозов и сохраняет их в `out`."""
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    z = np.load(windows)
    horizon = int(z["h"])
    hist = [z[k] for k in sorted(z.files, key=lambda s: (len(s), s)) if k.startswith("h") and k != "h"]
    ds = ListDataset([{"start": pd.Period("2000-01-01", freq="D"), "target": x} for x in hist], freq="D")
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(ckpt_dir),
        prediction_length=horizon,
        context_length=context,
        patch_size=patch,
        num_samples=n_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    torch.manual_seed(0)
    medians = [f.quantile(0.5) for f in model.create_predictor(batch_size=len(hist)).predict(ds)]
    np.save(out, np.stack(medians))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], int(a[4]), int(a[5]), int(a[6]))
