"""Прямой проход оригинального `MoiraiModule` из uni2ts на сохранённых входах.

    python _uni2ts_forward.py <каталог чекпойнта> <inputs.npz> <out.npz>

Запускается интерпретатором с установленным uni2ts 2.0.0. Входы: массивы `target`,
`observed_mask`, `sample_id`, `time_id`, `variate_id`, `prediction_mask` и скаляр `patch_size`.
Выход: `loc`, `scale`, `weights_logits` и параметры компонент после отображений в область
определения, `log_prob` на `target`.
"""

import sys

import numpy as np
import torch


def main(ckpt_dir: str, inputs: str, out: str) -> None:
    """Считает выходы оригинала и сохраняет их в `out`."""
    from uni2ts.model.moirai import MoiraiModule

    module = MoiraiModule.from_pretrained(ckpt_dir).eval()
    z = np.load(inputs)
    patch_size = int(z["patch_size"])
    target = torch.from_numpy(z["target"])
    width = max(module.patch_sizes)
    target = torch.nn.functional.pad(target, (0, width - target.shape[-1]))
    observed = torch.nn.functional.pad(torch.from_numpy(z["observed_mask"]), (0, width - z["observed_mask"].shape[-1]))
    sample_id = torch.from_numpy(z["sample_id"])
    ps = torch.full_like(sample_id, patch_size)
    with torch.no_grad():
        distr = module(
            target=target,
            observed_mask=observed,
            sample_id=sample_id,
            time_id=torch.from_numpy(z["time_id"]),
            variate_id=torch.from_numpy(z["variate_id"]),
            prediction_mask=torch.from_numpy(z["prediction_mask"]),
            patch_size=ps,
        )
        mixture = distr.base_dist
        result = {
            "loc": distr.loc.numpy(),
            "scale": distr.scale.numpy(),
            "weights_logits": mixture.weights.logits.numpy(),
            "log_prob": distr.log_prob(target).numpy(),
        }
        for i, comp in enumerate(mixture.components):
            for name in ("df", "loc", "scale", "total_count", "logits"):
                if hasattr(comp, name):
                    val = getattr(comp, name)
                    if torch.is_tensor(val) and val.shape == mixture.weights.logits.shape[:-1]:
                        result[f"comp{i}_{name}"] = val.numpy()
    np.savez(out, **result)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
