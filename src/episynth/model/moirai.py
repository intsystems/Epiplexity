r"""Перенос Moirai 1.x (Woo и др., 2024, arXiv:2402.02592) из uni2ts 2.0.0 (Salesforce, Apache-2.0).

Архитектура, инициализация и имена параметров повторяют `uni2ts.model.moirai.MoiraiModule`,
поэтому опубликованные чекпойнты (`Salesforce/moirai-1.1-R-small` и другие) загружаются как есть
(`load_uni2ts_checkpoint`). Отличия переноса:

- размер патча один на батч (целое число), а не на токен: упаковка рядов с разными патчами в
  одну последовательность не поддерживается;
- параметры распределения, лог-плотность и лосс считаются в float32 независимо от autocast;
- смесь и её компоненты реализованы формулами (годится для `torch.compile`), сэмплирование
  через `torch.distributions`;
- без jaxtyping, einops и hydra.

Сохранённая ради чекпойнтов особенность оригинала: проекция логитов смеси для патча :math:`P`
заполняет весами первые :math:`K P` строк матрицы ширины :math:`K P_{\max}`, а читается как
:math:`(K, P_{\max})`, поэтому при нескольких размерах патчей и :math:`P < P_{\max}` логиты части компонент
не зависят от представления и равны обучаемым смещениям. При одном размере патча в
конфигурации (:math:`P_{\max} = P`) особенность исчезает.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

_COMPONENT_PARAMS = {
    "student_t": ("df", "loc", "scale"),
    "normal_fixed": ("loc",),
    "negative_binomial": ("total_count", "logits"),
    "log_normal": ("loc", "scale"),
}
_LOG_2PI = math.log(2 * math.pi)


class MoiraiConfig(BaseModel):
    """Конфигурация модели.

    Attributes:
        d_model: Ширина представлений; число голов внимания равно `d_model // 64`.
        num_layers: Число слоёв трансформера.
        patch_sizes: Поддерживаемые размеры патча; размер патча батча должен быть из этого набора.
        max_seq_len: Наибольшее число токенов в последовательности (длина таблиц RoPE).
        components: Компоненты смеси по порядку из {"student_t", "normal_fixed", "negative_binomial", "log_normal"}.
        normal_fixed_scale: Стандартное отклонение компоненты "normal_fixed" в нормированных единицах.
        scaling: Нормировать ли ряд по наблюдаемому контексту.
    """

    d_model: int = 384
    num_layers: int = 6
    patch_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    max_seq_len: int = 512
    components: tuple[str, ...] = ("student_t", "normal_fixed", "negative_binomial", "log_normal")
    normal_fixed_scale: float = 1e-3
    scaling: bool = True


def small(patch_sizes: Sequence[int] = (8, 16, 32, 64, 128), components: Sequence[str] | None = None) -> MoiraiConfig:
    """Конфигурация Moirai-Small (13,8 млн параметров при всех патчах и четырёх компонентах).

    Args:
        patch_sizes: Размеры патча.
        components: Компоненты смеси; `None` — четыре компоненты оригинала.

    Returns:
        MoiraiConfig: Конфигурация.
    """
    kwargs = {} if components is None else {"components": tuple(components)}
    return MoiraiConfig(patch_sizes=tuple(patch_sizes), **kwargs)


class RMSNorm(nn.Module):
    """Нормировка по среднеквадратичному значению с обучаемым масштабом.

    Args:
        dim: Нормируемая размерность.
        eps: Слагаемое под корнем.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Нормирует последнюю ось `x` и умножает на масштаб."""
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class RotaryProjection(nn.Module):
    """Поворотное позиционное кодирование (RoPE) первой половины размерности головы.

    Args:
        width: Число поворачиваемых координат (чётное).
        max_len: Наибольший индекс времени плюс один.
        base: Основание геометрической сетки частот.
    """

    def __init__(self, width: int, max_len: int, base: int = 10000) -> None:
        super().__init__()
        self.width = width
        theta = 1.0 / torch.pow(base, torch.arange(0, width, 2, dtype=torch.float) / width)
        m_theta = torch.outer(torch.arange(max_len, dtype=torch.float), theta).repeat_interleave(2, dim=-1)
        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, time_id: torch.Tensor) -> torch.Tensor:
        """Поворачивает первые `width` координат `x` `(B, H, S, hd)` по индексам времени `(B, S)`."""
        cos = self.cos[time_id].unsqueeze(1).to(x.dtype)
        sin = self.sin[time_id].unsqueeze(1).to(x.dtype)
        head, tail = x[..., : self.width], x[..., self.width :]
        return torch.cat([cos * head + sin * self._rotate(head), tail], dim=-1)


class BinaryAttentionBias(nn.Module):
    """Смещение внимания по признаку «та же переменная или другая», своё на каждую голову.

    Args:
        num_heads: Число голов.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(2, num_heads)

    def forward(self, variate_id: torch.Tensor) -> torch.Tensor:
        """Смещение `(B, H, S, S)` по индексам переменных `(B, S)`."""
        same = (variate_id.unsqueeze(-1) == variate_id.unsqueeze(-2)).unsqueeze(1)
        w = self.emb.weight.view(2, -1, 1, 1)
        return torch.where(same, w[1], w[0])


class Attention(nn.Module):
    """Многоголовое внимание с нормировкой запросов и ключей, RoPE по времени и смещением по переменной.

    Args:
        d_model: Ширина представлений.
        num_heads: Число голов.
        rope: Общий для слоёв модуль RoPE.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryProjection) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.var_attn_bias = BinaryAttentionBias(num_heads)
        self.rope = rope

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        time_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """Выход внимания `(B, S, d)` для `x` той же формы, булевой маски `(B, S, S)` и индексов `(B, S)`."""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        bias = self.var_attn_bias(variate_id).to(q.dtype)
        bias = bias.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        q, k = self.rope(q, time_id), self.rope(k, time_id)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1 / math.sqrt(self.head_dim))
        return self.out_proj(out.transpose(1, 2).reshape(B, S, -1))


class GatedFeedForward(nn.Module):
    """Полносвязный блок с SiLU-гейтом (SwiGLU) без смещений.

    Args:
        d_model: Ширина входа и выхода.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = (int(4 * d_model * 2 / 3) + 7) // 8 * 8
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc_gate = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Выход блока той же формы, что `x`."""
        return self.fc2(F.silu(self.fc_gate(x)) * self.fc1(x))


class EncoderLayer(nn.Module):
    """Слой трансформера с нормировкой до внимания и до полносвязного блока.

    Args:
        d_model: Ширина представлений.
        num_heads: Число голов.
        rope: Общий модуль RoPE.
    """

    def __init__(self, d_model: int, num_heads: int, rope: RotaryProjection) -> None:
        super().__init__()
        self.self_attn = Attention(d_model, num_heads, rope)
        self.ffn = GatedFeedForward(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        time_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """Выход слоя той же формы, что `x`."""
        x = x + self.self_attn(self.norm1(x), attn_mask, time_id, variate_id)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    """Стек слоёв трансформера с общим RoPE и финальной нормировкой.

    Args:
        d_model: Ширина представлений.
        num_layers: Число слоёв.
        max_seq_len: Длина таблиц RoPE.
    """

    def __init__(self, d_model: int, num_layers: int, max_seq_len: int) -> None:
        super().__init__()
        num_heads = d_model // 64
        self.rope = RotaryProjection((d_model // num_heads) // 2, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, self.rope) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        time_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """Представления `(B, S, d)` после всех слоёв."""
        for layer in self.layers:
            x = layer(x, attn_mask, time_id, variate_id)
        return self.norm(x)


class MultiInSizeLinear(nn.Module):
    """Линейный слой с отдельными весами на каждый размер патча, вход дополнен нулями до наибольшего.

    Args:
        in_sizes: Размеры патча.
        out_features: Ширина выхода.
    """

    def __init__(self, in_sizes: Sequence[int], out_features: int) -> None:
        super().__init__()
        self.in_sizes = tuple(in_sizes)
        self.weight = nn.Parameter(torch.empty(len(in_sizes), out_features, max(in_sizes)))
        self.bias = nn.Parameter(torch.empty(len(in_sizes), out_features))
        for idx, size in enumerate(self.in_sizes):
            nn.init.kaiming_uniform_(self.weight[idx, :, :size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, size:])
            nn.init.uniform_(self.bias[idx], -1 / math.sqrt(size), 1 / math.sqrt(size))

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Выход `(..., out_features)` для патчей `x` `(..., P_max)` с весами размера номер `idx`."""
        size = self.in_sizes[idx]
        return F.linear(x[..., :size], self.weight[idx, :, :size], self.bias[idx])


class MultiOutSizeLinear(nn.Module):
    """Линейный слой с отдельными весами на каждый размер патча, выход читается как `(P_max, dim)`.

    Args:
        in_features: Ширина входа.
        patch_sizes: Размеры патча.
        dim: Число параметров на точку патча.
    """

    def __init__(self, in_features: int, patch_sizes: Sequence[int], dim: int) -> None:
        super().__init__()
        self.patch_sizes = tuple(patch_sizes)
        self.dim = dim
        self.out_size = max(patch_sizes)
        width = dim * self.out_size
        self.weight = nn.Parameter(torch.empty(len(patch_sizes), width, in_features))
        self.bias = nn.Parameter(torch.empty(len(patch_sizes), width))
        for idx, size in enumerate(self.patch_sizes):
            rows = dim * size
            nn.init.kaiming_uniform_(self.weight[idx, :rows], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, rows:])
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias[idx, :rows], -bound, bound)
            nn.init.zeros_(self.bias[idx, rows:])

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Параметры `(..., P_max, dim)` для представлений `x` `(..., in_features)`.

        Веса за пределами первых `dim * P` строк обнулены, смещение действует целиком, как в
        оригинале: у строк без весов остаётся обучаемая константа.
        """
        rows = self.dim * self.patch_sizes[idx]
        out = F.pad(F.linear(x, self.weight[idx, :rows]), (0, self.weight.shape[1] - rows)) + self.bias[idx]
        return out.view(*x.shape[:-1], self.dim, self.out_size).transpose(-1, -2)


class ParamProj(nn.Module):
    """Проекция представлений в параметры смеси: логиты весов и параметры компонент.

    Args:
        d_model: Ширина представлений.
        patch_sizes: Размеры патча.
        components: Имена компонент смеси.
    """

    def __init__(self, d_model: int, patch_sizes: Sequence[int], components: Sequence[str]) -> None:
        super().__init__()
        self.components = tuple(components)
        self.proj = nn.ModuleDict(
            {
                "weights_logits": MultiOutSizeLinear(d_model, patch_sizes, len(components)),
                "components": nn.ModuleList(
                    [
                        nn.ModuleDict({p: MultiOutSizeLinear(d_model, patch_sizes, 1) for p in _COMPONENT_PARAMS[c]})
                        for c in components
                    ]
                ),
            }
        )

    def forward(self, h: torch.Tensor, idx: int) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Логиты весов `(B, S, P_max, K)` и список словарей параметров компонент `(B, S, P_max)`, всё float32."""
        logits = self.proj["weights_logits"](h, idx).float()
        params = []
        for name, proj in zip(self.components, self.proj["components"], strict=True):
            raw = {p: proj[p](h, idx).float().squeeze(-1) for p in _COMPONENT_PARAMS[name]}
            params.append(_constrain(name, raw))
        return logits, params


def _constrain(name: str, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    eps = torch.finfo(torch.float32).eps
    if name == "student_t":
        return {"df": 2.0 + F.softplus(raw["df"]), "loc": raw["loc"], "scale": F.softplus(raw["scale"]).clamp_min(eps)}
    if name == "normal_fixed":
        return {"loc": raw["loc"]}
    if name == "negative_binomial":
        return {"total_count": F.softplus(raw["total_count"]), "logits": raw["logits"]}
    return {"loc": raw["loc"], "scale": F.softplus(raw["scale"]).clamp_min(eps)}


class Mixture:
    r"""Смесь распределений на каждую точку патча в исходных единицах ряда.

    Компоненты заданы в нормированных единицах :math:`z = (y - \mathrm{loc}) / \mathrm{scale}`, где
    `loc` и `scale` получены по наблюдаемому контексту. Лог-плотность в исходных единицах равна
    лог-плотности смеси в :math:`z` минус :math:`\log \mathrm{scale}`. Компоненты с ограниченным носителем
    (отрицательное биномиальное на :math:`z \ge 0`, логнормальное на :math:`z > 0`) вне носителя дают нулевую
    плотность. Отрицательное биномиальное вычисляется на нецелых :math:`z` через гамма-функцию, как в
    оригинале: на вещественной оси это не плотность, и смесь с ним не нормирована.

    Args:
        logits: Логиты весов `(B, S, O, K)`.
        params: Параметры компонент по порядку, каждый тензор `(B, S, O)`.
        names: Имена компонент.
        normal_fixed_scale: Стандартное отклонение компоненты "normal_fixed".
        loc: Сдвиг нормировки `(B, S, 1)`.
        scale: Масштаб нормировки `(B, S, 1)`.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        params: list[dict[str, torch.Tensor]],
        names: Sequence[str],
        normal_fixed_scale: float,
        loc: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.logits = logits
        self.params = params
        self.names = names
        self.normal_fixed_scale = normal_fixed_scale
        self.loc = loc
        self.scale = scale

    def _component_log_prob(self, name: str, p: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        if name == "student_t":
            y = (z - p["loc"]) / p["scale"]
            df = p["df"]
            log_z = (
                torch.log(p["scale"])
                + 0.5 * torch.log(df)
                + 0.5 * math.log(math.pi)
                + torch.lgamma(0.5 * df)
                - torch.lgamma(0.5 * (df + 1.0))
            )
            return -0.5 * (df + 1.0) * torch.log1p(y**2 / df) - log_z
        if name == "normal_fixed":
            s = self.normal_fixed_scale
            return -((z - p["loc"]) ** 2) / (2 * s * s) - math.log(s) - 0.5 * _LOG_2PI
        if name == "negative_binomial":
            valid = z >= 0
            v = torch.where(valid, z, torch.ones_like(z))
            r = p["total_count"]
            lp = (
                r * F.logsigmoid(-p["logits"])
                + F.logsigmoid(p["logits"]) * v
                - (torch.lgamma(1 + v) + torch.lgamma(r) - torch.lgamma(1 + v + r) + torch.log(r + v))
            )
            return torch.where(valid, lp, torch.full_like(lp, float("-inf")))
        valid = z > 0
        v = torch.where(valid, z, torch.ones_like(z))
        log_v = torch.log(v)
        lp = -((log_v - p["loc"]) ** 2) / (2 * p["scale"] ** 2) - torch.log(p["scale"]) - 0.5 * _LOG_2PI - log_v
        return torch.where(valid, lp, torch.full_like(lp, float("-inf")))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Лог-плотность `(B, S, O)` значений `value` `(B, S, P)` в исходных единицах, в натах.

        При `P < O` значения дополняются нулями до ширины `O`; осмысленны первые `P` столбцов.
        """
        width = self.logits.shape[-2]
        if value.shape[-1] < width:
            value = F.pad(value, (0, width - value.shape[-1]))
        z = ((value - self.loc) / self.scale).float()
        log_w = F.log_softmax(self.logits, dim=-1)
        comp = torch.stack([self._component_log_prob(n, p, z) for n, p in zip(self.names, self.params, strict=True)], dim=-1)
        log_w = torch.where(torch.isinf(comp), torch.zeros_like(log_w), log_w)
        return torch.logsumexp(log_w + comp, dim=-1) - torch.log(self.scale)

    def sample(self, n: int) -> torch.Tensor:
        """Сэмплы `(n, B, S, O)` в исходных единицах."""
        with torch.no_grad():
            dists = []
            for name, p in zip(self.names, self.params, strict=True):
                if name == "student_t":
                    dists.append(torch.distributions.StudentT(p["df"], p["loc"], p["scale"]))
                elif name == "normal_fixed":
                    dists.append(torch.distributions.Normal(p["loc"], self.normal_fixed_scale))
                elif name == "negative_binomial":
                    gamma = torch.distributions.Gamma(p["total_count"], torch.exp(-p["logits"]))
                    dists.append(_PoissonGamma(gamma))
                else:
                    dists.append(torch.distributions.LogNormal(p["loc"], p["scale"]))
            comps = torch.stack([d.sample((n,)) for d in dists], dim=-1)
            which = torch.distributions.Categorical(logits=self.logits).sample((n,)).unsqueeze(-1)
            z = torch.gather(comps, -1, which).squeeze(-1)
            return z * self.scale + self.loc


class _PoissonGamma:
    """Отрицательное биномиальное как пуассоновская смесь гамма-распределений, только сэмплирование."""

    def __init__(self, gamma) -> None:
        self.gamma = gamma

    def sample(self, shape) -> torch.Tensor:
        """Целочисленные сэмплы формы `shape + batch`."""
        return torch.poisson(self.gamma.sample(shape))


class PackedStdScaler(nn.Module):
    """Нормировка ряда по среднему и несмещённому стандартному отклонению наблюдаемого контекста.

    Статистики считаются в float64 по токенам с теми же `sample_id` и `variate_id`; у токенов
    заполнения (`sample_id == 0`) сдвиг 0 и масштаб 1.

    Args:
        minimum_scale: Слагаемое к дисперсии под корнем.
    """

    def __init__(self, minimum_scale: float = 1e-5) -> None:
        super().__init__()
        self.minimum_scale = minimum_scale

    def forward(
        self, target: torch.Tensor, observed_mask: torch.Tensor,
        sample_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Сдвиг и масштаб `(B, S, 1)` float32 для `target` `(B, S, P)`, маски `(B, S, P)` и индексов `(B, S)`."""
        x = target.double()
        same = (sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2)) & (variate_id.unsqueeze(-1) == variate_id.unsqueeze(-2))
        same = same.to(x.dtype)
        obs = observed_mask.to(x.dtype)
        tobs = (same * obs.sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
        loc = (same * (x * obs).sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
        loc = loc / torch.where(tobs == 0, torch.ones_like(tobs), tobs)
        var = (same * (((x - loc) ** 2) * obs).sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
        denom = tobs - 1
        var = var / torch.where(denom == 0, torch.ones_like(denom), denom)
        scale = torch.sqrt(var + self.minimum_scale)
        pad = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(pad, torch.zeros_like(loc), loc)
        scale = torch.where(pad, torch.ones_like(scale), scale)
        return loc.float(), scale.float()


class MoiraiModule(nn.Module):
    """Модель Moirai: патчи ряда в представления, трансформер, смесь распределений на точки патча.

    Args:
        config: Конфигурация.

    Attributes:
        config: Конфигурация модели.
    """

    def __init__(self, config: MoiraiConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_encoding = nn.Embedding(1, config.d_model)
        self.scaler = PackedStdScaler() if config.scaling else None
        self.in_proj = MultiInSizeLinear(config.patch_sizes, config.d_model)
        self.encoder = Encoder(config.d_model, config.num_layers, config.max_seq_len)
        self.param_proj = ParamProj(config.d_model, config.patch_sizes, config.components)

    def forward(
        self,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_id: torch.Tensor,
        time_id: torch.Tensor,
        variate_id: torch.Tensor,
        prediction_mask: torch.Tensor,
        patch_size: int,
    ) -> Mixture:
        """Предсказательная смесь для всех токенов батча.

        Args:
            target: Значения патчей `(B, S, P_max)`; у прогнозных и ненаблюдаемых позиций любые.
            observed_mask: Булева маска наблюдаемых значений `(B, S, P_max)`.
            sample_id: Номер ряда токена `(B, S)`, 0 — заполнение вне внимания и нормировки.
            time_id: Индекс времени токена `(B, S)`, меньше `max_seq_len`.
            variate_id: Индекс переменной токена `(B, S)`.
            prediction_mask: Булева маска прогнозных токенов `(B, S)`.
            patch_size: Размер патча батча из `config.patch_sizes`.

        Returns:
            Mixture: Смесь с параметрами `(B, S, P_max, ...)` в исходных единицах ряда.
        """
        idx = self.config.patch_sizes.index(patch_size)
        if self.scaler is not None:
            loc, scale = self.scaler(target, observed_mask & ~prediction_mask.unsqueeze(-1), sample_id, variate_id)
        else:
            loc = torch.zeros(*target.shape[:2], 1, device=target.device)
            scale = torch.ones_like(loc)
        reprs = self.in_proj((target - loc) / scale, idx)
        reprs = torch.where(prediction_mask.unsqueeze(-1), self.mask_encoding.weight.to(reprs.dtype), reprs)
        attn_mask = sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2)
        h = self.encoder(reprs, attn_mask, time_id, variate_id)
        logits, params = self.param_proj(h, idx)
        return Mixture(logits, params, self.config.components, self.config.normal_fixed_scale, loc, scale)


def load_uni2ts_checkpoint(module: MoiraiModule, path: str) -> None:
    """Загружает веса из `model.safetensors` чекпойнта uni2ts в модуль той же конфигурации.

    Args:
        module: Модель с теми же размерами патчей и компонентами, что у чекпойнта.
        path: Путь к файлу safetensors.
    """
    from safetensors.torch import load_file

    module.load_state_dict(load_file(path), strict=True)
