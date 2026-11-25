r"""Port of Moirai 1.x (Woo et al., 2024, arXiv:2402.02592) from uni2ts 2.0.0
(Salesforce, Apache-2.0).

The port follows `uni2ts.model.moirai.MoiraiModule` for architecture,
initialization, and parameter names. `load_uni2ts_checkpoint` can load
published checkpoints such as `Salesforce/moirai-1.1-R-small` directly.

- Use one integer patch size for the whole batch. Packing mixed patch sizes
  within a sequence is unsupported.
- Distribution parameters, log densities, and losses are computed in float32,
  regardless of autocast.
- The mixture and its components use explicit formulas compatible with
  `torch.compile`. Sampling uses `torch.distributions`.
- The port has no dependency on jaxtyping, einops, or Hydra.

Checkpoint compatibility preserves a projection detail. For patch size
:math:`P`, the mixture-logit projection fills the first :math:`KP` rows of
the output of width :math:`KP_{\max}` and reshapes it to :math:`(K,P_{\max})`.
With multiple patch sizes and :math:`P < P_{\max}`, some logits equal the
learned biases independently of the input representation. With a single
patch size, :math:`P=P_{\max}` and this no longer occurs.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from pydantic import BaseModel, model_validator
from torch import nn

_COMPONENT_PARAMS = {
    "student_t": ("df", "loc", "scale"),
    "normal_fixed": ("loc",),
    "negative_binomial": ("total_count", "logits"),
    "log_normal": ("loc", "scale"),
}
_LOG_2PI = math.log(2 * math.pi)


class MoiraiConfig(BaseModel):
    """Model configuration.

    Attributes:
        d_model: Model width. The number of attention heads is `d_model // 64`.
        num_layers: Number of transformer layers.
        patch_sizes: Supported patch sizes; the batch patch size should be from
            this set.
        max_seq_len: The largest number of tokens in the sequence (the length
            of the RoPE tables).
        components: The components of the mixture in order of {"student_t",
            "normal_fixed", "negative_binomial", "log_normal"}.
        normal_fixed_scale: The standard deviation of the "normal_fixed"
            component in normalized units.
        scaling: Normalize the series according to the observed context.
        fixed_scaling: Fixed (shift, scale) instead of contextual statistics.
        ignore_context: Predict from canonical horizon tokens without context.
    """

    d_model: int = 384
    num_layers: int = 6
    patch_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    max_seq_len: int = 512
    components: tuple[str, ...] = (
        "student_t",
        "normal_fixed",
        "negative_binomial",
        "log_normal",
    )
    normal_fixed_scale: float = 1e-3
    scaling: bool = True
    fixed_scaling: tuple[float, float] | None = None
    ignore_context: bool = False

    @model_validator(mode="after")
    def _check_scaling(self) -> "MoiraiConfig":
        if self.fixed_scaling is not None:
            loc, scale = self.fixed_scaling
            if (
                not self.scaling
                or not math.isfinite(loc)
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError(
                    "fixed_scaling requires scaling=True, finite loc and"
                    " positive finite scale"
                )
        return self


def small(
    patch_sizes: Sequence[int] = (8, 16, 32, 64, 128),
    components: Sequence[str] | None = None,
) -> MoiraiConfig:
    """Moirai-Small configuration (13,8 million parameters for all patches and
    four components).

    Args:
        patch_sizes: Patch sizes.
        components: The components of the mixture; `None` are the four
            components of the original.

    Returns:
        MoiraiConfig: Config.
    """
    kwargs = {} if components is None else {"components": tuple(components)}
    return MoiraiConfig(patch_sizes=tuple(patch_sizes), **kwargs)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned scale.

    Args:
        dim: Size of the normalized dimension.
        eps: Constant added before taking the square root.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the last axis of `x` and multiplies by scale."""
        return (
            x
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


class RotaryProjection(nn.Module):
    """Rotary positional embedding (RoPE) of the first half of the head
    dimension.

    Args:
        width: The number of rotating coordinates (even).
        max_len: Highest time index plus one.
        base: The basis of the geometric frequency grid.
    """

    def __init__(self, width: int, max_len: int, base: int = 10000) -> None:
        super().__init__()
        self.width = width
        theta = 1.0 / torch.pow(
            base, torch.arange(0, width, 2, dtype=torch.float) / width
        )
        m_theta = torch.outer(
            torch.arange(max_len, dtype=torch.float), theta
        ).repeat_interleave(2, dim=-1)
        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, time_id: torch.Tensor) -> torch.Tensor:
        """Rotates the first `width` coordinates `x` `(B, H, S, hd)` by time
        indices `(B, S)`.
        """
        cos = self.cos[time_id].unsqueeze(1).to(x.dtype)
        sin = self.sin[time_id].unsqueeze(1).to(x.dtype)
        head, tail = x[..., : self.width], x[..., self.width:]
        return torch.cat([cos * head + sin * self._rotate(head), tail], dim=-1)


class BinaryAttentionBias(nn.Module):
    """Displacement of attention on the basis of "same variable or another" on
    each head.

    Args:
        num_heads: Number of heads.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(2, num_heads)

    def forward(self, variate_id: torch.Tensor) -> torch.Tensor:
        """`(B, H, S, S)` shifts over the `(B, S)` variable indices."""
        same = (
            variate_id.unsqueeze(-1) == variate_id.unsqueeze(-2)
        ).unsqueeze(1)
        w = self.emb.weight.view(2, -1, 1, 1)
        return torch.where(same, w[1], w[0])


class Attention(nn.Module):
    """Multi-headed attention with standardization of queries and keys, RoPE in
    time and offset by variable.

    Args:
        d_model: Model width.
        num_heads: Number of heads.
        rope: Common for layers module RoPE.
    """

    def __init__(
        self, d_model: int, num_heads: int, rope: RotaryProjection
    ) -> None:
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
        """Attention `(B, S, d)` for `x` of the same shape, Boolean mask `(B,
        S, S)` and indexes `(B, S)`.
        """
        B, S, _ = x.shape
        q = (
            self.q_proj(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        q, k = self.q_norm(q), self.k_norm(k)
        bias = self.var_attn_bias(variate_id).to(q.dtype)
        bias = bias.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))
        q, k = self.rope(q, time_id), self.rope(k, time_id)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=1 / math.sqrt(self.head_dim)
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, S, -1))


class GatedFeedForward(nn.Module):
    """A feed-forward block with a SiLU gate (SwiGLU) without bias.

    Args:
        d_model: Width of entry and exit.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        hidden = (int(4 * d_model * 2 / 3) + 7) // 8 * 8
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc_gate = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The output is the same shape as the `x`."""
        return self.fc2(F.silu(self.fc_gate(x)) * self.fc1(x))


class EncoderLayer(nn.Module):
    """Transformer layer with normalization to attention and to a fully
    connected unit.

    Args:
        d_model: Model width.
        num_heads: Number of heads.
        rope: General module RoPE.
    """

    def __init__(
        self, d_model: int, num_heads: int, rope: RotaryProjection
    ) -> None:
        super().__init__()
        self.self_attn = Attention(d_model, num_heads, rope)
        self.ffn = GatedFeedForward(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        time_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """The output layer is the same shape as the `x`."""
        x = x + self.self_attn(self.norm1(x), attn_mask, time_id, variate_id)
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    """Transformer layer stack with common RoPE and final normalization.

    Args:
        d_model: Model width.
        num_layers: Number of layers.
        max_seq_len: Table length RoPE.
    """

    def __init__(
        self, d_model: int, num_layers: int, max_seq_len: int
    ) -> None:
        super().__init__()
        num_heads = d_model // 64
        self.rope = RotaryProjection((d_model // num_heads) // 2, max_seq_len)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, self.rope)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        time_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> torch.Tensor:
        """Views of `(B, S, d)` after all layers."""
        for layer in self.layers:
            x = layer(x, attn_mask, time_id, variate_id)
        return self.norm(x)


class MultiInSizeLinear(nn.Module):
    """A linear layer with separate weights for each patch size, the input is
    supplemented with zeros to the largest.

    Args:
        in_sizes: Patch sizes.
        out_features: Exit width.
    """

    def __init__(self, in_sizes: Sequence[int], out_features: int) -> None:
        super().__init__()
        self.in_sizes = tuple(in_sizes)
        self.weight = nn.Parameter(
            torch.empty(len(in_sizes), out_features, max(in_sizes))
        )
        self.bias = nn.Parameter(torch.empty(len(in_sizes), out_features))
        for idx, size in enumerate(self.in_sizes):
            nn.init.kaiming_uniform_(
                self.weight[idx, :, :size], a=math.sqrt(5)
            )
            nn.init.zeros_(self.weight[idx, :, size:])
            nn.init.uniform_(
                self.bias[idx], -1 / math.sqrt(size), 1 / math.sqrt(size)
            )

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Output `(..., out_features)` for `x` `(..., P_max)` patches with
        `idx` size scales.
        """
        size = self.in_sizes[idx]
        return F.linear(
            x[..., :size], self.weight[idx, :, :size], self.bias[idx]
        )


class MultiOutSizeLinear(nn.Module):
    """A linear layer with separate weights for each patch size, the output
    reads as `(P_max, dim)`.

    Args:
        in_features: Width of entrance.
        patch_sizes: Patch sizes.
        dim: Number of parameters per patch point.
    """

    def __init__(
        self, in_features: int, patch_sizes: Sequence[int], dim: int
    ) -> None:
        super().__init__()
        self.patch_sizes = tuple(patch_sizes)
        self.dim = dim
        self.out_size = max(patch_sizes)
        width = dim * self.out_size
        self.weight = nn.Parameter(
            torch.empty(len(patch_sizes), width, in_features)
        )
        self.bias = nn.Parameter(torch.empty(len(patch_sizes), width))
        for idx, size in enumerate(self.patch_sizes):
            rows = dim * size
            nn.init.kaiming_uniform_(self.weight[idx, :rows], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, rows:])
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias[idx, :rows], -bound, bound)
            nn.init.zeros_(self.bias[idx, rows:])

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """`(..., P_max, dim)` settings for `x` `(..., in_features)`
        representations.

        The weights outside the first `dim * P` lines are zeroed, the offset
        acts entirely as in the
        original: the lines without weights remain a trainable constant.
        """
        rows = self.dim * self.patch_sizes[idx]
        out = (
            F.pad(
                F.linear(x, self.weight[idx, :rows]),
                (0, self.weight.shape[1] - rows),
            )
            + self.bias[idx]
        )
        return out.view(*x.shape[:-1], self.dim, self.out_size).transpose(
            -1, -2
        )


class ParamProj(nn.Module):
    """Projection of representations into the parameters of the mixture: logics
    of weights and parameters of components.

    Args:
        d_model: Model width.
        patch_sizes: Patch sizes.
        components: The names of the components of the mixture.
    """

    def __init__(
        self,
        d_model: int,
        patch_sizes: Sequence[int],
        components: Sequence[str],
    ) -> None:
        super().__init__()
        self.components = tuple(components)
        self.proj = nn.ModuleDict(
            {
                "weights_logits": MultiOutSizeLinear(
                    d_model, patch_sizes, len(components)
                ),
                "components": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                p: MultiOutSizeLinear(d_model, patch_sizes, 1)
                                for p in _COMPONENT_PARAMS[c]
                            }
                        )
                        for c in components
                    ]
                ),
            }
        )

    def forward(
        self, h: torch.Tensor, idx: int
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """`(B, S, P_max, K)` scale logs and a list of parameter dictionaries
        of `(B, S, P_max)` components, all float32.
        """
        logits = self.proj["weights_logits"](h, idx).float()
        params = []
        for name, proj in zip(
            self.components, self.proj["components"], strict=True
        ):
            raw = {
                p: proj[p](h, idx).float().squeeze(-1)
                for p in _COMPONENT_PARAMS[name]
            }
            params.append(_constrain(name, raw))
        return logits, params


def _constrain(
    name: str, raw: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    eps = torch.finfo(torch.float32).eps
    if name == "student_t":
        return {
            "df": 2.0 + F.softplus(raw["df"]),
            "loc": raw["loc"],
            "scale": F.softplus(raw["scale"]).clamp_min(eps),
        }
    if name == "normal_fixed":
        return {"loc": raw["loc"]}
    if name == "negative_binomial":
        return {
            "total_count": F.softplus(raw["total_count"]),
            "logits": raw["logits"],
        }
    return {
        "loc": raw["loc"],
        "scale": F.softplus(raw["scale"]).clamp_min(eps),
    }


class Mixture:
    r"""A predictive mixture for each patch point in the original series units.

    Components use the normalized value :math:`z=(y-\mu)/\sigma`.
    Subtract :math:`\log\sigma` from the log density in normalized units
    to obtain the log density in the original units.
    The lognormal component has support on positive values of :math:`z`.
    The negative binomial component uses the gamma extension for nonnegative
    noninteger values, matching the reference implementation. This extension
    does not define a normalized continuous density.

    Args:
        logits: `(B, S, O, K)` scale logs.
        params: The parameters of the components are in order, each tensor `(B,
            S, O)`.
        names: Component names.
        normal_fixed_scale: Standard deviation of the "normal_fixed" component.
        loc: Shift of `(B, S, 1)` normalization.
        scale: The scale of the normalization of `(B, S, 1)`.
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

    def _component_log_prob(
        self, name: str, p: dict[str, torch.Tensor], z: torch.Tensor
    ) -> torch.Tensor:
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
            return (
                -((z - p["loc"]) ** 2) / (2 * s * s)
                - math.log(s)
                - 0.5 * _LOG_2PI
            )
        if name == "negative_binomial":
            valid = z >= 0
            v = torch.where(valid, z, torch.ones_like(z))
            r = p["total_count"]
            lp = (
                r * F.logsigmoid(-p["logits"])
                + F.logsigmoid(p["logits"]) * v
                - (
                    torch.lgamma(1 + v)
                    + torch.lgamma(r)
                    - torch.lgamma(1 + v + r)
                    + torch.log(r + v)
                )
            )
            return torch.where(valid, lp, torch.full_like(lp, float("-inf")))
        valid = z > 0
        v = torch.where(valid, z, torch.ones_like(z))
        log_v = torch.log(v)
        lp = (
            -((log_v - p["loc"]) ** 2) / (2 * p["scale"] ** 2)
            - torch.log(p["scale"])
            - 0.5 * _LOG_2PI
            - log_v
        )
        return torch.where(valid, lp, torch.full_like(lp, float("-inf")))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log density `(B, S, O)` values `value` `(B, S, P)` in the initial
        units, in nats.

        With `P < O`, values are supplemented with zeros up to the width of
        `O`; the first `P` columns are meaningful.
        """
        width = self.logits.shape[-2]
        if value.shape[-1] < width:
            value = F.pad(value, (0, width - value.shape[-1]))
        z = ((value - self.loc) / self.scale).float()
        log_w = F.log_softmax(self.logits, dim=-1)
        comp = torch.stack(
            [
                self._component_log_prob(n, p, z)
                for n, p in zip(self.names, self.params, strict=True)
            ],
            dim=-1,
        )
        log_w = torch.where(torch.isinf(comp), torch.zeros_like(log_w), log_w)
        return torch.logsumexp(log_w + comp, dim=-1) - torch.log(self.scale)

    def sample(self, n: int) -> torch.Tensor:
        """Samples of `(n, B, S, O)` in initial units."""
        with torch.no_grad():
            dists = []
            for name, p in zip(self.names, self.params, strict=True):
                if name == "student_t":
                    dists.append(
                        torch.distributions.StudentT(
                            p["df"], p["loc"], p["scale"]
                        )
                    )
                elif name == "normal_fixed":
                    dists.append(
                        torch.distributions.Normal(
                            p["loc"], self.normal_fixed_scale
                        )
                    )
                elif name == "negative_binomial":
                    gamma = torch.distributions.Gamma(
                        p["total_count"], torch.exp(-p["logits"])
                    )
                    dists.append(_PoissonGamma(gamma))
                else:
                    dists.append(
                        torch.distributions.LogNormal(p["loc"], p["scale"])
                    )
            comps = torch.stack([d.sample((n,)) for d in dists], dim=-1)
            which = (
                torch.distributions.Categorical(logits=self.logits)
                .sample((n,))
                .unsqueeze(-1)
            )
            z = torch.gather(comps, -1, which).squeeze(-1)
            return z * self.scale + self.loc


class _PoissonGamma:
    """Negative binomial as a Poissonian mixture of gamma distributions, only
    sampling.
    """

    def __init__(self, gamma) -> None:
        self.gamma = gamma

    def sample(self, shape) -> torch.Tensor:
        """Integer samples of the form `shape + batch`."""
        return torch.poisson(self.gamma.sample(shape))


class PackedStdScaler(nn.Module):
    """Scale each series using the mean and unbiased standard deviation of
    observed context.

    Statistics use float64 and share only tokens with the same `sample_id`
    and `variate_id`. Padding tokens (`sample_id == 0`) have location zero
    and scale one.

    Args:
        minimum_scale: Variance offset inside the square root.
    """

    def __init__(self, minimum_scale: float = 1e-5) -> None:
        super().__init__()
        self.minimum_scale = minimum_scale

    def forward(
        self, target: torch.Tensor, observed_mask: torch.Tensor,
        sample_id: torch.Tensor, variate_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift and scale `(B, S, 1)` float32 for `target` `(B, S, P)`, masks
        `(B, S, P)` and indexes `(B, S)`.
        """
        x = target.double()
        same = (sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2)) & (
            variate_id.unsqueeze(-1) == variate_id.unsqueeze(-2)
        )
        same = same.to(x.dtype)
        obs = observed_mask.to(x.dtype)
        tobs = (same * obs.sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
        loc = (same * (x * obs).sum(-1).unsqueeze(-2)).sum(-1, keepdim=True)
        loc = loc / torch.where(tobs == 0, torch.ones_like(tobs), tobs)
        var = (same * (((x - loc) ** 2) * obs).sum(-1).unsqueeze(-2)).sum(
            -1, keepdim=True
        )
        denom = tobs - 1
        var = var / torch.where(denom == 0, torch.ones_like(denom), denom)
        scale = torch.sqrt(var + self.minimum_scale)
        pad = (sample_id == 0).unsqueeze(-1)
        loc = torch.where(pad, torch.zeros_like(loc), loc)
        scale = torch.where(pad, torch.ones_like(scale), scale)
        return loc.float(), scale.float()


class MoiraiModule(nn.Module):
    """Moirai model: series patches in views, transformer, mix of distributions
    on patch points.

    Args:
        config: Config.

    Attributes:
        config: Model configuration.
    """

    def __init__(self, config: MoiraiConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_encoding = nn.Embedding(1, config.d_model)
        self.scaler = PackedStdScaler() if config.scaling else None
        self.in_proj = MultiInSizeLinear(config.patch_sizes, config.d_model)
        self.encoder = Encoder(
            config.d_model, config.num_layers, config.max_seq_len
        )
        self.param_proj = ParamProj(
            config.d_model, config.patch_sizes, config.components
        )

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
        """Predictive mix for all batch tokens.

        Args:
            target: Values of `(B, S, P_max)` patches; for predictive and
                unobservable positions, any,
                including NaNHidden values are not transmitted to the model,
                the original tensor does not change.
            observed_mask: Boolean mask of observed values of `(B, S, P_max)`.
            sample_id: The number of the token series `(B, S)`, 0 is filling
                outside of attention and normalization.
            time_id: The time index of the `(B, S)` token is less than
                `max_seq_len`.
            variate_id: The index of variable token `(B, S)`.
            prediction_mask: Boolean mask of forecast tokens `(B, S)`.
            patch_size: The size of the patch batch from the
                `config.patch_sizes`.

        Returns:
            Mixture: Mixture with `(B, S, P_max, ...)` parameters in the
                original units of the series.
        """
        idx = self.config.patch_sizes.index(patch_size)
        if self.config.ignore_context:
            # Canonical null input: neither values, padding nor context length
            # can
            # influence the horizon. The caller retains the original batch for
            # NLL.
            target = torch.zeros_like(target)
            observed_mask = torch.zeros_like(observed_mask)
            sample_id = prediction_mask.to(sample_id.dtype)
            time_id = (prediction_mask.long().cumsum(-1) - 1).clamp_min(0)
            variate_id = torch.zeros_like(variate_id)
        context_mask = (
            observed_mask
            & ~prediction_mask.unsqueeze(-1)
            & (sample_id != 0).unsqueeze(-1)
        )
        # Missing values must not enter either scaling (NaN * 0 is NaN) or
        # partially observed patch projections. Keep the caller's labels
        # intact.
        target = target.masked_fill(~context_mask, 0)
        if self.config.fixed_scaling is not None:
            fixed_loc, fixed_scale = self.config.fixed_scaling
            # Missing inputs have normalized value zero, matching an external
            # fixed transform followed by identity scaling. Adaptive scaling
            # keeps the historical raw-zero convention for pretrained parity.
            target = target.masked_fill(~context_mask, fixed_loc)
            loc = torch.full(
                (*target.shape[:2], 1), fixed_loc, device=target.device
            )
            scale = torch.full_like(loc, fixed_scale)
        elif self.scaler is not None:
            loc, scale = self.scaler(
                target, context_mask, sample_id, variate_id
            )
        else:
            loc = torch.zeros(*target.shape[:2], 1, device=target.device)
            scale = torch.ones_like(loc)
        reprs = self.in_proj((target - loc) / scale, idx)
        reprs = torch.where(
            prediction_mask.unsqueeze(-1),
            self.mask_encoding.weight.to(reprs.dtype),
            reprs,
        )
        attn_mask = sample_id.unsqueeze(-1) == sample_id.unsqueeze(-2)
        h = self.encoder(reprs, attn_mask, time_id, variate_id)
        logits, params = self.param_proj(h, idx)
        return Mixture(
            logits,
            params,
            self.config.components,
            self.config.normal_fixed_scale,
            loc,
            scale,
        )


def load_uni2ts_checkpoint(module: MoiraiModule, path: str) -> None:
    """It loads weights from the `model.safetensors` checkpoint uni2ts into a
    module of the same configuration.

    Args:
        module: A model with the same patch sizes and components as a
            checkpoint.
        path: The path to the safetensors file.
    """
    from safetensors.torch import load_file

    module.load_state_dict(load_file(path), strict=True)
