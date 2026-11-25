"""Shared randomness, finite ordered random coding, and Elias-delta framing."""

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import torch

from ..model.moirai import Mixture


def _keyed_seed(seed: int, stream: str, step: int, window: int, index: int) -> int:
    key = f"{seed}/{stream}/{step}/{window}/{index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def seeded_draw(
    law: Mixture,
    seed: int,
    stream: str,
    step: int,
    window: int = 0,
    index: int = 0,
    count: int = 1,
) -> torch.Tensor:
    """Sample in original units while preserving the caller's random streams."""
    seed = _keyed_seed(seed, stream, step, window, index)
    device = law.logits.device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(seed)
        try:
            return law.sample(count)
        except ValueError as exc:
            raise FloatingPointError(
                f"invalid predictive law for {stream} draw at step {step + 1}"
            ) from exc


def window_law(law: Mixture, i: int) -> Mixture:
    row = slice(i, i + 1)
    return Mixture(
        law.logits[row],
        [{k: v[row] for k, v in p.items()} for p in law.params],
        law.names,
        law.normal_fixed_scale,
        law.loc[row],
        law.scale[row],
    )


def proposal(law: Mixture, seed: int, step: int, window: int, index: int) -> torch.Tensor:
    return seeded_draw(law, seed, "proposal", step, window, index)[0]


def ordered_index(log_ratios: Sequence[float], exponentials: Sequence[float]) -> int:
    """Return the one-based winner of the finite ORC race (Theis and Yosri, Alg. 4)."""
    n = len(log_ratios)
    arrivals = np.cumsum(np.asarray(exponentials, dtype=np.float64) * n / np.arange(n, 0, -1))
    return int(np.argmin(np.log(arrivals) - log_ratios)) + 1


def arrival_increment(seed: int, step: int, window: int, index: int) -> float:
    # Map 52 random bits to an open-interval uniform using bin midpoints.
    bits = _keyed_seed(seed, "arrival", step, window, index) >> 12
    return -math.log((bits + 0.5) * 2**-52)


def pack_indices(indices: Sequence[int]) -> tuple[bytes, int]:
    """Concatenate positive Elias-delta integers, MSB first, and zero-pad the last byte."""
    codes = []
    for index in indices:
        if index < 1:
            raise ValueError("Elias-delta indices start at one")
        value = f"{index:b}"
        length = f"{len(value):b}"
        codes.append("0" * (len(length) - 1) + length + value[1:])
    bits = "".join(codes)
    padded = bits + "0" * (-len(bits) % 8)
    return bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8)), len(bits)


def unpack_indices(payload: bytes, bit_length: int, candidates: int) -> list[int]:
    """Validate byte framing, zero padding, complete codewords, and pool bounds."""
    if bit_length < 0 or len(payload) != (bit_length + 7) // 8:
        raise ValueError("invalid ORC payload length")
    bits = "".join(f"{byte:08b}" for byte in payload)
    if "1" in bits[bit_length:]:
        raise ValueError("nonzero ORC padding")
    indices = []
    pos = 0
    while pos < bit_length:
        one = bits.find("1", pos, bit_length)
        if one < 0 or one - pos >= candidates.bit_length().bit_length():
            raise ValueError("invalid Elias-delta prefix")
        end = one + (one - pos) + 1
        if end > bit_length:
            raise ValueError("truncated Elias-delta length")
        width = int(bits[one:end], 2)
        pos = end + width - 1
        if pos > bit_length:
            raise ValueError("truncated Elias-delta index")
        index = int("1" + bits[end:pos], 2)
        if index > candidates:
            raise ValueError("index exceeds the ORC candidate pool")
        indices.append(index)
    return indices
