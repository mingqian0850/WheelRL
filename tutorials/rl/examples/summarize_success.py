"""Summarize success counts from independent RL training seeds."""

from __future__ import annotations

import argparse
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def interquartile_mean(values: FloatArray) -> float:
    """Mean of the central 50% of an empirical distribution.

    Fractional endpoint weights make the definition work for any sample count.
    """

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    if n == 0:
        raise ValueError("values cannot be empty")
    lower, upper = 0.25, 0.75
    weighted_sum = 0.0
    for index, value in enumerate(ordered):
        left, right = index / n, (index + 1) / n
        overlap = max(0.0, min(right, upper) - max(left, lower))
        weighted_sum += overlap * float(value)
    return weighted_sum / (upper - lower)


def wilson_interval(
    successes: int,
    episodes: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if episodes <= 0 or not 0 <= successes <= episodes:
        raise ValueError("successes must be in [0, episodes]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / episodes
    denominator = 1.0 + z**2 / episodes
    center = (proportion + z**2 / (2.0 * episodes)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / episodes
            + z**2 / (4.0 * episodes**2)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def seed_bootstrap_interval(
    rates: FloatArray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rates), size=(samples, len(rates)))
    means = rates[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(means, [tail, 1.0 - tail]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successes", type=int, nargs="+", required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="One shared episode count or one count per training seed.",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    successes = np.asarray(args.successes, dtype=np.int64)
    episodes = np.asarray(args.episodes, dtype=np.int64)
    if len(episodes) == 1:
        episodes = np.repeat(episodes, len(successes))
    if episodes.shape != successes.shape:
        raise ValueError("provide one episode count or one count per seed")
    if np.any(episodes <= 0) or np.any(successes < 0) or np.any(successes > episodes):
        raise ValueError("each success count must be in [0, episodes]")

    rates = successes / episodes
    seed_low, seed_high = seed_bootstrap_interval(
        rates,
        samples=args.bootstrap,
        seed=args.seed,
    )
    pooled_low, pooled_high = wilson_interval(
        int(successes.sum()),
        int(episodes.sum()),
    )
    print(f"seed_rates={np.round(rates, 4).tolist()}")
    print(
        f"across_seed mean={rates.mean():.4f} "
        f"median={np.median(rates):.4f} "
        f"iqm={interquartile_mean(rates):.4f} "
        f"bootstrap_95ci=[{seed_low:.4f}, {seed_high:.4f}]"
    )
    print(
        "pooled_episode_wilson_95ci="
        f"[{pooled_low:.4f}, {pooled_high:.4f}] "
        "(ignores between-policy training variation)"
    )


if __name__ == "__main__":
    main()
