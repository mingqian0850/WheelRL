"""Numerically expand generalized advantage estimation (GAE)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def compute_gae(
    rewards: FloatArray,
    values: FloatArray,
    terminated: BoolArray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Return advantages and value targets.

    ``values`` has length ``T + 1`` and contains the bootstrap value after the
    final collected transition. Only true MDP termination masks bootstrap.
    """

    if values.shape != (len(rewards) + 1,):
        raise ValueError("values must have length len(rewards) + 1")
    if terminated.shape != rewards.shape:
        raise ValueError("terminated must have the same shape as rewards")

    advantages = np.zeros_like(rewards, dtype=np.float64)
    next_advantage = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        nonterminal = 1.0 - float(terminated[t])
        delta = rewards[t] + gamma * nonterminal * values[t + 1] - values[t]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[t] = next_advantage
    returns = advantages + values[:-1]
    return advantages, returns


def discounted_returns(
    rewards: FloatArray,
    *,
    gamma: float,
) -> FloatArray:
    result = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        result[t] = running
    return result


def main() -> None:
    rewards = np.array([0.0, 0.2, 1.0], dtype=np.float64)
    values = np.array([0.40, 0.50, 0.60, 0.0], dtype=np.float64)
    terminated = np.array([False, False, True])
    gamma = 0.99

    print("true-terminal trajectory")
    for gae_lambda in (0.0, 0.95, 1.0):
        advantages, returns = compute_gae(
            rewards,
            values,
            terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        print(
            f"lambda={gae_lambda:>4}: "
            f"advantages={np.round(advantages, 4)} "
            f"returns={np.round(returns, 4)}"
        )

    advantages_one, returns_one = compute_gae(
        rewards,
        values,
        terminated,
        gamma=gamma,
        gae_lambda=1.0,
    )
    monte_carlo = discounted_returns(rewards, gamma=gamma)
    np.testing.assert_allclose(returns_one, monte_carlo)
    np.testing.assert_allclose(advantages_one, monte_carlo - values[:-1])

    # The collected rollout may end at a time limit while the MDP continues.
    # In that case the last value is a valid bootstrap estimate.
    timeout_values = np.array([0.40, 0.50, 0.60, 0.70], dtype=np.float64)
    no_true_terminal = np.array([False, False, False])
    _, timeout_returns = compute_gae(
        rewards,
        timeout_values,
        no_true_terminal,
        gamma=gamma,
        gae_lambda=1.0,
    )
    expected_first_return = (
        rewards[0]
        + gamma * rewards[1]
        + gamma**2 * rewards[2]
        + gamma**3 * timeout_values[-1]
    )
    np.testing.assert_allclose(timeout_returns[0], expected_first_return)
    print(
        "time-limit trajectory: "
        f"bootstrapped_returns={np.round(timeout_returns, 4)}"
    )
    print("checks=passed")


if __name__ == "__main__":
    main()
