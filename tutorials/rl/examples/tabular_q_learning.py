"""Original minimal Q-learning example for the WheelRL RL tutorial.

The environment is a short stochastic corridor. It deliberately exposes
``terminated`` and ``truncated`` separately so the bootstrap mask is visible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class Corridor:
    """A tiny episodic MDP with left/right actions and stochastic slipping."""

    length: int = 7
    slip_probability: float = 0.10
    max_steps: int = 30

    def reset(self) -> int:
        self.state = 0
        self.steps = 0
        return self.state

    def step(
        self,
        action: int,
        rng: np.random.Generator,
    ) -> tuple[int, float, bool, bool]:
        if action not in (0, 1):
            raise ValueError("action must be 0 (left) or 1 (right)")
        executed_action = 1 - action if rng.random() < self.slip_probability else action
        displacement = -1 if executed_action == 0 else 1
        self.state = int(np.clip(self.state + displacement, 0, self.length - 1))
        self.steps += 1

        terminated = self.state == self.length - 1
        truncated = self.steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        return self.state, reward, terminated, truncated


def linear_schedule(start: float, end: float, fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return start + fraction * (end - start)


def train_q_learning(
    env: Corridor,
    *,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    seed: int,
) -> FloatArray:
    """Train a tabular off-policy Q learner."""

    rng = np.random.default_rng(seed)
    q_values = np.zeros((env.length, 2), dtype=np.float64)

    for episode in range(episodes):
        state = env.reset()
        epsilon = linear_schedule(
            epsilon_start,
            epsilon_end,
            episode / max(episodes - 1, 1),
        )

        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(2))
            else:
                # Random tie-breaking prevents the initial all-zero table from
                # deterministically preferring the left action.
                best = np.flatnonzero(q_values[state] == q_values[state].max())
                action = int(rng.choice(best))

            next_state, reward, terminated, truncated = env.step(action, rng)
            # A time-limit truncation is not an MDP terminal, so it retains the
            # bootstrap term. A true terminal cuts it off.
            bootstrap = 0.0 if terminated else float(q_values[next_state].max())
            target = reward + gamma * bootstrap
            q_values[state, action] += alpha * (target - q_values[state, action])
            state = next_state

            if terminated or truncated:
                break

    return q_values


def evaluate(
    env: Corridor,
    q_values: FloatArray,
    *,
    episodes: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    successes = 0
    returns: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        episode_return = 0.0
        while True:
            action = int(np.argmax(q_values[state]))
            state, reward, terminated, truncated = env.step(action, rng)
            episode_return += reward
            if terminated:
                successes += 1
            if terminated or truncated:
                break
        returns.append(episode_return)

    return successes / episodes, float(np.mean(returns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=4_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--length", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--slip-probability", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.eval_episodes <= 0:
        raise ValueError("episode counts must be positive")
    if args.length < 2 or args.max_steps <= 0:
        raise ValueError("length must be >= 2 and max_steps must be positive")
    if not 0.0 <= args.slip_probability <= 1.0:
        raise ValueError("slip probability must be in [0, 1]")
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")

    env = Corridor(
        length=args.length,
        slip_probability=args.slip_probability,
        max_steps=args.max_steps,
    )
    q_values = train_q_learning(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
    )
    success_rate, mean_return = evaluate(
        env,
        q_values,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
    )
    arrows = np.array(["←", "→"])
    greedy_policy = "".join(arrows[np.argmax(q_values[:-1], axis=1)])

    np.set_printoptions(precision=3, suppress=True)
    print(f"greedy_policy={greedy_policy} (goal is to the right)")
    print(f"q_values=\n{q_values}")
    print(
        f"evaluation episodes={args.eval_episodes} "
        f"success_rate={success_rate:.3f} mean_return={mean_return:.3f}"
    )


if __name__ == "__main__":
    main()
