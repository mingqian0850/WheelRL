import numpy as np

from tutorials.rl.examples.gae_walkthrough import compute_gae, discounted_returns
from tutorials.rl.examples.summarize_success import (
    interquartile_mean,
    wilson_interval,
)
from tutorials.rl.examples.tabular_q_learning import (
    Corridor,
    evaluate,
    train_q_learning,
)


def test_corridor_separates_terminal_and_time_limit() -> None:
    rng = np.random.default_rng(1)
    terminal_env = Corridor(length=2, slip_probability=0.0, max_steps=1)
    terminal_env.reset()
    _, _, terminated, truncated = terminal_env.step(1, rng)
    assert terminated
    assert not truncated

    timeout_env = Corridor(length=3, slip_probability=0.0, max_steps=1)
    timeout_env.reset()
    _, _, terminated, truncated = timeout_env.step(1, rng)
    assert not terminated
    assert truncated


def test_q_learning_learns_short_deterministic_corridor() -> None:
    env = Corridor(length=5, slip_probability=0.0, max_steps=10)
    q_values = train_q_learning(
        env,
        episodes=500,
        alpha=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        seed=3,
    )
    success_rate, _ = evaluate(env, q_values, episodes=20, seed=4)
    assert np.all(np.argmax(q_values[:-1], axis=1) == 1)
    assert success_rate == 1.0


def test_gae_lambda_one_matches_monte_carlo_at_terminal() -> None:
    rewards = np.array([0.1, 0.2, 1.0])
    values = np.array([0.4, 0.5, 0.6, 0.0])
    terminated = np.array([False, False, True])
    advantages, returns = compute_gae(
        rewards,
        values,
        terminated,
        gamma=0.99,
        gae_lambda=1.0,
    )
    expected = discounted_returns(rewards, gamma=0.99)
    np.testing.assert_allclose(returns, expected)
    np.testing.assert_allclose(advantages, expected - values[:-1])


def test_success_summary_statistics() -> None:
    values = np.array([0.1, 0.2, 0.8, 0.9])
    assert interquartile_mean(values) == 0.5
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high
