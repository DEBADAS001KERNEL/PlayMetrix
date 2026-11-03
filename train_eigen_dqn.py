"""
PlayMetrix - Specialized Eigen DQN Training & Benchmark Script
Trains a dedicated RL Agent to compute Eigenvalues and Eigenvectors
via Orthogonal Similarity Rotations.
Compares steps against classical Jacobi and records discoveries in the History Ledger.
"""

import os
import sys
import time
from collections import deque
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from playmetrix.environments.eigen_env import EigenEnv
from playmetrix.agents.dqn_agent import DQNAgent
from playmetrix.engine.validators import MatrixValidator
from playmetrix.engine.matrix_ops import MatrixOperationEngine
from playmetrix.engine.history import DiscoveryLedger


def train_eigen_agent(
    dim: int = 2,
    num_episodes: int = 600,
    save_path: str = "checkpoints/eigen_agent_dqn.pt"
):
    print("=" * 70)
    print(" PLAYMETRIX: TRAINING SPECIALIZED EIGENVALUE & EIGENVECTOR AGENT")
    print("=" * 70)
    print(f"Matrix Dimension  : {dim} x {dim}")
    print(f"Total Episodes    : {num_episodes}")

    env = EigenEnv(dim=dim, max_steps=8)
    state_dim = env.observation_dim
    num_actions = env.num_actions

    print(f"Observation Dim   : {state_dim}")
    print(f"Action Space Size : {num_actions} similarity rotation operations")

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=128,
        lr=8e-4,
        gamma=0.96,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay=0.992,
        buffer_capacity=15000,
        batch_size=64,
        target_update_freq=150
    )

    rewards_window = deque(maxlen=50)
    success_window = deque(maxlen=50)
    steps_window = deque(maxlen=50)

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            episode_reward += reward
            steps += 1

        is_success = info.get("is_diagonal", False)
        rewards_window.append(episode_reward)
        success_window.append(1.0 if is_success else 0.0)
        steps_window.append(steps)

        if episode % 50 == 0:
            avg_rew = np.mean(rewards_window)
            avg_succ = np.mean(success_window) * 100
            avg_steps = np.mean(steps_window)
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:4d}/{num_episodes:4d} | "
                f"Success Rate (last 50): {avg_succ:5.1f}% | "
                f"Avg Reward: {avg_rew:6.2f} | "
                f"Avg Steps: {avg_steps:4.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    agent.save(save_path)
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training Complete in {total_time:.2f} seconds!")
    print(f"Final Success Rate: {np.mean(success_window)*100:.1f}%")
    print(f"Model saved to: {os.path.abspath(save_path)}")
    print("=" * 70)

    return agent, env


def evaluate_eigen_benchmarks(agent: DQNAgent, env: EigenEnv):
    print("\n" + "#" * 70)
    print(" EVALUATION ON CLASSICAL EIGENVALUE BENCHMARKS & DISCOVERY AUDIT")
    print("#" * 70)

    validator = MatrixValidator()
    engine = MatrixOperationEngine()
    ledger = DiscoveryLedger()

    test_cases = [
        {
            "name": "Standard Symmetric 2x2 (Known lambda = [3.0, 1.0])",
            "matrix": np.array([[2.0, 1.0],
                                [1.0, 2.0]])
        },
        {
            "name": "Rank-1 Degenerate Symmetric (Known lambda = [5.0, 0.0])",
            "matrix": np.array([[4.0, 2.0],
                                [2.0, 1.0]])
        },
        {
            "name": "Negative Off-Diagonal (Known lambda = [8.0, 2.0])",
            "matrix": np.array([[5.0, -3.0],
                                [-3.0, 5.0]])
        },
        {
            "name": "Diagonal Matrix (Zero Off-Diagonal - 0 Steps Needed)",
            "matrix": np.array([[3.0, 0.0],
                                [0.0, -1.0]])
        }
    ]

    action_meanings = env.get_action_meanings()

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Benchmark #{i}: {test['name']} ---")
        M = test["matrix"].copy()
        print("Initial Matrix A:")
        for r in M:
            print(f"  [ {'  '.join(f'{x:6.2f}' for x in r)} ]")

        # Classical baseline
        c_vals, c_vecs, c_steps, c_history = engine.classical_jacobi_eigen(M)

        # Agent Solution
        state = env.reset(matrix=M)
        done = False
        steps = 0
        agent_actions = []

        while not done and steps < 6:
            steps += 1
            action = agent.select_action(state, evaluate=True)
            act_desc = action_meanings.get(action, f"Op #{action}")
            agent_actions.append(act_desc)
            next_state, reward, done, info = env.step(action)
            state = next_state

        final_mat = info["matrix"]
        eigenvalues = info["eigenvalues"]
        eigenvectors = info["eigenvectors"]
        is_valid, residual = validator.validate_eigenpairs(M, eigenvalues, eigenvectors)

        print("\nAgent Solution:")
        print(f"  Steps Taken   : {steps} (Classical Jacobi took: {c_steps} steps)")
        print(f"  Actions       : {', '.join(agent_actions)}")
        print(f"  Eigenvalues : {np.round(eigenvalues, 4)}")
        print(f"  Eigenvectors V (columns):")
        for row in eigenvectors:
            print(f"    [ {'  '.join(f'{x:7.4f}' for x in row)} ]")
        print(f"  Eigen Invariant (A*v == lambda*v) : {'VALID [OK]' if is_valid else 'INVALID [X]'} (Residual: {residual:.2e})")

        # Log to ledger
        entry = ledger.log_solution(
            task="EIGEN",
            initial_matrix=M,
            final_matrix=final_mat,
            agent_steps=steps,
            classical_steps=c_steps,
            agent_actions=agent_actions,
            classical_actions=c_history,
            is_valid=is_valid,
            metadata={"eigenvalues": eigenvalues.tolist(), "residual": residual}
        )

        if entry["is_novel_shorter"]:
            print(f"  [DISCOVERY] Agent found a SHORTER sequence ({steps} < {c_steps})!")
        elif entry["is_tied"]:
            print(f"  [TIED] Matched classical optimal efficiency ({steps} steps).")

    summary = ledger.get_summary()
    print("\n" + "=" * 70)
    print(f"DISCOVERY LEDGER SUMMARY: {summary['total']} tasks recorded | "
          f"Faster: {summary['faster_than_classical']} | "
          f"Tied: {summary['tied_with_classical']} | "
          f"Efficiency Rate: {summary['efficiency_rate']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_file = os.path.join(ckpt_dir, "eigen_agent_dqn.pt")

    agent, env = train_eigen_agent(dim=2, num_episodes=500, save_path=save_file)
    evaluate_eigen_benchmarks(agent, env)
