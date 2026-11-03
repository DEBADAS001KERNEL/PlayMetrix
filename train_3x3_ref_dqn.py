"""
PlayMetrix - 3x3 Matrix Row Echelon Form (REF) DQN Experiment
Tests scaling the specialized REF agent to 3x3 matrices requiring multi-step
cascading row operations (swapping, multi-row elimination).
"""

import os
import sys
import time
from collections import deque
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from playmetrix.environments.ref_env import RowEchelonEnv
from playmetrix.agents.dqn_agent import DQNAgent
from playmetrix.engine.validators import MatrixValidator
from playmetrix.orchestrator.playmetrix_platform import PlayMetrixPlatform


def train_3x3_dqn(
    num_episodes: int = 1500,
    save_path: str = "checkpoints/ref_agent_3x3_dqn.pt"
):
    print("=" * 70)
    print(" PLAYMETRIX: SCALING TO 3x3 MATRICES (MULTI-STEP ROW ECHELON FORM)")
    print("=" * 70)
    rows, cols = 3, 3
    print(f"Matrix Dimensions : {rows} x {cols}")
    print(f"Total Episodes    : {num_episodes}")

    env = RowEchelonEnv(rows=rows, cols=cols, max_steps=12)
    state_dim = env.observation_dim
    num_actions = env.num_actions

    print(f"Observation Dim   : {state_dim}")
    print(f"Action Space Size : {num_actions} operations")

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=256,
        lr=7e-4,
        gamma=0.97,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay=0.996,
        buffer_capacity=30000,
        batch_size=64,
        target_update_freq=200
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

        is_success = info.get("is_ref", False)
        rewards_window.append(episode_reward)
        success_window.append(1.0 if is_success else 0.0)
        steps_window.append(steps)

        if episode % 100 == 0:
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

    agent.save(save_path)
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training Complete in {total_time:.2f} seconds!")
    print(f"Final 50-episode Success Rate: {np.mean(success_window)*100:.1f}%")
    print("=" * 70)

    return agent, env


def evaluate_3x3_benchmarks(agent: DQNAgent, env: RowEchelonEnv):
    print("\n" + "#" * 70)
    print(" EVALUATION ON 3x3 LINEAR ALGEBRA BENCHMARKS")
    print("#" * 70)

    validator = MatrixValidator()

    test_cases = [
        {
            "name": "3x3 Full System (Requires multi-row cascading elimination)",
            "matrix": np.array([
                [ 2.0,  1.0, -1.0],
                [-3.0, -1.0,  2.0],
                [-2.0,  1.0,  2.0]
            ])
        },
        {
            "name": "3x3 Zero Pivot at Top (Requires Swap + Elimination)",
            "matrix": np.array([
                [ 0.0,  2.0,  1.0],
                [ 1.0, -1.0,  3.0],
                [ 2.0,  0.0,  4.0]
            ])
        },
        {
            "name": "3x3 Triangular with Non-Zero in Corner",
            "matrix": np.array([
                [ 3.0,  2.0,  1.0],
                [ 0.0,  2.0,  4.0],
                [ 6.0,  0.0,  1.0]
            ])
        }
    ]

    action_meanings = env.get_action_meanings()

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- 3x3 Benchmark #{i}: {test['name']} ---")
        M = test["matrix"].copy()
        print("Initial Matrix:")
        for r in M:
            print(f"  [ {'  '.join(f'{x:6.2f}' for x in r)} ]")

        state = env.reset(matrix=M)
        done = False
        steps = 0
        path = []

        while not done and steps < 8:
            steps += 1
            action = agent.select_action(state, evaluate=True)
            act_desc = action_meanings.get(action, f"Op #{action}")
            next_state, reward, done, info = env.step(action)
            path.append((act_desc, info["matrix"].copy(), reward))
            state = next_state

        final_mat = path[-1][1] if path else M
        is_ref = validator.is_row_echelon_form(final_mat)

        print("\nAgent Solution Trace:")
        for s_idx, (act, mat, r) in enumerate(path, 1):
            print(f"  Step {s_idx}: {act}")
            for r_row in mat:
                print(f"          [ {'  '.join(f'{x:6.2f}' if abs(x) > 1e-4 else '  0.00' for x in r_row)} ]")

        if is_ref:
            print(f"Result: PASS [OK] (Achieved REF in {steps} steps)")
        else:
            print(f"Result: FAIL [X] (Did not achieve REF)")


def evaluate_3x3_random(agent: DQNAgent, env: RowEchelonEnv, count: int = 50):
    print(f"\nEvaluating 3x3 Agent on {count} Random Unseen Matrices...")
    validator = MatrixValidator()
    successes = 0
    step_counts = []

    for _ in range(count):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 8:
            steps += 1
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, info = env.step(action)
            state = next_state

        if validator.is_row_echelon_form(info["matrix"]):
            successes += 1
            step_counts.append(steps)

    print(f"Generalization Accuracy (3x3): {successes}/{count} ({successes/count*100:.1f}%)")
    if step_counts:
        print(f"Average Steps to REF: {np.mean(step_counts):.2f} steps (Min: {min(step_counts)}, Max: {max(step_counts)})")


if __name__ == "__main__":
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_file = os.path.join(checkpoint_dir, "ref_agent_3x3_dqn.pt")

    agent, env = train_3x3_dqn(num_episodes=1200, save_path=ckpt_file)
    evaluate_3x3_benchmarks(agent, env)
    evaluate_3x3_random(agent, env, count=50)
