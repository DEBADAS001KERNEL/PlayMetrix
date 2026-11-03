"""
PlayMetrix - Interactive Matrix Demonstration
Demonstrates the 3-Layer Architecture solving user-defined or random matrices
using the trained specialized DQN REF Agent.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from playmetrix.orchestrator.playmetrix_platform import PlayMetrixPlatform


def run_demo():
    print("=" * 70)
    print(" PLAYMETRIX: MULTI-AGENT MATRIX INTELLIGENCE PLATFORM")
    print("=" * 70)
    print("Demonstrating the 3-Layer Multi-Agent Architecture:")
    print("  [Agent Layer]      : Decides mathematical action via Deep Q-Network")
    print("  [Execution Layer]  : Executes verified elementary row operations")
    print("  [Evaluation Layer] : Validates Row Echelon Form & assigns rewards")
    print("=" * 70)

    platform = PlayMetrixPlatform()
    ckpt_path = os.path.join(os.path.dirname(__file__), "checkpoints", "ref_agent_dqn.pt")

    # Load REF Task & Agent
    platform.load_task("REF", weights_path=ckpt_path if os.path.exists(ckpt_path) else None)

    # Example 1: Classic Gaussian elimination
    m1 = np.array([
        [2.0, 1.0, 4.0],
        [4.0, 5.0, 6.0]
    ])
    platform.solve(m1, render=True)

    # Example 2: Zero pivot requiring swap
    m2 = np.array([
        [0.0, 4.0, -2.0],
        [3.0, 1.0,  5.0]
    ])
    platform.solve(m2, render=True)

    # Example 3: Linearly dependent row (creating zero row at bottom)
    m3 = np.array([
        [1.0, -2.0, 3.0],
        [3.0, -6.0, 9.0]
    ])
    platform.solve(m3, render=True)


if __name__ == "__main__":
    run_demo()
