"""
PlayMetrix - Run Scientific Discovery Campaign
Executes a large-scale automated discovery campaign comparing trained RL agents
against classical algorithms to find and prove shorter matrix transformation sequences.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from playmetrix.engine.discovery_lab import ScientificDiscoveryLab
from playmetrix.environments.ref_env import RowEchelonEnv
from playmetrix.agents.dqn_agent import DQNAgent


def run_campaign():
    print("=" * 75)
    print(" PLAYMETRIX: SCIENTIFIC DISCOVERY CAMPAIGN (AI vs CLASSICAL BASELINES)")
    print("=" * 75)
    print("Searching for matrix configurations where RL Agent discovers")
    print("strictly shorter transformation sequences than standard textbook algorithms...")
    print("=" * 75)

    lab = ScientificDiscoveryLab()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    all_discoveries = []

    # 1. Test 3x3 REF Agent (where multi-step cascading elimination allows algorithmic shortcuts)
    print("\n[Phase 1] Searching in 3x3 Matrix Space (Complex cascading eliminations)...")
    env_3x3 = RowEchelonEnv(rows=3, cols=3, max_steps=10)
    agent_3x3 = DQNAgent(state_dim=env_3x3.observation_dim, num_actions=env_3x3.num_actions, hidden_dim=256)
    ckpt_3x3 = os.path.join(base_dir, "checkpoints", "ref_agent_3x3_dqn.pt")
    if os.path.exists(ckpt_3x3):
        agent_3x3.load(ckpt_3x3)
        print("  Loaded trained 3x3 Agent.")
        disc_3x3 = lab.run_ref_discovery_campaign(agent_3x3, env_3x3, num_trials=250)
        print(f"  Found {len(disc_3x3)} cases where RL Agent beat classical Gaussian elimination!")
        all_discoveries.extend(disc_3x3)

    # 2. Test 2x3 REF Agent
    print("\n[Phase 2] Searching in 2x3 Matrix Space...")
    env_2x3 = RowEchelonEnv(rows=2, cols=3, max_steps=8)
    agent_2x3 = DQNAgent(state_dim=env_2x3.observation_dim, num_actions=env_2x3.num_actions, hidden_dim=128)
    ckpt_2x3 = os.path.join(base_dir, "checkpoints", "ref_agent_dqn.pt")
    if os.path.exists(ckpt_2x3):
        agent_2x3.load(ckpt_2x3)
        print("  Loaded trained 2x3 Agent.")
        disc_2x3 = lab.run_ref_discovery_campaign(agent_2x3, env_2x3, num_trials=150)
        print(f"  Found {len(disc_2x3)} cases where RL Agent beat classical Gaussian elimination!")
        all_discoveries.extend(disc_2x3)

    # Save discoveries JSON
    disc_file = os.path.join(base_dir, "checkpoints", "novel_discoveries.json")
    with open(disc_file, "w", encoding="utf-8") as f:
        json.dump(all_discoveries, f, indent=2)
    print(f"\nSaved {len(all_discoveries)} discoveries to: {disc_file}")

    # Generate Research Paper Draft
    report_file = os.path.join(base_dir, "DISCOVERY_RESEARCH_REPORT.md")
    lab.generate_research_report(all_discoveries, output_file=report_file)
    print(f"Generated formal research report at: {report_file}")

    print("\n" + "=" * 75)
    print(f"CAMPAIGN COMPLETE: Total Novel Discoveries Found = {len(all_discoveries)}")
    print("=" * 75)

    if all_discoveries:
        best = max(all_discoveries, key=lambda d: d["steps_saved"])
        print("\nTOP DISCOVERY HIGHLIGHT:")
        print(f"Task: {best['task']}")
        print(f"Classical Gaussian Steps : {best['classical_steps']}")
        print(f"PlayMetrix Agent Steps   : {best['agent_steps']} (Saved {best['steps_saved']} steps! +{best['efficiency_gain_pct']}%)")
        print("Agent Operations Sequence:")
        for idx, act in enumerate(best["agent_trajectory"], 1):
            print(f"  {idx}. {act}")
        print("Classical Operations Sequence:")
        for idx, act in enumerate(best["classical_trajectory"], 1):
            print(f"  {idx}. {act}")


if __name__ == "__main__":
    run_campaign()
