# PlayMetrix — Multi-Agent Matrix Intelligence

> **An extensible multi-agent reinforcement learning platform for learning, optimizing, and discovering mathematical matrix transformations.**

---

## 🏛️ The 3-Layer Architecture

PlayMetrix cleanly decouples decision-making, mathematical execution, and validation into three independent layers:

```
┌────────────────────────────────────────────────────────┐
│                   AGENT LAYER                          │
│  Specialized RL Agents (REF, RREF, LU, Eigen, SVD...)  │
│  - Decides WHICH elementary action to perform           │
│  - Deep Q-Network (DQN) with experience replay         │
└──────────────────────────┬─────────────────────────────┘
                           │ Action Choice (e.g. R2 <- R2 - 2*R1)
                           ▼
┌────────────────────────────────────────────────────────┐
│                 EXECUTION LAYER                        │
│  MatrixOperationEngine                                 │
│  - Executes verified elementary row operations         │
│  - Swaps, additions, scalings, pivot eliminations      │
│  - Maintains immutable mathematical audit trail        │
└──────────────────────────┬─────────────────────────────┘
                           │ Resulting Matrix State
                           ▼
┌────────────────────────────────────────────────────────┐
│                 EVALUATION LAYER                       │
│  MatrixValidator & Reward System                       │
│  - Checks mathematical invariants (REF, RREF, Pivots)  │
│  - Computes violation distances and dense rewards      │
│  - Informs agent of success / termination              │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Results: REF DQN Agent

We trained and evaluated specialized DQN agents on reducing arbitrary matrices to **Row Echelon Form (REF)**:

| Experiment | Matrix Size | Action Space | Training Time | Convergence Episode | Benchmark Pass Rate | Generalization (Random Unseen) |
|---|---|---|---|---|---|---|
| **REF 2x3 Agent** | $2 \times 3$ | 15 EROs | **3.64 seconds** | Episode 100 (100%) | **5/5 (100.0%)** | **100/100 (100.0%)** |
| **REF 3x3 Agent** | $3 \times 3$ | 45 EROs | **20.39 seconds**| Episode 200 (98%)  | **3/3 (100.0%)** | **48/50 (96.0%)** |

---

## 📁 Project Structure

```
playmetrix/
├── engine/
│   ├── matrix_ops.py          # Layer 1: MatrixOperationEngine (Row operations & history)
│   └── validators.py          # Layer 2: MatrixValidator (REF/RREF checkers & metrics)
├── environments/
│   ├── matrix_env.py          # Base Matrix Environment interface
│   └── ref_env.py             # Row Echelon Form RL Environment
├── agents/
│   ├── base_agent.py          # Base Agent interface
│   └── dqn_agent.py           # Layer 3: Deep Q-Network Agent (PyTorch)
├── orchestrator/
│   └── playmetrix_platform.py # Unified Mathematical Interface & Orchestrator
├── checkpoints/
│   ├── ref_agent_dqn.pt       # Trained 2x3 REF Agent weights
│   └── ref_agent_3x3_dqn.pt   # Trained 3x3 REF Agent weights
├── train_ref_dqn.py           # Training & benchmark script (2x3)
├── train_3x3_ref_dqn.py       # Training & benchmark script (3x3)
├── demo_interactive.py        # Interactive trace demonstrator
└── README.md
```

---

## 💻 How to Run

1. **Interactive Demo (Live Step-by-Step Trajectory)**:
   ```powershell
   python demo_interactive.py
   ```

2. **Train & Benchmark 2x3 REF Agent**:
   ```powershell
   python train_ref_dqn.py
   ```

3. **Train & Benchmark 3x3 REF Agent**:
   ```powershell
   python train_3x3_ref_dqn.py
   ```
