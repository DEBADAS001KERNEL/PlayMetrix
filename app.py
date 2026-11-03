"""
PlayMetrix - Multi-Agent Matrix Intelligence Web Application
Interactive platform for specialized RL agents (REF, Eigen, etc.)
Includes mathematical validation, classical benchmarking, and discovery history tracking.
"""

import os
import sys
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from playmetrix.engine.matrix_ops import MatrixOperationEngine
from playmetrix.engine.validators import MatrixValidator
from playmetrix.engine.history import DiscoveryLedger
from playmetrix.environments.ref_env import RowEchelonEnv
from playmetrix.environments.eigen_env import EigenEnv
from playmetrix.agents.dqn_agent import DQNAgent

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PlayMetrix — Multi-Agent Matrix Intelligence",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .badge-action {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: bold;
        background-color: #e3f2fd;
        color: #0d47a1;
        font-family: monospace;
    }
    .discovery-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Model & Engine Caching
# ---------------------------------------------------------
@st.cache_resource
def load_agent_and_env(task_type: str) -> Tuple[DQNAgent, Any]:
    """Loads specialized environment and trained DQN agent weights."""
    if task_type == "REF (2x3)":
        env = RowEchelonEnv(rows=2, cols=3, max_steps=12)
        agent = DQNAgent(state_dim=env.observation_dim, num_actions=env.num_actions, hidden_dim=128)
        weights_name = "ref_agent_dqn.pt"
    elif task_type == "REF (3x3)":
        env = RowEchelonEnv(rows=3, cols=3, max_steps=12)
        agent = DQNAgent(state_dim=env.observation_dim, num_actions=env.num_actions, hidden_dim=256)
        weights_name = "ref_agent_3x3_dqn.pt"
    else:  # Eigen Agent (2x2)
        env = EigenEnv(dim=2, max_steps=8)
        agent = DQNAgent(state_dim=env.observation_dim, num_actions=env.num_actions, hidden_dim=128)
        weights_name = "eigen_agent_dqn.pt"

    ckpt_path = os.path.join(CURRENT_DIR, "checkpoints", weights_name)
    if os.path.exists(ckpt_path):
        agent.load(ckpt_path)
    return agent, env


def matrix_to_latex(M: np.ndarray) -> str:
    """Converts a numpy matrix to clean LaTeX representation."""
    lines = []
    for row in M:
        row_str = " & ".join([f"{val:.2f}".rstrip('0').rstrip('.') if abs(val) >= 1e-4 else "0" for val in row])
        lines.append(row_str)
    content = r" \\ ".join(lines)
    return rf"\begin{{bmatrix}} {content} \end{{bmatrix}}"


def parse_matrix_text(text: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Parses text into a numpy matrix."""
    try:
        clean_text = text.strip().replace(";", "\n").replace("[", "").replace("]", "")
        rows_str = [r.strip() for r in clean_text.split("\n") if r.strip()]
        matrix_data = []
        for r in rows_str:
            elements = [float(x.strip()) for x in r.replace(",", " ").split() if x.strip()]
            matrix_data.append(elements)
        mat = np.array(matrix_data, dtype=np.float32)
        if mat.shape != target_shape:
            return None
        return mat
    except Exception:
        return None


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
validator = MatrixValidator()
engine = MatrixOperationEngine()
ledger = DiscoveryLedger()

with st.sidebar:
    st.markdown("### 🧮 PlayMetrix Multi-Agent Suite")
    st.markdown("---")

    task_choice = st.selectbox(
        "Select Specialized Agent:",
        ["REF (2x3)", "REF (3x3)", "Eigen Agent (2x2)"],
        help="Loads the trained RL agent specialized for this mathematical problem."
    )

    if task_choice == "REF (2x3)":
        expected_rows, expected_cols = 2, 3
        is_eigen = False
    elif task_choice == "REF (3x3)":
        expected_rows, expected_cols = 3, 3
        is_eigen = False
    else:
        expected_rows, expected_cols = 2, 2
        is_eigen = True

    st.markdown("#### 🏛️ 3-Layer Architecture")
    st.markdown("""
    - **Agent Layer**: Specialized DQN Policy
    - **Execution Layer**: MatrixOperationEngine
    - **Evaluation Layer**: MatrixValidator
    """)

    st.markdown("---")
    max_steps = st.slider("Maximum Step Limit:", min_value=3, max_value=15, value=10)
    compare_classical = st.checkbox("Compare with Classical Baseline", value=True)

    agent, env = load_agent_and_env(task_choice)
    st.info(f"Loaded **{task_choice}**\n\n• State Features: `{env.observation_dim}`\n• Discrete Actions: `{env.num_actions}`")

    # History summary in sidebar
    st.markdown("---")
    summary = ledger.get_summary()
    st.markdown(f"**📚 Total Solved History:** `{summary['total']}` tasks")
    st.markdown(f"- Shorter than Classical: `{summary['faster_than_classical']}`")
    st.markdown(f"- Tied with Classical: `{summary['tied_with_classical']}`")


# ---------------------------------------------------------
# Main Page Content
# ---------------------------------------------------------
st.markdown('<div class="main-header">PlayMetrix — Multi-Agent Matrix Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Specialized Reinforcement Learning agents learning to solve mathematical matrix transformations.</div>', unsafe_allow_html=True)

main_tab1, main_tab2 = st.tabs(["🚀 Agent Workspace & Solver", "📜 Discovery & Benchmarking Ledger"])

with main_tab1:
    # ---------------------------------------------------------
    # Step 1: Matrix Input Form
    # ---------------------------------------------------------
    st.markdown(f"### 1. Matrix Input ({'Symmetric Matrix for Eigenvalues' if is_eigen else f'{expected_rows}x{expected_cols} Matrix'})")

    input_tab1, input_tab2, input_tab3, input_tab4 = st.tabs([
        "⚡ Quick Presets",
        "📝 Interactive Grid Form",
        "📋 Paste Matrix Text / CSV",
        "🎲 Random Generator"
    ])

    selected_matrix = None

    with input_tab1:
        if task_choice == "REF (2x3)":
            presets = {
                "Classic Gaussian Elimination (Needs R2 - 2*R1)": np.array([[2.0, 1.0, 4.0], [4.0, 5.0, 6.0]]),
                "Zero Pivot at Top (Requires Row Swap R1 <-> R2)": np.array([[0.0, 3.0, 6.0], [2.0, 1.0, 4.0]]),
                "Linearly Dependent Row (Generates all-zero bottom row)": np.array([[1.0, -2.0, 3.0], [3.0, -6.0, 9.0]]),
                "Negative & Opposite Coefficients": np.array([[-3.0, 2.0, 1.0], [6.0, 1.0, 5.0]])
            }
        elif task_choice == "REF (3x3)":
            presets = {
                "3x3 Full Linear System (Multi-row cascading elimination)": np.array([
                    [ 2.0,  1.0, -1.0],
                    [-3.0, -1.0,  2.0],
                    [-2.0,  1.0,  2.0]
                ]),
                "3x3 Zero Pivot at Top (Requires Swap + Elimination)": np.array([
                    [ 0.0,  2.0,  1.0],
                    [ 1.0, -1.0,  3.0],
                    [ 2.0,  0.0,  4.0]
                ]),
                "3x3 Corner Elimination": np.array([
                    [ 3.0,  2.0,  1.0],
                    [ 0.0,  2.0,  4.0],
                    [ 6.0,  0.0,  1.0]
                ])
            }
        else:  # Eigen 2x2
            presets = {
                "Standard Symmetric 2x2 (Known λ = [3.0, 1.0])": np.array([[2.0, 1.0], [1.0, 2.0]]),
                "Rank-1 Degenerate Symmetric (Known λ = [5.0, 0.0])": np.array([[4.0, 2.0], [2.0, 1.0]]),
                "Negative Off-Diagonal (Known λ = [8.0, 2.0])": np.array([[5.0, -3.0], [-3.0, 5.0]]),
                "Identity Matrix (Already Diagonal, 0 steps needed)": np.array([[3.0, 0.0], [0.0, -1.0]])
            }
        preset_name = st.selectbox("Choose a benchmark preset:", list(presets.keys()))
        selected_matrix = presets[preset_name].copy()

    with input_tab2:
        st.markdown(f"**Enter values for your {expected_rows}x{expected_cols} matrix:**")
        grid_vals = np.zeros((expected_rows, expected_cols), dtype=np.float32)
        for r in range(expected_rows):
            row_cols = st.columns(expected_cols)
            for c in range(expected_cols):
                default_val = float(selected_matrix[r, c]) if selected_matrix is not None else 1.0
                grid_vals[r, c] = row_cols[c].number_input(
                    f"M[{r+1},{c+1}]",
                    value=float(default_val),
                    step=1.0,
                    key=f"grid_{task_choice}_{r}_{c}"
                )
        if st.button("Apply Grid Matrix", key="apply_grid"):
            if is_eigen and not np.allclose(grid_vals, grid_vals.T):
                grid_vals = 0.5 * (grid_vals + grid_vals.T)
                st.info("Symmetrized matrix for orthogonal diagonalization.")
            selected_matrix = grid_vals.copy()

    with input_tab3:
        st.markdown(f"Paste your {expected_rows} rows separated by newlines, with elements space/comma-separated:")
        default_text = "\n".join([" ".join([f"{int(x)}" for x in row]) for row in selected_matrix])
        pasted_text = st.text_area("Matrix Text:", value=default_text, height=120)

        uploaded_file = st.file_uploader("Or upload CSV/TXT matrix file:", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode("utf-8")
                parsed = parse_matrix_text(content, (expected_rows, expected_cols))
                if parsed is not None:
                    selected_matrix = parsed
                    st.success("Uploaded matrix successfully parsed!")
                else:
                    st.error(f"Uploaded file does not match required shape ({expected_rows}x{expected_cols})")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            parsed = parse_matrix_text(pasted_text, (expected_rows, expected_cols))
            if parsed is not None:
                selected_matrix = parsed

    with input_tab4:
        col_r1, col_r2 = st.columns(2)
        rand_min = col_r1.number_input("Minimum Integer:", value=-4, step=1)
        rand_max = col_r2.number_input("Maximum Integer:", value=4, step=1)
        if st.button("🎲 Generate Random Matrix"):
            selected_matrix = env.generate_random_matrix(low=int(rand_min), high=int(rand_max) + 1)
            st.success("Generated new random matrix!")

    # ---------------------------------------------------------
    # Step 2: Evaluation Layer Validation
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 2. Matrix Inspection (Evaluation Layer)")

    col_disp1, col_disp2 = st.columns([1, 1.2])

    with col_disp1:
        st.markdown("**Current Matrix:**")
        st.latex(rf"A_0 = {matrix_to_latex(selected_matrix)}")

    with col_disp2:
        st.markdown("**Initial Property Check:**")
        if not is_eigen:
            is_initial_done = validator.is_row_echelon_form(selected_matrix)
            violations = validator.count_ref_violations(selected_matrix)
            pivots = validator.get_pivots(selected_matrix)
            if is_initial_done:
                st.success("✓ Matrix is ALREADY in Row Echelon Form (REF)!")
            else:
                st.warning(f"⚠ Matrix is NOT in REF ({violations} structural violation{'s' if violations != 1 else ''})")
            st.markdown(f"- **Detected Pivots**: `{pivots if pivots else 'None'}`")
        else:
            off_norm = validator.off_diagonal_norm(selected_matrix)
            is_initial_done = validator.is_diagonal(selected_matrix)
            if is_initial_done:
                st.success("✓ Matrix is ALREADY diagonal (Eigenvalues on diagonal)!")
            else:
                st.warning(f"⚠ Matrix is NOT diagonal (Off-diagonal norm: `{off_norm:.4f}`)")
            st.markdown(f"- **Initial Trace (Sum of λ)**: `{float(np.trace(selected_matrix)):.2f}`")

    # ---------------------------------------------------------
    # Step 3: Solve with Specialized RL Agent
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 3. Solution Trace ({task_choice})")

    if st.button(f"🚀 Solve Matrix with {task_choice}", type="primary", use_container_width=True):
        with st.spinner("Agent is selecting mathematical transformations..."):
            state = env.reset(matrix=selected_matrix)
            action_meanings = env.get_action_meanings()

            trace: List[Dict[str, Any]] = []
            done = False
            steps = 0
            total_reward = 0.0

            while not done and steps < max_steps:
                steps += 1
                q_vals = agent.get_action_q_values(state)
                action = int(np.argmax(q_vals))
                action_desc = action_meanings.get(action, f"Action #{action}")

                next_state, reward, done, info = env.step(action)
                total_reward += reward

                trace.append({
                    "step": steps,
                    "action": action_desc,
                    "matrix": info["matrix"].copy(),
                    "reward": reward,
                    "info": info,
                    "q_vals": q_vals,
                    "chosen_action": action
                })
                state = next_state

            final_info = trace[-1]["info"] if trace else {}

            # Classical comparison calculations
            if not is_eigen:
                c_mat, c_actions = engine.deterministic_gaussian_elimination(selected_matrix)
                c_steps = len(c_actions)
                success = validator.is_row_echelon_form(trace[-1]["matrix"])
            else:
                c_vals, c_vecs, c_steps, c_actions = engine.classical_jacobi_eigen(selected_matrix)
                success = validator.is_diagonal(trace[-1]["matrix"])

            # Log to History Ledger
            ledger_entry = ledger.log_solution(
                task=task_choice,
                initial_matrix=selected_matrix,
                final_matrix=trace[-1]["matrix"],
                agent_steps=steps,
                classical_steps=c_steps,
                agent_actions=[t["action"] for t in trace],
                classical_actions=c_actions,
                is_valid=success,
                metadata={
                    "total_reward": total_reward,
                    "eigenvalues": final_info.get("eigenvalues", []).tolist() if is_eigen else None
                }
            )

            # Results Banner
            if success:
                st.success(f"### 🎉 SUCCESS! Task completed in {steps} step{'s' if steps != 1 else ''}!")
                if ledger_entry["is_novel_shorter"]:
                    st.markdown(f"<div class='discovery-badge'>★ NOVEL DISCOVERY: Agent solved this in {steps} steps vs Classical Baseline ({c_steps} steps)!</div>", unsafe_allow_html=True)
                elif ledger_entry["is_tied"]:
                    st.info(f"✨ Matched the optimal classical baseline ({steps} steps)!")
            else:
                st.error(f"### ✗ Terminated after {steps} steps without reaching goal.")

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Agent Steps", f"{steps}")
            m2.metric("Classical Steps", f"{c_steps}")
            m3.metric("Cumulative Reward", f"{total_reward:+.2f}")
            m4.metric("Mathematical Invariant", "VERIFIED ✓" if success else "NOT MET ✗")

            # Display Eigenvalues and Eigenvectors if Eigen task
            if is_eigen and success:
                e_vals = final_info["eigenvalues"]
                e_vecs = final_info["eigenvectors"]
                st.markdown("#### 🎯 Computed Eigenvalues & Orthogonal Eigenvectors:")
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    st.markdown(r"**Eigenvalues ($\lambda$):**")
                    st.code(f"λ_1 = {e_vals[0]:.4f}\nλ_2 = {e_vals[1]:.4f}")
                with ev_col2:
                    st.markdown("**Eigenvectors Matrix $V$ (columns are eigenvectors):**")
                    st.latex(rf"V = {matrix_to_latex(e_vecs)}")

                is_valid_inv, res = validator.validate_eigenpairs(selected_matrix, e_vals, e_vecs)
                st.markdown(rf"**Eigen-Equation Verification ($A v_i = \lambda_i v_i$):** `{'PASSED [Residual: ' + f'{res:.2e}]' if is_valid_inv else 'FAILED'}`")

            # Step-by-Step Visualization
            st.markdown("#### Step-by-Step Transformation Trace:")
            for idx, t_item in enumerate(trace, 1):
                prev_mat = selected_matrix if idx == 1 else trace[idx - 2]["matrix"]
                curr_mat = t_item["matrix"]

                with st.expander(f"Step {t_item['step']}: {t_item['action']}", expanded=True):
                    col_step_a, col_step_b, col_step_c = st.columns([1, 1.2, 1])
                    with col_step_a:
                        st.markdown("**Before Operation:**")
                        st.latex(matrix_to_latex(prev_mat))
                    with col_step_b:
                        st.markdown(f"**Action Selected (Agent Layer):**\n\n<span class='badge-action'>{t_item['action']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Step Reward:** `{t_item['reward']:+.3f}`")
                    with col_step_c:
                        st.markdown("**Resulting Matrix (Execution Layer):**")
                        st.latex(matrix_to_latex(curr_mat))

                    # Q-values inspection
                    q_vals = t_item["q_vals"]
                    top_indices = np.argsort(q_vals)[::-1][:3]
                    st.markdown("**🧠 Agent Decision Confidence (Top-3 Q-Values):**")
                    q_cols = st.columns(3)
                    for rank, act_idx in enumerate(top_indices):
                        is_chosen = (act_idx == t_item["chosen_action"])
                        q_cols[rank].caption(f"#{rank+1} {'(CHOSEN)' if is_chosen else ''}: {action_meanings.get(act_idx, f'Op {act_idx}')}")
                        q_cols[rank].progress(min(1.0, max(0.0, float(q_vals[act_idx]) / 35.0)))
                        q_cols[rank].code(f"Q = {q_vals[act_idx]:.3f}")

            # Final Matrix
            st.markdown("#### Final Matrix State:")
            st.latex(rf"M_{{\text{{final}}}} = {matrix_to_latex(trace[-1]['matrix'])}")


# ---------------------------------------------------------
# Tab 2: Discovery & Benchmarking Ledger
# ---------------------------------------------------------
with main_tab2:
    st.markdown("### 📜 PlayMetrix Discovery & Benchmarking Ledger")
    st.markdown("Every matrix solved is archived with step comparison against classical numerical algorithms.")

    summary = ledger.get_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs Archived", summary["total"])
    c2.metric("Shorter Than Classical (Discoveries)", summary["faster_than_classical"])
    c3.metric("Matched Classical (Tied)", summary["tied_with_classical"])
    c4.metric("Efficiency Rate", f"{summary['efficiency_rate']:.1f}%")

    st.markdown("---")
    records = ledger.records[::-1]  # Newest first

    if not records:
        st.info("No runs recorded yet. Solve a matrix in the Agent Workspace to log your first run!")
    else:
        for rec in records[:15]:
            is_faster = rec.get("is_novel_shorter", False)
            is_tied = rec.get("is_tied", False)
            status_text = "★ SHORTER THAN CLASSICAL" if is_faster else ("TIED OPTIMAL" if is_tied else "COMPLETED")

            with st.expander(f"Run #{rec['id']} | Task: {rec['task']} | {status_text} (Agent: {rec['agent_steps']} vs Classical: {rec['classical_steps']})", expanded=is_faster):
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("**Initial Matrix:**")
                    st.latex(matrix_to_latex(np.array(rec["initial_matrix"])))
                    st.markdown(f"**Agent Steps:** `{rec['agent_steps']}` | **Classical Steps:** `{rec['classical_steps']}`")
                    st.markdown(f"**Step Difference:** `{rec['step_difference']:+d}` {'(Agent Faster!)' if is_faster else ''}")
                with col_h2:
                    st.markdown("**Final Result Matrix:**")
                    st.latex(matrix_to_latex(np.array(rec["final_matrix"])))
                    st.markdown(f"**Agent Actions:** `{', '.join(rec['agent_actions'])}`")
                    st.markdown(f"**Timestamp:** `{rec['timestamp']}`")
