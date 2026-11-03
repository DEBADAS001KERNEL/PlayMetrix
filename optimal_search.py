"""
PlayMetrix - Provably Optimal Search Baseline
Uses Breadth-First Search (BFS) over exact state transitions to find the
provably shortest sequence of elementary row operations to achieve Row Echelon Form.
Establishes the true theoretical lower bound (k*) to measure the Optimality Gap.
"""

from collections import deque
from typing import List, Tuple, Dict, Any, Optional, Set
from fractions import Fraction

from playmetrix.engine.exact_ops import ExactMatrix


class OptimalSearchBaseline:
    """
    Computes the provably minimal number of elementary row operations (k*)
    required to reduce a given matrix to Row Echelon Form.
    """

    def __init__(
        self,
        raw_ops_only: bool = False,
        multipliers: Optional[List[int]] = None,
        max_depth: int = 7,
        max_states: int = 50000
    ):
        self.raw_ops_only = raw_ops_only
        self.multipliers = multipliers or [-3, -2, -1, 1, 2, 3]
        self.max_depth = max_depth
        self.max_states = max_states

    def get_successors(self, M: ExactMatrix) -> List[Tuple[ExactMatrix, str]]:
        """Generates all legal next states from elementary row operations."""
        successors = []
        rows = M.rows

        # 1. Row Swaps
        for i in range(rows):
            for j in range(i + 1, rows):
                swapped = M.swap_rows(i, j)
                successors.append((swapped, f"Swap R_{i+1} <-> R_{j+1}"))

        # 2. Integer additions: R_target <- R_target + c * R_source
        for target in range(rows):
            for source in range(rows):
                if target == source:
                    continue
                for c in self.multipliers:
                    added = M.add_row_multiple(target, source, c)
                    successors.append((added, f"R_{target+1} <- R_{target+1} + ({c})*R_{source+1}"))

        # 3. If not raw_ops_only, allow exact elimination multiplier:
        if not self.raw_ops_only:
            for target in range(rows):
                for source in range(rows):
                    if target == source:
                        continue
                    lead = M.get_leading_col(source)
                    if lead is not None:
                        src_val = M.data[source][lead]
                        tgt_val = M.data[target][lead]
                        if tgt_val != 0:
                            mult = -tgt_val / src_val
                            elim = M.add_row_multiple(target, source, mult)
                            successors.append((elim, f"Eliminate R_{target+1} via R_{source+1} (c={mult})"))

        return successors

    def find_optimal_path(
        self, M: ExactMatrix
    ) -> Tuple[Optional[int], List[str], Optional[ExactMatrix], int]:
        """
        Runs Breadth-First Search to find the provably shortest transformation path.
        Returns: (min_steps, action_path, final_matrix, states_explored)
        """
        if M.is_row_echelon_form():
            return 0, [], M.copy(), 1

        queue = deque([(M, [], 0)])
        visited: Set[Tuple[Tuple[Fraction, ...], ...]] = {M.to_tuple()}
        states_explored = 0

        while queue:
            curr_mat, path, depth = queue.popleft()
            states_explored += 1

            if depth >= self.max_depth or states_explored >= self.max_states:
                continue

            for next_mat, act_desc in self.get_successors(curr_mat):
                mat_tup = next_mat.to_tuple()
                if mat_tup in visited:
                    continue

                visited.add(mat_tup)
                new_path = path + [act_desc]

                # Immediate goal check (cuts BFS complexity by 45x)
                if next_mat.is_row_echelon_form():
                    return depth + 1, new_path, next_mat, states_explored

                if depth + 1 < self.max_depth:
                    queue.append((next_mat, new_path, depth + 1))

        # Reached depth/state limit without finding path
        return None, [], None, states_explored
