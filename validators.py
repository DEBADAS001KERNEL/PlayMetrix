"""
PlayMetrix - Matrix Evaluation Layer (Layer 2: Evaluation)
Provides mathematical validation for matrix properties:
Row Echelon Form (REF), Reduced Row Echelon Form (RREF),
Rank, Pivot Detection, and Distance-to-Goal metrics.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class MatrixValidator:
    """
    Evaluation Layer:
    Mathematically verifies whether a matrix satisfies target properties,
    and computes distance/shaping metrics for RL reward calculation.
    """

    def __init__(self, tol: float = 1e-4):
        self.tol = tol

    def get_leading_entry_column(self, row: np.ndarray) -> Optional[int]:
        """
        Finds the column index of the first non-zero element in a row.
        Returns None if row is all zeros.
        """
        non_zeros = np.where(np.abs(row) > self.tol)[0]
        if len(non_zeros) == 0:
            return None
        return int(non_zeros[0])

    def get_pivots(self, M: np.ndarray) -> List[Tuple[int, int]]:
        """
        Returns list of (row, col) coordinates of all pivots in the matrix.
        """
        pivots = []
        for r in range(M.shape[0]):
            c = self.get_leading_entry_column(M[r])
            if c is not None:
                pivots.append((r, c))
        return pivots

    def is_row_echelon_form(self, M: np.ndarray) -> bool:
        """
        Validates if matrix M is in Row Echelon Form (REF).
        Requirements:
        1. All all-zero rows are at the bottom.
        2. The leading entry of each non-zero row is strictly to the right
           of the leading entry of the row above it.
        3. All entries in a column below a leading entry are zero.
        """
        rows, cols = M.shape
        last_leading_col = -1
        encountered_zero_row = False

        for r in range(rows):
            lead_col = self.get_leading_entry_column(M[r])
            if lead_col is None:
                # This is an all-zero row
                encountered_zero_row = True
            else:
                # If we previously saw an all-zero row, non-zero row below it is invalid
                if encountered_zero_row:
                    return False
                # Pivot must be strictly to the right of previous pivot
                if lead_col <= last_leading_col:
                    return False
                last_leading_col = lead_col

        return True

    def is_reduced_row_echelon_form(self, M: np.ndarray) -> bool:
        """
        Validates if matrix M is in Reduced Row Echelon Form (RREF).
        Requirements:
        1. Satisfies REF.
        2. Every leading entry is equal to 1.
        3. Each leading entry is the ONLY non-zero entry in its column.
        """
        if not self.is_row_echelon_form(M):
            return False

        pivots = self.get_pivots(M)
        for r, c in pivots:
            # Pivot must be 1.0
            if abs(M[r, c] - 1.0) > self.tol:
                return False
            # All other elements in this column must be 0
            col_vals = M[:, c].copy()
            col_vals[r] = 0.0
            if np.any(np.abs(col_vals) > self.tol):
                return False

        return True

    def count_ref_violations(self, M: np.ndarray) -> int:
        """
        Counts the number of structural violations preventing M from being in REF.
        Used as a heuristic distance metric for reinforcement learning reward shaping.
        """
        rows, cols = M.shape
        violations = 0
        pivots = []
        zero_rows = []

        for r in range(rows):
            lead_col = self.get_leading_entry_column(M[r])
            if lead_col is None:
                zero_rows.append(r)
            else:
                pivots.append((r, lead_col))

        # Check zero row positioning violations (non-zero row below zero row)
        if zero_rows:
            min_zero_row = min(zero_rows)
            for r, c in pivots:
                if r > min_zero_row:
                    violations += 2  # High penalty for zero row above non-zero row

        # Check column progression violations
        for i in range(len(pivots) - 1):
            r1, c1 = pivots[i]
            r2, c2 = pivots[i + 1]
            if c2 <= c1:
                violations += (c1 - c2 + 1)

        # Check non-zeros below pivots
        for r, c in pivots:
            for sub_r in range(r + 1, rows):
                if abs(M[sub_r, c]) > self.tol:
                    violations += 1

        return violations

    def matrix_norm_diff(self, M: np.ndarray) -> float:
        """Measure of overall matrix magnitude to penalize explosion."""
        return float(np.sum(np.abs(M)))

    def off_diagonal_norm(self, M: np.ndarray) -> float:
        """
        Computes the Frobenius norm of all off-diagonal entries:
        sqrt(sum_{i != j} M_{ij}^2)
        Direct measure of how far M is from being diagonal (eigen-basis).
        """
        assert M.shape[0] == M.shape[1], "Matrix must be square"
        M_diag_zero = M.copy()
        np.fill_diagonal(M_diag_zero, 0.0)
        return float(np.linalg.norm(M_diag_zero))

    def is_diagonal(self, M: np.ndarray, tol: Optional[float] = None) -> bool:
        """Checks if all off-diagonal entries are within tolerance of zero."""
        t = tol if tol is not None else self.tol
        return self.off_diagonal_norm(M) < t

    def validate_eigenpairs(
        self, A_orig: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, tol: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        Validates the fundamental definition of eigenvalues & eigenvectors:
        A * v_i = lambda_i * v_i  for all i
        Returns: (is_valid, max_residual_norm)
        """
        n = A_orig.shape[0]
        max_residual = 0.0

        for i in range(n):
            lam = eigenvalues[i]
            v = eigenvectors[:, i]
            # Normalize vector
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-9:
                v = v / v_norm

            residual = np.linalg.norm(A_orig @ v - lam * v)
            max_residual = max(max_residual, float(residual))

        is_valid = max_residual < tol
        return is_valid, max_residual
