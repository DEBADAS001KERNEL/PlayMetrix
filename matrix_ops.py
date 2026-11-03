"""
PlayMetrix - Shared Matrix Execution Engine (Layer 1: Execution)
Executes exact mathematical operations, tracks transformation histories,
and records elementary transformation matrices.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class MatrixOperationEngine:
    """
    Execution Layer:
    Performs verified mathematical operations on matrices and records
    every step in an immutable transformation history.
    """

    def __init__(self, record_history: bool = True):
        self.record_history = record_history
        self.history: List[Dict[str, Any]] = []

    def reset_history(self) -> None:
        self.history.clear()

    def swap_rows(self, M: np.ndarray, r1: int, r2: int) -> np.ndarray:
        """
        Elementary Row Operation 1: Swap row r1 and row r2.
        R_{r1} <-> R_{r2}
        """
        assert 0 <= r1 < M.shape[0] and 0 <= r2 < M.shape[0], f"Invalid row indices: {r1}, {r2}"
        if r1 == r2:
            return M.copy()

        M_new = M.copy()
        M_new[[r1, r2]] = M_new[[r2, r1]]

        if self.record_history:
            self.history.append({
                "op": "swap",
                "r1": r1,
                "r2": r2,
                "notation": f"R_{r1+1} <-> R_{r2+1}",
                "result_state": M_new.copy()
            })
        return M_new

    def add_row_multiple(self, M: np.ndarray, target: int, source: int, scalar: float) -> np.ndarray:
        """
        Elementary Row Operation 2: Add scalar * source row to target row.
        R_{target} <- R_{target} + scalar * R_{source}
        """
        assert 0 <= target < M.shape[0] and 0 <= source < M.shape[0], f"Invalid indices: {target}, {source}"
        assert target != source, "Target and source row must be different"

        M_new = M.copy()
        M_new[target] = M_new[target] + scalar * M_new[source]

        # Clean numerical noise near zero
        M_new[np.abs(M_new) < 1e-12] = 0.0

        if self.record_history:
            sign = "+" if scalar >= 0 else "-"
            scalar_abs = abs(scalar)
            if scalar_abs == 1.0:
                scalar_str = ""
            elif scalar_abs == int(scalar_abs):
                scalar_str = f"{int(scalar_abs)}"
            else:
                scalar_str = f"{scalar_abs:.2f}"

            notation = f"R_{target+1} <- R_{target+1} {sign} {scalar_str}R_{source+1}"
            self.history.append({
                "op": "add_multiple",
                "target": target,
                "source": source,
                "scalar": scalar,
                "notation": notation,
                "result_state": M_new.copy()
            })
        return M_new

    def scale_row(self, M: np.ndarray, target: int, scalar: float) -> np.ndarray:
        """
        Elementary Row Operation 3: Multiply row by a non-zero scalar.
        R_{target} <- scalar * R_{target}
        """
        assert 0 <= target < M.shape[0], f"Invalid target index: {target}"
        assert abs(scalar) > 1e-9, "Scalar cannot be zero"

        M_new = M.copy()
        M_new[target] = M_new[target] * scalar
        M_new[np.abs(M_new) < 1e-12] = 0.0

        if self.record_history:
            notation = f"R_{target+1} <- {scalar:.2f} * R_{target+1}"
            self.history.append({
                "op": "scale",
                "target": target,
                "scalar": scalar,
                "notation": notation,
                "result_state": M_new.copy()
            })
        return M_new

    def similarity_rotation(
        self, A: np.ndarray, V: np.ndarray, p: int, q: int, theta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Similarity Transformation via Givens / Jacobi Orthogonal Rotation:
        A_new = R^T * A * R
        V_new = V * R  (Accumulates eigenvectors)
        Preserves all eigenvalues exactly since R is orthogonal (R^T * R = I).
        """
        n = A.shape[0]
        assert A.shape == (n, n), "Matrix must be square for similarity rotation"
        assert 0 <= p < q < n, f"Invalid rotation plane indices: p={p}, q={q}"

        c = float(np.cos(theta))
        s = float(np.sin(theta))

        # Build Givens rotation matrix R
        R = np.eye(n, dtype=float)
        R[p, p] = c
        R[q, q] = c
        R[p, q] = -s
        R[q, p] = s

        A_new = R.T @ A @ R
        V_new = V @ R

        # Clean numerical noise
        A_new[np.abs(A_new) < 1e-12] = 0.0
        V_new[np.abs(V_new) < 1e-12] = 0.0

        if self.record_history:
            deg = np.degrees(theta)
            self.history.append({
                "op": "similarity_rotation",
                "plane": (p, q),
                "theta_deg": deg,
                "notation": f"Rotate plane ({p+1}, {q+1}) by {deg:.1f}°",
                "result_state": A_new.copy()
            })

        return A_new, V_new

    def classical_jacobi_eigen(
        self, A: np.ndarray, tol: float = 1e-4, max_iter: int = 25
    ) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
        """
        Classical Cyclic Jacobi Eigenvalue Algorithm for benchmarking.
        Iteratively zeroes out off-diagonal elements in cyclic row order.
        Returns: (eigenvalues, eigenvectors, steps_taken, steps_history)
        """
        n = A.shape[0]
        M = A.astype(float).copy()
        # Make symmetric for Jacobi diagonalization if slightly asymmetric
        if not np.allclose(M, M.T, atol=1e-5):
            M = 0.5 * (M + M.T)

        V = np.eye(n, dtype=float)
        steps_history = []
        step = 0

        for it in range(max_iter):
            # Check off-diagonal norm
            off_diag = np.sqrt(np.sum(M**2) - np.sum(np.diag(M)**2))
            if off_diag < tol:
                break

            for p in range(n):
                for q in range(p + 1, n):
                    if abs(M[p, q]) > tol:
                        step += 1
                        # Classical Jacobi angle
                        if abs(M[p, p] - M[q, q]) < 1e-12:
                            theta = np.pi / 4.0 if M[p, q] > 0 else -np.pi / 4.0
                        else:
                            theta = 0.5 * np.arctan2(2.0 * M[p, q], M[p, p] - M[q, q])

                        M, V = self.similarity_rotation(M, V, p, q, theta)
                        steps_history.append(f"Rotate plane ({p+1}, {q+1}) by {np.degrees(theta):.1f}°")

        eigenvalues = np.diag(M)
        return eigenvalues, V, step, steps_history

    def deterministic_gaussian_elimination(self, M: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Standard classical Gaussian elimination for benchmarking and verification.
        Returns the row echelon form and list of steps taken.
        """
        A = M.astype(float).copy()
        rows, cols = A.shape
        steps = []
        pivot_row = 0

        for col in range(cols):
            if pivot_row >= rows:
                break

            max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))
            if np.abs(A[max_row, col]) < 1e-9:
                continue

            if max_row != pivot_row:
                A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
                steps.append(f"Swap R_{pivot_row+1} <-> R_{max_row+1}")

            for r in range(pivot_row + 1, rows):
                if np.abs(A[r, col]) > 1e-9:
                    factor = A[r, col] / A[pivot_row, col]
                    A[r] = A[r] - factor * A[pivot_row]
                    A[np.abs(A) < 1e-12] = 0.0
                    steps.append(f"R_{r+1} <- R_{r+1} - ({factor:.2f})*R_{pivot_row+1}")

            pivot_row += 1

        return A, steps

