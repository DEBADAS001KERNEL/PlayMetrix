"""
PlayMetrix - Multi-Heuristic Classical Baselines (Exact Arithmetic)
Provides standard textbook heuristics for matrix elimination:
1. Gaussian Elimination without Pivoting (GE-NoPivot)
2. Gaussian Elimination with Partial Pivoting (GE-PartialPivot)
3. Gaussian Elimination with Complete Pivoting (GE-CompletePivot)
4. Minimum Non-Zero Fill-in Greedy Heuristic (GE-MinFillin)
"""

from fractions import Fraction
from typing import List, Tuple, Dict, Any
import numpy as np

from playmetrix.engine.exact_ops import ExactMatrix


class ClassicalBaselines:
    """
    Executes various textbook Gaussian Elimination heuristics using exact arithmetic.
    """

    @staticmethod
    def ge_no_pivoting(M: ExactMatrix) -> Tuple[ExactMatrix, List[str], bool]:
        """
        Gaussian Elimination WITHOUT Pivoting:
        Only swaps if the diagonal entry is strictly 0.
        Otherwise directly eliminates down the column.
        """
        A = M.copy()
        rows, cols = A.rows, A.cols
        steps = []
        pivot_row = 0

        for c in range(cols):
            if pivot_row >= rows:
                break

            # If current diagonal is 0, find first non-zero below it
            if A.data[pivot_row][c] == 0:
                swap_target = None
                for r in range(pivot_row + 1, rows):
                    if A.data[r][c] != 0:
                        swap_target = r
                        break
                if swap_target is None:
                    continue  # Column is all zeros
                A = A.swap_rows(pivot_row, swap_target)
                steps.append(f"Swap R_{pivot_row+1} <-> R_{swap_target+1}")

            # Eliminate below
            pivot_val = A.data[pivot_row][c]
            for r in range(pivot_row + 1, rows):
                if A.data[r][c] != 0:
                    mult = -A.data[r][c] / pivot_val
                    A = A.add_row_multiple(r, pivot_row, mult)
                    steps.append(f"R_{r+1} <- R_{r+1} + ({mult})*R_{pivot_row+1}")

            pivot_row += 1

        is_ref = A.is_row_echelon_form()
        return A, steps, is_ref

    @staticmethod
    def ge_partial_pivoting(M: ExactMatrix) -> Tuple[ExactMatrix, List[str], bool]:
        """
        Standard Gaussian Elimination with Partial Pivoting:
        In each column, selects the row with the maximum absolute entry.
        """
        A = M.copy()
        rows, cols = A.rows, A.cols
        steps = []
        pivot_row = 0

        for c in range(cols):
            if pivot_row >= rows:
                break

            # Find row with maximum absolute entry in column c from pivot_row downwards
            best_row = pivot_row
            best_val = abs(A.data[pivot_row][c])

            for r in range(pivot_row + 1, rows):
                val = abs(A.data[r][c])
                if val > best_val:
                    best_val = val
                    best_row = r

            if best_val == 0:
                continue

            if best_row != pivot_row:
                A = A.swap_rows(pivot_row, best_row)
                steps.append(f"Swap R_{pivot_row+1} <-> R_{best_row+1}")

            pivot_val = A.data[pivot_row][c]
            for r in range(pivot_row + 1, rows):
                if A.data[r][c] != 0:
                    mult = -A.data[r][c] / pivot_val
                    A = A.add_row_multiple(r, pivot_row, mult)
                    steps.append(f"R_{r+1} <- R_{r+1} + ({mult})*R_{pivot_row+1}")

            pivot_row += 1

        is_ref = A.is_row_echelon_form()
        return A, steps, is_ref

    @staticmethod
    def ge_complete_pivoting(M: ExactMatrix) -> Tuple[ExactMatrix, List[str], bool]:
        """
        Gaussian Elimination with Complete (Full) Pivoting:
        Searches entire remaining submatrix for maximum element and swaps rows.
        """
        A = M.copy()
        rows, cols = A.rows, A.cols
        steps = []
        pivot_row = 0

        for col_idx in range(cols):
            if pivot_row >= rows:
                break

            # Search active submatrix [pivot_row:, col_idx:] for max element
            max_r = pivot_row
            max_c = col_idx
            max_val = Fraction(0)

            for r in range(pivot_row, rows):
                for c in range(col_idx, cols):
                    val = abs(A.data[r][c])
                    if val > max_val:
                        max_val = val
                        max_r = r
                        max_c = c

            if max_val == 0:
                continue

            # Row swap
            if max_r != pivot_row:
                A = A.swap_rows(pivot_row, max_r)
                steps.append(f"Swap R_{pivot_row+1} <-> R_{max_r+1}")

            pivot_val = A.data[pivot_row][max_c]
            # Eliminate below in column max_c
            for r in range(pivot_row + 1, rows):
                if A.data[r][max_c] != 0:
                    mult = -A.data[r][max_c] / pivot_val
                    A = A.add_row_multiple(r, pivot_row, mult)
                    steps.append(f"R_{r+1} <- R_{r+1} + ({mult})*R_{pivot_row+1}")

            pivot_row += 1

        is_ref = A.is_row_echelon_form()
        return A, steps, is_ref

    @staticmethod
    def ge_min_fillin(M: ExactMatrix) -> Tuple[ExactMatrix, List[str], bool]:
        """
        Markowitz-style minimum fill-in heuristic:
        Picks the pivot row that introduces the fewest non-zeros in the target rows.
        """
        A = M.copy()
        rows, cols = A.rows, A.cols
        steps = []
        pivot_row = 0

        for c in range(cols):
            if pivot_row >= rows:
                break

            # Candidates: all rows r >= pivot_row where A[r][c] != 0
            candidates = [r for r in range(pivot_row, rows) if A.data[r][c] != 0]
            if not candidates:
                continue

            # Score each candidate by row sparsity (count non-zeros)
            best_row = min(candidates, key=lambda r: sum(1 for x in A.data[r] if x != 0))

            if best_row != pivot_row:
                A = A.swap_rows(pivot_row, best_row)
                steps.append(f"Swap R_{pivot_row+1} <-> R_{best_row+1}")

            pivot_val = A.data[pivot_row][c]
            for r in range(pivot_row + 1, rows):
                if A.data[r][c] != 0:
                    mult = -A.data[r][c] / pivot_val
                    A = A.add_row_multiple(r, pivot_row, mult)
                    steps.append(f"R_{r+1} <- R_{r+1} + ({mult})*R_{pivot_row+1}")

            pivot_row += 1

        is_ref = A.is_row_echelon_form()
        return A, steps, is_ref
