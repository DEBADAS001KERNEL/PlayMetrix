"""
PlayMetrix - Exact Rational Arithmetic Engine
Uses Python's fractions.Fraction to guarantee 100% exact symbolic computation
with zero floating-point rounding errors or precision drift.
"""

from fractions import Fraction
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np


class ExactMatrix:
    """
    Exact Rational Matrix representation where every entry is an exact Fraction.
    Eliminates all floating-point thresholding and numerical tolerance issues.
    """

    def __init__(self, data: Union[List[List[Any]], np.ndarray, 'ExactMatrix']):
        if isinstance(data, ExactMatrix):
            self.rows = data.rows
            self.cols = data.cols
            self.data = [[Fraction(x) for x in row] for row in data.data]
        elif isinstance(data, np.ndarray):
            self.rows, self.cols = data.shape
            self.data = [[Fraction(int(x)) if float(x).is_integer() else Fraction(str(x)) for x in row] for row in data]
        else:
            self.rows = len(data)
            self.cols = len(data[0]) if self.rows > 0 else 0
            self.data = [[Fraction(x) for x in row] for row in data]

    def copy(self) -> 'ExactMatrix':
        return ExactMatrix(self.data)

    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        return np.array([[float(x) for x in row] for row in self.data], dtype=dtype)

    def to_tuple(self) -> Tuple[Tuple[Fraction, ...], ...]:
        """Immutable hashable representation for search state graphs."""
        return tuple(tuple(row) for row in self.data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExactMatrix) or self.rows != other.rows or self.cols != other.cols:
            return False
        return self.data == other.data

    def __hash__(self) -> int:
        return hash(self.to_tuple())

    def __repr__(self) -> str:
        lines = []
        for row in self.data:
            lines.append("  [ " + "  ".join(f"{str(x):>6}" for x in row) + " ]")
        return "ExactMatrix([\n" + "\n".join(lines) + "\n])"

    def is_zero_row(self, r: int) -> bool:
        return all(x == 0 for x in self.data[r])

    def get_leading_col(self, r: int) -> Optional[int]:
        """Returns column of first non-zero entry, or None if row is all zeros."""
        for c in range(self.cols):
            if self.data[r][c] != 0:
                return c
        return None

    def is_row_echelon_form(self) -> bool:
        """
        Exact mathematical Row Echelon Form validation:
        1. All zero rows are strictly below all non-zero rows.
        2. Leading entry of each non-zero row is strictly to the right of the row above it.
        3. All entries below leading pivots are strictly zero.
        """
        last_lead = -1
        seen_zero_row = False

        for r in range(self.rows):
            lead = self.get_leading_col(r)
            if lead is None:
                seen_zero_row = True
            else:
                if seen_zero_row:
                    return False
                if lead <= last_lead:
                    return False
                last_lead = lead
        return True

    def count_ref_violations(self) -> int:
        """Exact violation count for heuristic distance / reward shaping."""
        violations = 0
        leads = [self.get_leading_col(r) for r in range(self.rows)]
        zero_rows = [r for r, l in enumerate(leads) if l is None]

        if zero_rows:
            min_zero = min(zero_rows)
            for r, l in enumerate(leads):
                if l is not None and r > min_zero:
                    violations += 2

        # Check lead progression
        non_zeros = [(r, l) for r, l in enumerate(leads) if l is not None]
        for i in range(len(non_zeros) - 1):
            r1, c1 = non_zeros[i]
            r2, c2 = non_zeros[i + 1]
            if c2 <= c1:
                violations += (c1 - c2 + 1)

        # Check non-zeros below pivots
        for r, c in non_zeros:
            for sub_r in range(r + 1, self.rows):
                if self.data[sub_r][c] != 0:
                    violations += 1

        return violations

    # -------------------------------------------------------------
    # Raw Elementary Row Operations (EROs)
    # -------------------------------------------------------------
    def swap_rows(self, r1: int, r2: int) -> 'ExactMatrix':
        """R_r1 <-> R_r2"""
        assert 0 <= r1 < self.rows and 0 <= r2 < self.rows
        if r1 == r2:
            return self.copy()
        new_mat = self.copy()
        new_mat.data[r1], new_mat.data[r2] = new_mat.data[r2], new_mat.data[r1]
        return new_mat

    def add_row_multiple(self, target: int, source: int, scalar: Union[int, Fraction]) -> 'ExactMatrix':
        """R_target <- R_target + scalar * R_source"""
        assert 0 <= target < self.rows and 0 <= source < self.rows and target != source
        frac = Fraction(scalar)
        new_mat = self.copy()
        for c in range(self.cols):
            new_mat.data[target][c] += frac * new_mat.data[source][c]
        return new_mat

    def scale_row(self, target: int, scalar: Union[int, Fraction]) -> 'ExactMatrix':
        """R_target <- scalar * R_target"""
        assert 0 <= target < self.rows
        frac = Fraction(scalar)
        assert frac != 0, "Cannot scale row by zero"
        new_mat = self.copy()
        for c in range(self.cols):
            new_mat.data[target][c] *= frac
        return new_mat
