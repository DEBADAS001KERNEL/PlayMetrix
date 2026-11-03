"""
PlayMetrix - Discovery & Execution History Ledger
Tracks every matrix solution, compares Agent steps vs Classical Baselines,
and logs novel discoveries (shorter step sequences or alternative paths).
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np


class DiscoveryLedger:
    """
    Records benchmark comparisons between RL agents and classical mathematical algorithms.
    Flags and archives novel discoveries where the agent found a shorter path.
    """

    def __init__(self, ledger_file: Optional[str] = None):
        if ledger_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ledger_file = os.path.join(base_dir, "history_ledger.json")
        self.ledger_file = ledger_file
        self.records: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.ledger_file)), exist_ok=True)
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def log_solution(
        self,
        task: str,
        initial_matrix: np.ndarray,
        final_matrix: np.ndarray,
        agent_steps: int,
        classical_steps: int,
        agent_actions: List[str],
        classical_actions: List[str],
        is_valid: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Logs a completed matrix task and determines if a shorter sequence was discovered.
        """
        step_difference = classical_steps - agent_steps
        # Novel discovery: Agent found a verified correct solution in FEWER steps than classical baseline!
        is_novel_shorter = (step_difference > 0) and is_valid
        is_tied = (step_difference == 0) and is_valid

        entry = {
            "id": len(self.records) + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task.upper(),
            "matrix_shape": list(initial_matrix.shape),
            "initial_matrix": initial_matrix.tolist(),
            "final_matrix": final_matrix.tolist(),
            "agent_steps": agent_steps,
            "classical_steps": classical_steps,
            "step_difference": step_difference,
            "is_novel_shorter": is_novel_shorter,
            "is_tied": is_tied,
            "is_valid": is_valid,
            "agent_actions": agent_actions,
            "classical_actions": classical_actions,
            "metadata": metadata or {}
        }

        self.records.append(entry)
        self._save()
        return entry

    def get_novel_discoveries(self) -> List[Dict[str, Any]]:
        """Returns all cases where agent outperformed the classical baseline."""
        return [r for r in self.records if r.get("is_novel_shorter", False)]

    def get_summary(self) -> Dict[str, Any]:
        """Returns win/tie/loss statistics comparing agent vs classical baseline."""
        if not self.records:
            return {"total": 0, "faster": 0, "tied": 0, "slower": 0, "win_rate": 0.0}

        faster = sum(1 for r in self.records if r.get("is_novel_shorter", False))
        tied = sum(1 for r in self.records if r.get("is_tied", False))
        slower = sum(1 for r in self.records if r.get("step_difference", 0) < 0 and r.get("is_valid", False))

        return {
            "total": len(self.records),
            "faster_than_classical": faster,
            "tied_with_classical": tied,
            "slower_than_classical": slower,
            "efficiency_rate": (faster + tied) / len(self.records) * 100.0 if self.records else 0.0
        }
