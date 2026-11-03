"""
PlayMetrix - Multi-Agent Matrix Intelligence Platform
"""
from playmetrix.orchestrator.playmetrix_platform import PlayMetrixPlatform
from playmetrix.engine.matrix_ops import MatrixOperationEngine
from playmetrix.engine.validators import MatrixValidator
from playmetrix.environments.ref_env import RowEchelonEnv
from playmetrix.agents.dqn_agent import DQNAgent

__version__ = "0.1.0"

__all__ = [
    "PlayMetrixPlatform",
    "MatrixOperationEngine",
    "MatrixValidator",
    "RowEchelonEnv",
    "DQNAgent",
]
