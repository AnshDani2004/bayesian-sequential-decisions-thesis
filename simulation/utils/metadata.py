"""
Metadata Utility Module

Provides consistent metadata tracking for reproducibility:
- Random seeds
- Git commit hash
- Timestamps
- Environment info

All experiment results should include this metadata.
"""

import json
import platform
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RunMetadata:
    """Container for experiment metadata."""
    seed: Optional[int]
    git_commit: str
    timestamp: str
    python_version: str
    platform: str
    numpy_version: str
    config: Dict[str, Any]


def get_git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def get_run_metadata(
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate metadata dictionary for an experiment run.
    
    Args:
        seed: Random seed used for the experiment
        config: Experiment configuration parameters
    
    Returns:
        Dictionary with reproducibility metadata
    """
    import numpy as np
    
    return {
        'seed': seed,
        'git_commit': get_git_commit(),
        'timestamp': datetime.datetime.now().isoformat(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'config': config or {}
    }


def save_with_metadata(
    data: Dict[str, Any],
    path: str,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save experiment results with metadata.
    
    Args:
        data: Experiment results to save
        path: Output file path (JSON)
        seed: Random seed used
        config: Experiment configuration
    """
    result = {
        'metadata': get_run_metadata(seed, config),
        'data': data
    }
    
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def load_with_metadata(path: str) -> Dict[str, Any]:
    """
    Load experiment results with metadata.
    
    Args:
        path: Path to JSON results file
    
    Returns:
        Dictionary with 'metadata' and 'data' keys
    """
    with open(path, 'r') as f:
        return json.load(f)


def print_metadata(metadata: Dict[str, Any]) -> None:
    """Pretty-print metadata for logging."""
    print("=" * 60)
    print("EXPERIMENT METADATA")
    print("=" * 60)
    print(f"  Seed:        {metadata.get('seed', 'N/A')}")
    print(f"  Git Commit:  {metadata.get('git_commit', 'unknown')}")
    print(f"  Timestamp:   {metadata.get('timestamp', 'N/A')}")
    print(f"  Python:      {metadata.get('python_version', 'N/A')}")
    print(f"  NumPy:       {metadata.get('numpy_version', 'N/A')}")
    print(f"  Platform:    {metadata.get('platform', 'N/A')}")
    print("=" * 60)
