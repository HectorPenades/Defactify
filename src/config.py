"""Defactify Framework - Configuration Loader"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration loader and manager for experiments."""

    def __init__(self, config_path: str):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to experiment config YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and merge base config with experiment config."""
        # Base config is always in configs/base_config.yaml
        base_config_path = Path(__file__).parent.parent / "configs" / "base_config.yaml"

        # Load base config
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)

        # Load experiment config
        with open(self.config_path, 'r') as f:
            exp_config = yaml.safe_load(f)

        # Handle extends field if it exists (for future use, but not needed now)
        if 'extends' in exp_config:
            exp_config.pop('extends')  # Remove it, we don't use it since we auto-include base

        # Merge experiment config into base (experiment overrides base)
        config = self._deep_merge(base_config, exp_config)

        return config

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __getitem__(self, key: str) -> Any:
        """Get config value by key."""
        return self.config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self.config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.config.copy()

    def __repr__(self) -> str:
        """Return string representation of config."""
        return f"Config({self.config_path})"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config = Config(config_path)
    return config.to_dict()
