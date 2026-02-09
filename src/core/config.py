"""
Argus Configuration Loader
==========================

Loads and validates YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the repo root by walking upward for known anchors."""
    start_path = (start or Path.cwd()).resolve()
    for current in [start_path, *start_path.parents]:
        if (current / ".git").exists() or (current / "config" / "config.yaml").exists():
            return current
    return start_path


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_root = find_repo_root(Path(__file__).resolve())
    return repo_root / candidate


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load main configuration file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    path = _resolve_path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    
    return load_yaml(str(path))


def load_secrets(secrets_path: str = "config/secrets.yaml") -> Dict[str, Any]:
    """
    Load secrets configuration file.
    
    Args:
        secrets_path: Path to secrets.yaml
        
    Returns:
        Secrets dictionary
    """
    override_path = os.getenv("ARGUS_SECRETS")
    if override_path:
        secrets_path = override_path

    path = _resolve_path(secrets_path)
    if not path.exists():
        raise ConfigurationError(
            f"Secrets file not found: {secrets_path}\n"
            f"Copy config/secrets.template.yaml to config/secrets.yaml and add your API keys."
        )
    
    return load_yaml(str(path))


def load_thresholds(thresholds_path: str = "config/thresholds.yaml") -> Dict[str, Any]:
    """
    Load detection thresholds configuration.
    
    Args:
        thresholds_path: Path to thresholds.yaml
        
    Returns:
        Thresholds dictionary
    """
    path = _resolve_path(thresholds_path)
    if not path.exists():
        raise ConfigurationError(f"Thresholds file not found: {thresholds_path}")
    
    return load_yaml(str(path))


def load_all_config(
    config_dir: str = "config"
) -> Dict[str, Any]:
    """
    Load all configuration files and merge them.
    
    Returns:
        Merged configuration with keys: 'system', 'secrets', 'thresholds'
    """
    config_dir = Path(config_dir)
    
    config = load_config(str(config_dir / "config.yaml"))
    secrets = load_secrets(str(config_dir / "secrets.yaml"))
    thresholds = load_thresholds(str(config_dir / "thresholds.yaml"))
    
    return {
        **config,
        'secrets': secrets,
        'thresholds': thresholds
    }


def validate_secrets(secrets: Dict[str, Any]) -> list:
    """
    Validate that required secrets are present.
    
    Returns:
        List of missing/invalid secret names
    """
    issues = []
    
    # Required secrets for observation mode
    required = {
        'binance': ['api_key', 'api_secret'],
        'okx': ['api_key', 'api_secret', 'passphrase'],
        'coinglass': ['api_key'],
        'telegram': ['bot_token', 'chat_id'],
    }
    
    for service, keys in required.items():
        if service not in secrets:
            issues.append(f"Missing {service} configuration")
            continue
        
        for key in keys:
            value = secrets[service].get(key, '')
            if not value or value.startswith('PASTE_') or value.startswith('YOUR_'):
                issues.append(f"Missing or placeholder: {service}.{key}")
    
    return issues


def get_exchange_config(config: Dict, exchange: str) -> Dict[str, Any]:
    """Get configuration for a specific exchange."""
    return config.get('exchanges', {}).get(exchange, {})


def get_threshold_config(config: Dict, detector: str) -> Dict[str, Any]:
    """Get threshold configuration for a specific detector."""
    return config.get('thresholds', {}).get(detector, {})


def get_secret(secrets: Dict, service: str, key: str) -> Optional[str]:
    """Safely get a secret value."""
    return secrets.get(service, {}).get(key)
