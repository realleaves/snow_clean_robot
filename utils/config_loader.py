"""Central YAML configuration loading and validation."""
from pathlib import Path
import yaml

from utils.errors import ConfigError

DEFAULT_SECTIONS = ('system', 'camera', 'perception', 'fusion', 'planner', 'cleaning')


def load_config(root: str | Path, sections: tuple[str, ...] = DEFAULT_SECTIONS) -> dict:
    """Load every ``config/<section>.yaml`` below *root* into one dictionary."""
    root = Path(root)
    config: dict = {}
    for name in sections:
        path = root / 'config' / f'{name}.yaml'
        if not path.exists():
            raise ConfigError(f'missing configuration file: {path}')
        try:
            loaded = yaml.safe_load(path.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:  # pragma: no cover - defensive
            raise ConfigError(f'malformed YAML in {path.name}: {exc}') from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f'{path.name} must contain a mapping at the top level')
        config[name] = loaded
    return config


def require_keys(section: dict, keys: tuple[str, ...], name: str) -> None:
    """Raise :class:`ConfigError` listing every key missing from *section*."""
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f'{name} is missing required keys: {", ".join(missing)}')
