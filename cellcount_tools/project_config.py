"""Helpers for loading project-level configuration for channel aliases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

CONFIG_FILENAME = "channel_aliases.json"


def load_channel_aliases(project_root: Path | None) -> Dict[str, str]:
    """Return channel alias overrides defined in ``channel_aliases.json``.

    The configuration file is expected to live at ``<project_root>/channel_aliases.json``
    and contain a simple mapping from channel tokens (e.g. ``"cy5"``) to display
    labels (e.g. ``"SOX2"``). Tokens are normalised to lowercase.
    """

    if project_root is None:
        return {}

    config_path = (project_root / CONFIG_FILENAME).expanduser().resolve()
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - runtime guard
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top level of {config_path}")

    return {str(key).strip().lower(): str(value).strip() for key, value in data.items()}

