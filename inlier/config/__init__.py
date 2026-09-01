"""Config loading, merging, and resolution.

Layering, lowest precedence first:

1. the packaged ``inlier/config/default.yaml`` -- always the base, so every key
   has a value and the resolver never needs an inline fallback
2. the user's ``--config FILE``, deep merged on top
3. ``--set dotted.key=value`` overrides from the command line

``resolve()`` then turns the merged dict into the core dataclasses.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from inlier.config.schema import (
    EVAL_SCORE_THRESHOLD,
    SCHEMA,
    Mode,
    ResolvedConfig,
    resolve,
    validate,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # lazy: keeps `import inlier` free of the dependency

    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; ``over`` wins.  Neither input is mutated."""
    out = copy.deepcopy(dict(base))
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(item: str) -> Dict[str, Any]:
    """``"stage1.topk=50"`` -> ``{"stage1": {"topk": 50}}``.

    The value is parsed as YAML, so ``null``, ``true``, ``0.2`` and ``[1,2]``
    all mean what they look like.
    """
    if "=" not in item:
        raise ValueError(f"--set expects 'key=value', got {item!r}")
    dotted, _, raw = item.partition("=")
    dotted = dotted.strip()
    if not dotted:
        raise ValueError(f"--set expects 'key=value', got {item!r}")

    import yaml

    value = yaml.safe_load(raw)
    out: Dict[str, Any] = {}
    cursor = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return out


def load(
    path: Optional[str | Path] = None,
    overrides: Iterable[str] = (),
    *,
    defaults: bool = True,
) -> Dict[str, Any]:
    """Return the merged, validated config dict.

    Args:
        path:      user config; ``None`` uses the packaged defaults alone.
        overrides: ``--set`` strings, applied last.
        defaults:  start from the packaged ``default.yaml`` (leave this on --
            it is what removes the need for inline fallbacks).
    """
    cfg: Dict[str, Any] = _load_yaml(DEFAULT_CONFIG_PATH) if defaults else {}
    if path is not None:
        user_path = Path(path)
        if not user_path.exists():
            raise FileNotFoundError(f"config file not found: {user_path}")
        cfg = deep_merge(cfg, _load_yaml(user_path))
    for item in overrides:
        cfg = deep_merge(cfg, parse_override(item))
    validate(cfg)
    return cfg


def dump(cfg: Mapping[str, Any]) -> str:
    """Serialise a config dict back to YAML (for ``inlier config show``)."""
    import yaml

    return yaml.safe_dump(dict(cfg), sort_keys=False, default_flow_style=False)


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "EVAL_SCORE_THRESHOLD",
    "Mode",
    "ResolvedConfig",
    "SCHEMA",
    "deep_merge",
    "dump",
    "load",
    "parse_override",
    "resolve",
    "validate",
]
