from types import SimpleNamespace

import yaml


def apply_overrides(cfg, overrides):
    """Apply 'section.key=value' strings to a nested SimpleNamespace config. Values are parsed as YAML,
    so 0.1 becomes a float and True a bool; missing sections are created."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override {item!r} is not of the form section.key=value")
        *path, last = key.split(".")
        node = cfg
        for p in path:
            if not hasattr(node, p):
                setattr(node, p, SimpleNamespace())
            node = getattr(node, p)
        setattr(node, last, yaml.safe_load(raw))
    return cfg
