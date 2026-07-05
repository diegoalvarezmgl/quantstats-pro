"""
Registry of Montecarlo simulation models, keyed by name.
"""

from __future__ import annotations

from .base import SimulationModel

_REGISTRY: dict[str, type[SimulationModel]] = {}


def register(model_cls: type[SimulationModel]) -> type[SimulationModel]:
    """Class decorator that registers a :class:`SimulationModel` by ``name``."""
    name = model_cls.name
    if not name or name == "base":
        raise ValueError(f"Model {model_cls!r} must define a unique 'name'.")
    _REGISTRY[name] = model_cls
    return model_cls


def get_model(name: str) -> SimulationModel:
    """Instantiate a registered model by name."""
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown Montecarlo model '{name}'. "
            f"Available: {', '.join(available_models())}"
        ) from exc


def available_models() -> list[str]:
    """Return the registered model names, in registration order."""
    return list(_REGISTRY.keys())
