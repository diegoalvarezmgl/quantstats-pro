"""
Built-in simulation models. Importing this package registers them all.
"""

from . import (  # noqa: F401  (side-effect: registration)
    bayesian,
    block_bootstrap,
    bootstrap,
    garch,
    gbm,
    heston,
    jump_diffusion,
    shuffle,
    trimmed,
)
