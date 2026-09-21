"""Fast mixers: fused or chunked twins of the reference implementations.

Each one is tested against its reference twin, never against the paper, and
shares its parameters and state-dict keys so checkpoints move between tiers.

Importing this package registers every fast mixer.
"""

from litterbox.model.mixers.fast import full_attention  # noqa: F401
