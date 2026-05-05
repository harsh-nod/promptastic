"""Capture mode registry and compatibility checking.

Each capture mode describes a type of data that can be extracted during
(or after) a forward pass.  Modes declare their requirements -- whether
they need gradients, whether they require multiple forward passes, and
which hook targets they attach to.  The registry lets the engine resolve
a ``CaptureConfig`` dict into concrete mode objects and validate that
the selected combination is consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .._types import CaptureConfig

# ======================================================================
# Data class
# ======================================================================


@dataclass
class CaptureMode:
    """Descriptor for a single capture mode.

    Attributes
    ----------
    name:
        Short identifier (must match a key in ``CaptureConfig``).
    requires_grad:
        ``True`` if the forward pass must run with gradients enabled.
    requires_multiple_passes:
        ``True`` if the mode needs more than one forward pass (e.g.
        activation patching, generation).
    hook_targets:
        Which submodule types this mode hooks into.  Valid values:
        ``"layer"`` (decoder layer), ``"attn"`` (self-attention),
        ``"mlp"`` (feed-forward).
    """

    name: str
    requires_grad: bool = False
    requires_multiple_passes: bool = False
    hook_targets: list[str] = field(default_factory=list)


# ======================================================================
# Registry
# ======================================================================

_REGISTRY: dict[str, CaptureMode] = {}


def register_mode(mode: CaptureMode) -> None:
    """Register a capture mode by name.

    Raises ``ValueError`` if *mode.name* is already registered.
    """
    if mode.name in _REGISTRY:
        raise ValueError(f"Capture mode '{mode.name}' is already registered")
    _REGISTRY[mode.name] = mode


def get_mode(name: str) -> CaptureMode:
    """Retrieve a registered capture mode by *name*.

    Raises ``KeyError`` if the mode has not been registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown capture mode '{name}'. "
            f"Registered modes: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def get_active_modes(config: CaptureConfig) -> list[CaptureMode]:
    """Resolve a ``CaptureConfig`` dict to a list of active ``CaptureMode`` objects.

    A mode is considered active when the corresponding boolean key in
    *config* is ``True`` **and** the mode is registered.  Non-boolean
    keys (like ``patching_method``, ``sae_weights_path``, etc.) are
    configuration parameters -- not mode toggles -- and are ignored
    here.
    """
    active: list[CaptureMode] = []
    for name, mode in _REGISTRY.items():
        if config.get(name, False) is True:  # type: ignore[arg-type]
            active.append(mode)
    return active


def validate_compatibility(modes: list[CaptureMode]) -> list[str]:
    """Check a set of active modes for compatibility issues.

    Returns a (possibly empty) list of human-readable warning strings.
    Does **not** raise -- the caller decides whether warnings are fatal.
    """
    warnings: list[str] = []

    grad_modes = [m for m in modes if m.requires_grad]
    no_grad_modes = [m for m in modes if not m.requires_grad and m.hook_targets]

    if grad_modes and no_grad_modes:
        grad_names = ", ".join(m.name for m in grad_modes)
        ngrad_names = ", ".join(m.name for m in no_grad_modes)
        warnings.append(
            f"Mixing gradient-requiring modes ({grad_names}) with "
            f"no-grad modes ({ngrad_names}). The forward pass will run "
            f"with gradients enabled, which increases memory usage."
        )

    multi_pass = [m for m in modes if m.requires_multiple_passes]
    if len(multi_pass) > 1:
        names = ", ".join(m.name for m in multi_pass)
        warnings.append(
            f"Multiple multi-pass modes active ({names}). Each will "
            f"run its own forward passes, increasing total compute."
        )

    # Check for hook target conflicts: multiple modes hooking the same
    # target is fine, but we warn if a multi-pass mode shares targets
    # with a single-pass mode (the single-pass hooks will fire during
    # the multi-pass runs too, potentially capturing unwanted data).
    if multi_pass:
        single_pass = [m for m in modes if not m.requires_multiple_passes]
        multi_targets: set[str] = set()
        for m in multi_pass:
            multi_targets.update(m.hook_targets)

        for m in single_pass:
            overlap = multi_targets.intersection(m.hook_targets)
            if overlap:
                warnings.append(
                    f"Single-pass mode '{m.name}' shares hook targets "
                    f"{sorted(overlap)} with multi-pass modes. Hooks "
                    f"should be removed or guarded between passes."
                )

    return warnings


# ======================================================================
# Built-in mode registration
# ======================================================================

def _register_builtins() -> None:
    """Register all built-in capture modes.

    Called once at module import time.
    """
    register_mode(CaptureMode(
        name="attention",
        hook_targets=["attn"],
    ))

    register_mode(CaptureMode(
        name="per_head",
        hook_targets=["attn"],
    ))

    register_mode(CaptureMode(
        name="residual",
        hook_targets=["layer"],
    ))

    register_mode(CaptureMode(
        name="logit_lens",
        # No hooks -- post-processes from the residual cache.
    ))

    register_mode(CaptureMode(
        name="mlp",
        hook_targets=["mlp"],
    ))

    register_mode(CaptureMode(
        name="tuned_lens",
        # No hooks -- post-processes from the residual cache.
    ))

    register_mode(CaptureMode(
        name="sae",
        # No hooks -- post-processes from the residual cache.
    ))

    register_mode(CaptureMode(
        name="patching",
        requires_multiple_passes=True,
        hook_targets=["layer"],
    ))

    register_mode(CaptureMode(
        name="gradients",
        requires_grad=True,
        hook_targets=["layer"],
    ))

    register_mode(CaptureMode(
        name="generation",
        requires_multiple_passes=True,
        hook_targets=["attn", "layer"],
    ))


_register_builtins()
