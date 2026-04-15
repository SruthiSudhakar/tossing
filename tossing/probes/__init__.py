from abc import ABC, abstractmethod
from tossing.types import ProbeResult

class ProbeController(ABC):
    """Base class for all probe action primitives."""

    probe_type: str  # "vertical_toss" | "forward_toss" | "release_drop" | "wrist_flick" | "shake"
    cost: int = 1

    # Subclasses override with {param_name: (default, lo, hi)}. Empty = no tunable params.
    PARAM_SPEC: dict[str, tuple[float, float, float]] = {}

    @abstractmethod
    def execute(self, env, params: dict | None = None) -> ProbeResult:
        """Run the probe on the current object and return structured observations."""
        ...

    def resolve_params(self, params: dict | None) -> dict[str, float]:
        """Merge user-supplied params over defaults; clamp to declared ranges.

        Unknown keys raise ValueError so the VLM's typos surface as parse errors.
        """
        resolved = {name: spec[0] for name, spec in self.PARAM_SPEC.items()}
        for k, v in (params or {}).items():
            if k not in self.PARAM_SPEC:
                raise ValueError(
                    f"Unknown param '{k}' for probe '{self.probe_type}'. "
                    f"Known: {list(self.PARAM_SPEC)}"
                )
            _, lo, hi = self.PARAM_SPEC[k]
            resolved[k] = float(max(lo, min(hi, float(v))))
        return resolved


# Registry mapping probe IDs to controller classes
_PROBE_REGISTRY: dict[str, type[ProbeController]] = {}


def register_probe(probe_type: str):
    """Decorator to register a probe controller class."""
    def decorator(cls):
        _PROBE_REGISTRY[probe_type] = cls
        cls.probe_type = probe_type
        return cls
    return decorator


def get_probe(probe_type: str) -> ProbeController:
    """Instantiate a probe controller by type ID."""
    if probe_type not in _PROBE_REGISTRY:
        raise ValueError(f"Unknown probe type: {probe_type}. Available: {list(_PROBE_REGISTRY.keys())}")
    return _PROBE_REGISTRY[probe_type]()


def list_probes() -> list[str]:
    """Return all registered probe type IDs."""
    return list(_PROBE_REGISTRY.keys())


def get_param_spec(probe_type: str) -> dict[str, tuple[float, float, float]]:
    """Return the PARAM_SPEC for a probe type."""
    if probe_type not in _PROBE_REGISTRY:
        raise ValueError(f"Unknown probe type: {probe_type}")
    return dict(_PROBE_REGISTRY[probe_type].PARAM_SPEC)
