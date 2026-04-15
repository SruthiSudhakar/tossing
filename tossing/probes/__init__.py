from abc import ABC, abstractmethod
from tossing.types import ProbeResult

class ProbeController(ABC):
    """Base class for all probe action primitives."""

    probe_type: str  # "vertical_toss" | "forward_toss" | "release_drop" | "wrist_flick" | "shake"
    cost: int = 1

    @abstractmethod
    def execute(self, env) -> ProbeResult:
        """Run the probe on the current object and return structured observations."""
        ...


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
