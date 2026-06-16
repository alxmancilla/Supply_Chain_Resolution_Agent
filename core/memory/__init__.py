"""Memory layer.

Concrete LTM stores implement the protocols defined in
`core.protocols` and hide their storage backend. Today the only
backend is MongoDB Atlas; tomorrow could be anything implementing
the same protocols.
"""
from .episodic import MongoEpisodicMemory, get_episodic_memory
from .procedural import MongoProceduralMemory, get_procedural_memory
from .reflector import (
    LLMMemoryReflector,
    MemoryAdmin,
    MongoMemoryAdmin,
    ReflectionReport,
    StoredFact,
)
from .semantic import MongoSemanticMemory, get_semantic_memory


def reset_memory_cache() -> None:
    """Clear the cached LTM singletons so freshly-committed rules / facts
    take effect without restarting the process. Safe to call from the UI
    on a New-Session click; the underlying Atlas collections are unchanged.
    """
    get_semantic_memory.cache_clear()
    get_episodic_memory.cache_clear()
    get_procedural_memory.cache_clear()


__all__ = [
    "MongoSemanticMemory",
    "MongoEpisodicMemory",
    "MongoProceduralMemory",
    "get_semantic_memory",
    "get_episodic_memory",
    "get_procedural_memory",
    "reset_memory_cache",
    "LLMMemoryReflector",
    "MemoryAdmin",
    "MongoMemoryAdmin",
    "ReflectionReport",
    "StoredFact",
]
