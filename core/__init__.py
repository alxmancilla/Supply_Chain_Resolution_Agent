"""Layered, reusable building blocks for the Supply Chain Resolution Agent.

Layer map (CoALA-aligned):
  L0 storage    — shared MongoClient, DB/collection constants, registry
  L1 rag        — knowledge corpus retrieval
  L2 memory     — semantic / episodic / procedural long-term memory
  protocols     — abstract contracts shared across layers
  schemas       — Pydantic value shapes
  settings      — env-driven config + AgentContext
  latency       — cross-cutting timing decorator

The agent/ package depends on core/. core/ never depends on agent/.
"""
