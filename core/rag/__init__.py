"""RAG (Retrieval-Augmented Generation) layer.

Concrete retrievers implement `core.protocols.KnowledgeRetriever` and
hide their storage backend behind it.
"""
from .mongo import MongoKnowledgeRetriever

__all__ = ["MongoKnowledgeRetriever"]
