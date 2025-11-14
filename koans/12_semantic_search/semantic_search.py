"""
Koan 12: Semantic Search & Vector Databases

Búsqueda semántica using embeddings y vector databases:
- OpenAI Embeddings
- Sentence Transformers
- ChromaDB, FAISS
- Similarity search

Librerías: openai, sentence-transformers, chromadb, faiss-cpu
"""

from typing import List, Tuple
import numpy as np


def create_openai_embedding(
    text: str, model: str = "text-embedding-3-small"
) -> List[float]:
    """Creates embedding con OpenAI API"""
    # TODO: Implementr con openai.embeddings.create()
    # Hint: Consulta HINTS.md para detalles de la API
    pass


def create_sentence_transformer_embedding(
    text: str, model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """Creates embedding con Sentence Transformers (local, gratis)"""
    # TODO: Importa SentenceTransformer y crea embeddings localmente
    pass


def cosine_similarity_search(
    query_embedding: List[float], document_embeddings: List[List[float]], top_k: int = 5
) -> List[Tuple[int, float]]:
    """Encuentra documents más similares por similitud coseno"""
    # TODO: Implementr búsqueda por similitud coseno
    # Hint: Calculates la similitud entre query y cada documento
    pass


def create_chromadb_collection(collection_name: str):
    """Creates colección en ChromaDB para búsqueda vectorial"""
    # TODO: Createste cliente ChromaDB y colección
    pass


def add_documents_to_chromadb(
    collection, documents: List[str], metadatas: List[dict] = None
):
    """Añade documents a ChromaDB con embeddings automáticos"""
    # TODO: Añade documents con IDs y metadatos opcionales
    pass


def search_chromadb(collection, query: str, top_k: int = 5) -> List[dict]:
    """Busca en ChromaDB"""
    # TODO: Ejecuta búsqueda semántica en la colección
    pass


def create_faiss_index(embeddings: np.ndarray):
    """Creates índice FAISS para búsqueda rápida"""
    # TODO: Createste índice FAISS with the dimensión correcta
    pass


def semantic_search_with_reranking(
    query: str, documents: List[str], top_k: int = 5
) -> List[Tuple[str, float]]:
    """Búsqueda semántica con reranking"""
    # TODO: 1. Búsqueda inicial con embeddings, 2. Rerank con cross-encoder
    pass
