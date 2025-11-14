"""
Koan 07: Word Embeddings - Representaciones Vectoriales

Los word embeddings representan words como vectors numéricos,
capturando relaciones semánticas.

Example:
- "rey" - "hombre" + "mujer" ≈ "reina"
- Similar("gato", "perro") > Similar("gato", "coche")

Librerías:
- spaCy: Embeddings pre-entrenados
- gensim: Word2Vec, FastText
"""

import spacy
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cosine


def get_word_vector_spacy(word: str, lang: str = "es") -> np.ndarray:
    """
    Obtiene el vector de una palabra using spaCy.

    Example:
        >>> vector = get_word_vector_spacy("python")
        >>> vector.shape
        (96,)  # Depende del model

    Args:
        word: Palabra a vectorizar
        lang: Idioma

    Returns:
        Vector numpy
    """
    # TODO: Implement obtención de vector
    # Hint: spaCy puede procesar text y cada documento tiene un atributo de vector
    # Consulta HINTS.md para más detalles
    pass


def get_text_vector_spacy(text: str, lang: str = "es") -> np.ndarray:
    """
    Obtiene el vector promedio de un text.

    Example:
        >>> vector = get_text_vector_spacy("me gusta python")

    Args:
        text: Text a vectorizar
        lang: Idioma

    Returns:
        Vector numpy
    """
    # TODO: Similar a get_word_vector_spacy pero con text completo
    pass


def cosine_similarity_words(word1: str, word2: str, lang: str = "es") -> float:
    """
    Calculates la similitud coseno entre dos words.

    Valores cercanos a 1 = muy similares
    Valores cercanos a 0 = no relacionadas

    Example:
        >>> cosine_similarity_words("gato", "perro")
        0.85  # Alta similitud (ambos son animales)
        >>> cosine_similarity_words("gato", "coche")
        0.23  # Baja similitud

    Args:
        word1: Primera palabra
        word2: Segunda palabra
        lang: Idioma

    Returns:
        Similitud (0-1)
    """
    # TODO: Implement similitud coseno
    # Hint: Necesitas obtener vectors y calcular similitud coseno (ya está importada)
    pass


def find_most_similar(
    word: str, candidates: List[str], lang: str = "es", top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Encuentra las words más similares a una dada.

    Example:
        >>> find_most_similar("perro", ["gato", "coche", "pájaro", "manzana"])
        [('gato', 0.85), ('pájaro', 0.72), ('manzana', 0.15)]

    Args:
        word: Palabra de referencia
        candidates: List of words candidatas
        lang: Idioma
        top_n: Número de resultados

    Returns:
        List of tuplas (palabra, similitud) ordenadas por similitud
    """
    # TODO: Implement búsqueda de similares
    # Hint: Calculates similitud con cada candidato y ordena
    pass


def word_analogy(
    word_a: str, word_b: str, word_c: str, candidates: List[str], lang: str = "es"
) -> str:
    """
    Resuelve analogías: A es a B como C es a ?

    Ejemplo clásico:
        >>> word_analogy("rey", "hombre", "mujer", ["reina", "princesa", "niña"])
        "reina"  # rey - hombre + mujer ≈ reina

    Args:
        word_a: Palabra A
        word_b: Palabra B
        word_c: Palabra C
        candidates: Palabras candidatas for the respuesta
        lang: Idioma

    Returns:
        Palabra más apropiada
    """
    # TODO: Implement analogía
    # Hint: Los vectors se pueden sumar y restar algebraicamente
    pass


def get_document_similarity(text1: str, text2: str, lang: str = "es") -> float:
    """
    Calculates la similitud entre dos documents.

    Example:
        >>> text1 = "Me gusta programar en Python"
        >>> text2 = "Python es mi lenguaje favorito"
        >>> get_document_similarity(text1, text2)
        0.78  # Alta similitud

    Args:
        text1: Primer text
        text2: Segundo text
        lang: Idioma

    Returns:
        Similitud (0-1)
    """
    # TODO: Calculates similitud entre documents
    # Combina funciones anteriores
    pass


def cluster_words_by_similarity(
    words: List[str], threshold: float = 0.7, lang: str = "es"
) -> List[List[str]]:
    """
    Agrupa words similares en clusters.

    Example:
        >>> words = ["perro", "gato", "coche", "auto", "pájaro"]
        >>> cluster_words_by_similarity(words, threshold=0.7)
        [['perro', 'gato', 'pájaro'], ['coche', 'auto']]

    Args:
        words: List of words
        threshold: Umbral de similitud mínima
        lang: Idioma

    Returns:
        List of clusters (listas de words similares)
    """
    # TODO: Implement clustering simple
    # Agrupa words con similitud por encima del threshold
    pass
