"""
Koan 01: Tokenizesción - Dividiendo el Text en Unidades

La tokenización es el proceso de dividir text en unidades más pequeñas (tokens),
que pueden ser words, sentences, o incluso caracteres.

Este es el primer paso fundamental en casi cualquier pipeline de NLP.

Librerías usadas:
- NLTK: Natural Language Toolkit (clásico)
- spaCy: Procesamiento industrial de NLP
"""

import nltk
from typing import List
from nltk.tokenize import word_tokenize, sent_tokenize

# TODO: Descarga los recursos necesarios de NLTK
# Descomenta estas líneas cuando las necesites:
# nltk.download('punkt')
# nltk.download('punkt_tab')


def tokenize_words_nltk(text: str) -> List[str]:
    """
    Tokenizes un text en words using NLTK.

    Ejemplo:
        >>> tokenize_words_nltk("Hola, ¿cómo estás?")
        ['Hola', ',', '¿', 'cómo', 'estás', '?']

    Args:
        text: Text to tokenize

    Returns:
        List of tokens (words y signos de puntuación)
    """
    # TODO: Implement la tokenización de words
    # Hint: Necesitas usar la función word_tokenize que ya está importada arriba
    # ¿Qué debes retornar?
    pass


def tokenize_sentences_nltk(text: str) -> List[str]:
    """
    Tokenizes un text en sentences using NLTK.

    Ejemplo:
        >>> text = "Hola mundo. ¿Cómo estás? Yo estoy bien."
        >>> tokenize_sentences_nltk(text)
        ['Hola mundo.', '¿Cómo estás?', 'Yo estoy bien.']

    Args:
        text: Text to tokenize

    Returns:
        List of sentences
    """
    # TODO: Implement la tokenización de sentences
    # Hint: Similar a word_tokenize, pero for sentences. Ya está importada arriba.
    pass


def tokenize_words_spacy(text: str, lang: str = "es") -> List[str]:
    """
    Tokenizes un text en words using spaCy.

    spaCy es más sofisticado que NLTK y maneja mejor casos especiales.

    Ejemplo:
        >>> tokenize_words_spacy("Dr. Smith ganó $1,000 dólares.")
        ['Dr.', 'Smith', 'ganó', '$', '1,000', 'dólares', '.']

    Args:
        text: Text to tokenize
        lang: Language ('es' for español, 'en' for inglés)

    Returns:
        List of tokens
    """
    # TODO: Implement la tokenización with spaCy
    # Pistas:
    # 1. Importa spacy
    # 2. Carga el modelo correcto según el language
    # 3. Procesa el text with el modelo
    # 4. Extracts los tokens (cada token tiene un atributo .text)
    # Consulta HINTS.md si necesitas más ayuda
    pass


def custom_tokenize(text: str, delimiter: str = " ") -> List[str]:
    """
    Tokenizesción simple using un delimiter.

    A veces una simple división es suficiente.

    Ejemplo:
        >>> custom_tokenize("Hola-mundo-Python", delimiter="-")
        ['Hola', 'mundo', 'Python']

    Args:
        text: Text to tokenize
        delimiter: Delimitador for seforr tokens

    Returns:
        List of tokens
    """
    # TODO: Implement una tokenización simple
    # Hint: Los strings en Python tienen un método que divide por un delimiter
    pass


def count_tokens(text: str) -> dict:
    """
    Cuenta la frecuencia de cada token en un text.

    Ejemplo:
        >>> count_tokens("el gato y el perro")
        {'el': 2, 'gato': 1, 'y': 1, 'perro': 1}

    Args:
        text: Text a analizar

    Returns:
        Diccionario with frecuencias de tokens
    """
    # TODO: Implement el withteo de tokens
    # Pistas:
    # 1. Primero tokeniza el text (puedes usar una función que ya creaste)
    # 2. Normalizes a minúsculas
    # 3. Cuenta las frecuencias (mira collections.Counter o usa un dict)
    pass


def remove_punctuation_tokens(tokens: List[str]) -> List[str]:
    """
    Elimina signos de puntuación de una list of tokens.

    Ejemplo:
        >>> remove_punctuation_tokens(['Hola', ',', 'mundo', '!'])
        ['Hola', 'mundo']

    Args:
        tokens: List of tokens

    Returns:
        List of tokens sin puntuación
    """
    # TODO: Filtra los tokens que NO sean puntuación
    # Pistas:
    # 1. El módulo string tiene una withstante with todos los signos de puntuación
    # 2. Usa una list comprehension for filtrar
    pass

