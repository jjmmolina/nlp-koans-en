"""
Koan 03: POS Tagging - Etiquetado Gramatical

POS (Part-of-Speech) Tagging identifica la categoría gramatical de cada palabra:
sustantivo, verbo, adjetivo, etc.

Ejemplos:
- "Python" → NOUN (sustantivo)
- "es" → VERB (verbo)
- "genial" → ADJ (adjetivo)

Librerías:
- spaCy: POS tagging moderno y preciso
- NLTK: POS tagging clásico
"""

import nltk
from typing import List, Tuple, Dict

# TODO: Descarga recursos
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')


def pos_tag_nltk(text: str) -> List[Tuple[str, str]]:
    """
    Etiqueta gramaticalmente un text using NLTK.

    Ejemplo:
        >>> pos_tag_nltk("Python is awesome")
        [('Python', 'NNP'), ('is', 'VBZ'), ('awesome', 'JJ')]

    Args:
        text: Text a etiquetar

    Returns:
        List of tuplas (palabra, etiqueta)
    """
    # TODO: Implement POS tagging with NLTK
    # Pistas:
    # 1. Primero necesitas tokenizar el text
    # 2. Luego aplica nltk.pos_tag() a los tokens
    pass


def pos_tag_spacy(text: str, lang: str = "es") -> List[Tuple[str, str, str]]:
    """
    Etiqueta gramaticalmente using spaCy.

    spaCy proporciona etiquetas más detalladas y universales.

    Ejemplo:
        >>> pos_tag_spacy("Python es genial")
        [('Python', 'PROPN', 'nombre propio'),
         ('es', 'AUX', 'verbo auxiliar'),
         ('genial', 'ADJ', 'adjetivo')]

    Args:
        text: Text a etiquetar
        lang: Idioma ('es' o 'en')

    Returns:
        List of tuplas (palabra, etiqueta_universal, etiqueta_detallada)
    """
    # TODO: Implement with spaCy
    # Pistas:
    # 1. Carga el modelo de spaCy
    # 2. Procesa el text
    # 3. Cada token tiene atributos: .text, .pos_, .tag_
    pass


def extract_nouns(text: str, lang: str = "es") -> List[str]:
    """
    Extracts todos los sustantivos de un text.

    Ejemplo:
        >>> extract_nouns("El gato y el perro juegan")
        ['gato', 'perro']

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        List of sustantivos
    """
    # TODO: Extracts solo los tokens with POS == 'NOUN' o 'PROPN'
    # Hint: Usa la función pos_tag_spacy y filtra por categoría
    pass


def extract_verbs(text: str, lang: str = "es") -> List[str]:
    """
    Extracts todos los verbos de un text.

    Ejemplo:
        >>> extract_verbs("El gato come y el perro corre")
        ['come', 'corre']

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        List of verbos
    """
    # TODO: Extracts tokens with POS == 'VERB' o 'AUX'
    # Similar a extract_nouns pero with diferentes categorías
    pass


def extract_adjectives(text: str, lang: str = "es") -> List[str]:
    """
    Extracts todos los adjetivos de un text.

    Ejemplo:
        >>> extract_adjectives("El gato negro es muy rápido")
        ['negro', 'rápido']

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        List of adjetivos
    """
    # TODO: Extracts tokens with POS == 'ADJ'
    pass


def get_pos_statistics(text: str, lang: str = "es") -> Dict[str, int]:
    """
    Calcula estadísticas de POS tags en un text.

    Ejemplo:
        >>> get_pos_statistics("El gato negro come")
        {'DET': 1, 'NOUN': 1, 'ADJ': 1, 'VERB': 1}

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        Diccionario with withteo de cada POS tag
    """
    # TODO: Cuenta la frecuencia de cada POS tag
    # Hint: collections.Counter es muy útil aquí
    pass


def find_noun_phrases(text: str, lang: str = "es") -> List[str]:
    """
    Encuentra frases nominales (noun phrases) en un text.

    Una frase nominal es un grupo de words with un sustantivo como núcleo.
    Ejemplo: "el gato negro", "mi casa grande"

    Ejemplo:
        >>> find_noun_phrases("El gato negro duerme")
        ['El gato negro']

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        List of frases nominales
    """
    # TODO: Implement extracción de noun chunks with spaCy
    # Hint: spaCy tiene un atributo especial en el doc for esto
    pass


def pos_pattern_match(
    text: str, pattern: List[str], lang: str = "es"
) -> List[List[str]]:
    """
    Encuentra secuencias de words que coinciden with un patrón de POS tags.

    Ejemplo:
        >>> # Buscar patrón ADJ + NOUN (adjetivo + sustantivo)
        >>> pos_pattern_match("El gato negro y el perro blanco", ["ADJ", "NOUN"])
        [['negro', 'gato'], ['blanco', 'perro']]

    Args:
        text: Text a analizar
        pattern: List of POS tags a buscar (en orden)
        lang: Idioma

    Returns:
        List of secuencias que coinciden with el patrón
    """
    # TODO: Implement búsqueda de patrones
    # Pistas:
    # 1. Obtén los POS tags del text
    # 2. Recorre el text buscando ventanas del tamaño del patrón
    # 3. Comfor los POS tags de cada ventana with el patrón
    pass
