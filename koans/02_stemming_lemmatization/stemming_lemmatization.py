"""
Koan 02: Stemming y Lemmatization - Normalizesción de words

Stemming y Lemmatization son técnicas for reducir words a su forma base.

- Stemming: Corta el final de las words (rápido pero tosco)
  Ejemplo: "corriendo" → "corr"

- Lemmatization: Usa reglas lingüísticas (preciso pero más lento)
  Ejemplo: "corriendo" → "correr"

Librerías:
- NLTK: Varios stemmers (Porter, Snowball, etc.)
- spaCy: Lemmatization integrada
"""

import nltk
from typing import List

# TODO: Descarga recursos necesarios
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def stem_word_porter(word: str) -> str:
    """
    Aplica stemming using el algoritmo Porter.

    El Porter Stemmer es el más común for English.

    Ejemplo:
        >>> stem_word_porter("running")
        'run'
        >>> stem_word_porter("flies")
        'fli'

    Args:
        word: word a procesar

    Returns:
        Stem de la word
    """
    # TODO: Implement with PorterStemmer de NLTK
    # Hint: Necesitas importar la clase y crear una instancia
    # Consulta HINTS.md for más detalles
    pass


def stem_word_snowball(word: str, language: str = "spanish") -> str:
    """
    Aplica stemming using el algoritmo Snowball.

    Snowball soporta múltiples languages, incluyendo Spanish.

    Ejemplo:
        >>> stem_word_snowball("corriendo", "spanish")
        'corr'
        >>> stem_word_snowball("running", "english")
        'run'

    Args:
        word: word a procesar
        language: Idioma ('spanish', 'english', etc.)

    Returns:
        Stem de la word
    """
    # TODO: Implement with SnowballStemmer de NLTK
    # Hint: Este stemmer acepta un parámetro de language al crearse
    pass


def stem_sentence(sentence: str, language: str = "spanish") -> str:
    """
    Aplica stemming a todas las words de una oración.

    Ejemplo:
        >>> stem_sentence("Los gatos están corriendo")
        'los gat est corr'

    Args:
        sentence: Oración a procesar
        language: Idioma

    Returns:
        Oración with words stemmed
    """
    # TODO: Implement stemming de oración
    # Pistas:
    # 1. Divide la oración en words (tokenización simple)
    # 2. Aplica stemming a cada word (usa una función que ya creaste)
    # 3. Une las words procesadas
    pass


def lemmatize_word_nltk(word: str, pos: str = "n") -> str:
    """
    Lemmatiza una word using NLTK WordNet.

    Ejemplo:
        >>> lemmatize_word_nltk("running", pos="v")
        'run'
        >>> lemmatize_word_nltk("better", pos="a")
        'good'

    Args:
        word: word a lemmatizar
        pos: Part-of-speech tag ('n'=noun, 'v'=verb, 'a'=adjective, 'r'=adverb)

    Returns:
        Lema de la word
    """
    # TODO: Implement with WordNetLemmatizer
    # Hint: Crea una instancia del lemmatizer y usa su método lemmatize()
    pass


def lemmatize_with_spacy(text: str, lang: str = "es") -> List[str]:
    """
    Lemmatiza un text using spaCy.

    spaCy hace lemmatization automáticamente al procesar text.

    Ejemplo:
        >>> lemmatize_with_spacy("Los gatos están corriendo")
        ['el', 'gato', 'estar', 'correr']

    Args:
        text: Text a lemmatizar
        lang: Idioma ('es' o 'en')

    Returns:
        List of lemas
    """
    # TODO: Implement lemmatization with spaCy
    # Pistas:
    # 1. Carga el modelo spaCy apropiado
    # 2. Procesa el text
    # 3. Cada token tiene un atributo with su lema
    pass


def compare_stem_vs_lemma(word: str, language: str = "spanish") -> dict:
    """
    Comfor el resultado de stemming vs lemmatization.

    Ejemplo:
        >>> compare_stem_vs_lemma("corriendo")
        {
            'original': 'corriendo',
            'stem': 'corr',
            'lemma': 'correr'
        }

    Args:
        word: word a analizar
        language: Idioma

    Returns:
        Diccionario with resultados
    """
    # TODO: Implement la comforción
    # Usa las funciones que ya creaste for stemming y lemmatization
    return {
        "original": word,
        "stem": None,  # TODO: Aplica stemming aquí
        "lemma": None,  # TODO: Aplica lemmatization aquí
    }


def normalize_text(text: str, method: str = "lemma", language: str = "spanish") -> str:
    """
    Normalizes un text using stemming o lemmatization.

    Ejemplo:
        >>> normalize_text("Los gatos corrían rápidamente", method="stem")
        'los gat corr rapid'
        >>> normalize_text("Los gatos corrían rápidamente", method="lemma")
        'el gato correr rápidamente'

    Args:
        text: Text a normalizar
        method: 'stem' o 'lemma'
        language: Idioma

    Returns:
        Text normalizado
    """
    # TODO: Implement normalización de text
    # Hint: Usa if/else for elegir entre stemming y lemmatization
    # Cada método requiere un procesamiento diferente del text
    pass


