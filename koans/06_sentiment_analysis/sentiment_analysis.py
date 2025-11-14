"""
Koan 06: Sentiment Analysis - Análisis de Sentimientos

El análisis de sentimientos determina si un text es positivo, negativo o neutral.

Usaremos models de Transformers pre-entrenados de Hugging Face.

Ejemplos:
- "Me encanta Python!" → POSITIVO
- "Odio los bugs" → NEGATIVO
- "Python es un lenguaje" → NEUTRAL
"""

from transformers import pipeline
from typing import Dict, List


def analyze_sentiment_simple(text: str, lang: str = "es") -> Dict:
    """
    Analyzes el sentimiento de un text using un model pre-entrenado.

    Example:
        >>> analyze_sentiment_simple("Me encanta Python!")
        {'label': 'POSITIVE', 'score': 0.9987}

    Args:
        text: Text a analizar
        lang: Idioma ('es' o 'en')

    Returns:
        Dictionary with thebel y score
    """
    # TODO: Implement análisis de sentimientos con transformers
    # Hint: Usa pipeline de transformers para "sentiment-analysis"
    # Consulta HINTS.md para más detalles
    pass


def analyze_sentiment_batch(texts: List[str], lang: str = "es") -> List[Dict]:
    """
    Analyzes el sentimiento de múltiples texts.

    Example:
        >>> texts = ["Me gusta", "No me gusta"]
        >>> analyze_sentiment_batch(texts)
        [{'label': 'POSITIVE', ...}, {'label': 'NEGATIVE', ...}]

    Args:
        texts: List of texts
        lang: Idioma

    Returns:
        List of resultados
    """
    # TODO: Procesa múltiples texts
    # Hint: Los pipelines pueden procesar listas de texts
    pass


def get_sentiment_label(text: str, lang: str = "es") -> str:
    """
    Retorna solo la etiqueta del sentimiento (sin score).

    Example:
        >>> get_sentiment_label("Excelente producto")
        'POSITIVE'

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        Etiqueta de sentimiento
    """
    # TODO: Use la función anterior y extrae solo el label
    pass


def get_sentiment_score(text: str, lang: str = "es") -> float:
    """
    Retorna solo el score de confianza del sentimiento.

    Example:
        >>> get_sentiment_score("Me encanta!")
        0.9987

    Args:
        text: Text a analizar
        lang: Idioma

    Returns:
        Score de confianza (0-1)
    """
    # TODO: Extracts solo el score
    pass


def classify_sentiment_simple(text: str) -> str:
    """
    Classifies sentimiento en categories simples: positivo, negativo, neutral.

    Example:
        >>> classify_sentiment_simple("Me gusta Python")
        'positivo'

    Args:
        text: Text a analizar

    Returns:
        'positivo', 'negativo', o 'neutral'
    """
    # TODO: Implement clasificación simple
    # Mapea los resultados a categories en español
    pass


def analyze_text_emotions(text: str) -> Dict[str, float]:
    """
    Analyzes emociones específicas en el text.

    Puede detectar: alegría, tristeza, enojo, miedo, sorpresa.

    Example:
        >>> analyze_text_emotions("Estoy muy feliz!")
        {'joy': 0.95, 'sadness': 0.01, ...}

    Args:
        text: Text a analizar

    Returns:
        Dictionary con emociones y scores
    """
    # TODO: Implement análisis de emociones
    # Hint: Necesitas un model específico de emociones
    pass


def sentiment_statistics(texts: List[str]) -> Dict:
    """
    Calculates estadísticas de sentimiento sobre múltiples texts.

    Example:
        >>> texts = ["Me gusta", "Odio esto", "Es normal"]
        >>> sentiment_statistics(texts)
        {
            'total': 3,
            'positive': 1,
            'negative': 1,
            'neutral': 1,
            'avg_score': 0.65
        }

    Args:
        texts: List of texts

    Returns:
        Dictionary con estadísticas
    """
    # TODO: Calculates estadísticas agregadas
    # Analyzes todos los texts y cuenta por categoría
    pass
