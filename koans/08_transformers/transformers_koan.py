"""
Koan 08: Transformers - Models de Lenguaje Modernos

Los Transformers son la arquitectura más poderosa actual para NLP.

Models famosos:
- BERT: Bidirectional Encoder (comprensión)
- GPT: Generatestive Pre-trained (generación)
- T5: Text-to-Text Transfer Transformer

Usaremos Hugging Face Transformers.
"""

from transformers import pipeline, AutoTokenizer, AutoModel
from typing import List, Dict
import torch


def load_pretrained_pipeline(
    task: str, model: str = None, lang: str = "es"
) -> pipeline:
    """
    Loads un pipeline pre-entrenado de Hugging Face.

    Example:
        >>> pipe = load_pretrained_pipeline("sentiment-analysis")

    Args:
        task: Tarea (sentiment-analysis, ner, qa, etc.)
        model: Nombre del model (opcional)
        lang: Language

    Returns:
        Pipeline de Hugging Face
    """
    # TODO: Implement carga de pipeline
    # Hint: La librería transformers tiene una función pipeline muy útil
    pass


def extract_features_bert(
    text: str, model_name: str = "bert-base-multilingual-cased"
) -> torch.Tensor:
    """
    Extracts características using BERT.

    BERT genera representaciones contextuales de words.

    Example:
        >>> features = extract_features_bert("Python es genial")
        >>> features.shape
        torch.Size([1, 5, 768])  # [batch, tokens, hidden_size]

    Args:
        text: Text a procesar
        model_name: Nombre del model BERT

    Returns:
        Tensor con características
    """
    # TODO: Implement extracción de features con BERT
    # Hint: Necesitas tokenizar el text y pasarlo por el model
    # Consulta HINTS.md for the proceso completo
    pass


def question_answering(context: str, question: str, lang: str = "es") -> Dict:
    """
    Responde preguntas sobre un context using un model QA.

    Example:
        >>> context = "Python fue creado por Guido van Rossum en 1991"
        >>> question = "¿Quién creó Python?"
        >>> question_answering(context, question)
        {'answer': 'Guido van Rossum', 'score': 0.95}

    Args:
        context: Text de context
        question: Pregunta
        lang: Language

    Returns:
        Dictionary con respuesta y score
    """
    # TODO: Implement QA con transformers
    # Hint: Existe un pipeline específico para question-answering
    pass


def fill_mask(text: str, lang: str = "es") -> List[Dict]:
    """
    Rellena words enmascaradas en un text.

    Usa [MASK] o <mask> para marcar la palabra a predecir.

    Example:
        >>> fill_mask("Python es un [MASK] de programación")
        [{'token_str': 'lenguaje', 'score': 0.87}, ...]

    Args:
        text: Text con [MASK]
        lang: Language

    Returns:
        List of predicciones con scores
    """
    # TODO: Implement fill-mask
    # Hint: Pipeline de "fill-mask"
    pass


def zero_shot_classification(
    text: str, candidate_labels: List[str], lang: str = "es"
) -> Dict:
    """
    Classifies text sin entrenamiento específico.

    Zero-shot learning permite clasificar en categories nunca vistas.

    Example:
        >>> text = "Este código tiene un bug"
        >>> labels = ["problema", "éxito", "neutral"]
        >>> zero_shot_classification(text, labels)
        {'labels': ['problema', 'neutral', 'éxito'], 'scores': [0.89, 0.08, 0.03]}

    Args:
        text: Text a clasificar
        candidate_labels: Etiquetas posibles
        lang: Language

    Returns:
        Classifiesción con scores
    """
    # TODO: Implement zero-shot classification
    pass


def summarize_text(
    text: str, max_length: int = 130, min_length: int = 30, lang: str = "es"
) -> str:
    """
    Resume un text automáticamente.

    Example:
        >>> long_text = "Python es un lenguaje... (text largo)"
        >>> summarize_text(long_text)
        "Python es un lenguaje interpretado y de alto nivel..."

    Args:
        text: Text a resumir
        max_length: Longitud máxima del resumen
        min_length: Longitud mínima del resumen
        lang: Language

    Returns:
        Text resumido
    """
    # TODO: Implement summarization
    pass


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    """
    Traduce text entre Languages.

    Example:
        >>> translate_text("Hola mundo", source_lang="es", target_lang="en")
        "Hello world"

    Args:
        text: Text a traducir
        source_lang: Language origen
        target_lang: Language destino

    Returns:
        Text traducido
    """
    # TODO: Implement traducción
    # Hint: Busca models de traducción específicos for the par de Languages
    pass


def compare_models_performance(
    text: str, models: List[str], task: str = "sentiment-analysis"
) -> Dict[str, Dict]:
    """
    Comfor the rendimiento de diferentes models en la misma tarea.

    Example:
        >>> text = "Me encanta Python"
        >>> models = ["model1", "model2"]
        >>> compare_models_performance(text, models, "sentiment-analysis")
        {
            'model1': {'label': 'POSITIVE', 'score': 0.99, 'time': 0.5},
            'model2': {'label': 'POSITIVE', 'score': 0.95, 'time': 0.3}
        }

    Args:
        text: Text de prueba
        models: List of models a comparar
        task: Tarea a realizar

    Returns:
        Dictionary con resultados por model
    """
    # TODO: Implement comparación de models
    # Hint: Mide el tiempo de cada model y compara resultados
    pass


