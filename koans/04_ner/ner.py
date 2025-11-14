"""
Koan 04: Named Entity Recognition (NER) - Rewithocimiento de Entidades Nombradas

NER identifica y clasifica entidades en el text:
- Personas (PER): "Juan García"
- Organizaciones (ORG): "Google", "ONU"
- Lugares (LOC): "Madrid", "España"
- Fechas/Tiempos (DATE/TIME)
- Cantidades de dinero (MONEY)
- etc.

Librería principal: spaCy (excelente for NER multilanguage)
"""

import spacy
from typing import List, Dict, Tuple


def extract_entities_spacy(text: str, lang: str = "es") -> List[Tuple[str, str]]:
    """
    Extracts todas las entidades nombradas de un text.

    Ejemplo:
        >>> extract_entities_spacy("Steve Jobs fundó Apple en California")
        [('Steve Jobs', 'PER'), ('Apple', 'ORG'), ('California', 'LOC')]

    Args:
        text: Text a analizar
        lang: Language ('es' o 'en')

    Returns:
        List of tuplas (entidad, tipo)
    """
    # TODO: Implement extracción de entidades with spaCy
    # Pistas:
    # 1. Carga el modelo spaCy apropiado
    # 2. Procesa el text
    # 3. Los documentos procesados tienen un atributo .ents with las entidades
    # Consulta HINTS.md for más ayuda
    pass


def extract_persons(text: str, lang: str = "es") -> List[str]:
    """
    Extracts solo nombres de personas.

    Ejemplo:
        >>> extract_persons("Juan y María trabajan en Google")
        ['Juan', 'María']

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        List of nombres de personas
    """
    # TODO: Filtra entidades de tipo PERSONA
    # Hint: Primero extrae todas las entidades, luego filtra por tipo
    pass


def extract_organizations(text: str, lang: str = "es") -> List[str]:
    """
    Extracts nombres de organizaciones.

    Ejemplo:
        >>> extract_organizations("Google y Microsoft son grandes empresas")
        ['Google', 'Microsoft']

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        List of organizaciones
    """
    # TODO: Filtra entidades de tipo ORGANIZACIÓN
    pass


def extract_locations(text: str, lang: str = "es") -> List[str]:
    """
    Extracts nombres de lugares.

    Ejemplo:
        >>> extract_locations("Viajé de Madrid a Barcelona")
        ['Madrid', 'Barcelona']

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        List of lugares
    """
    # TODO: Filtra entidades de tipo LUGAR
    # Nota: Puede ser 'LOC' o 'GPE' dependiendo del modelo
    pass


def extract_dates(text: str, lang: str = "es") -> List[str]:
    """
    Extracts expresiones de fecha y tiempo.

    Ejemplo:
        >>> extract_dates("El 15 de enero de 2024 fue un día importante")
        ['15 de enero de 2024']

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        List of expresiones temporales
    """
    # TODO: Filtra entidades de tipo FECHA/TIEMPO
    pass


def group_entities_by_type(text: str, lang: str = "es") -> Dict[str, List[str]]:
    """
    Agrupa las entidades por tipo.

    Ejemplo:
        >>> group_entities_by_type("Juan trabaja en Google en Madrid")
        {
            'PER': ['Juan'],
            'ORG': ['Google'],
            'LOC': ['Madrid']
        }

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        Diccionario with entidades agrupadas por tipo
    """
    # TODO: Implement agrupación de entidades
    # Hint: Recorre las entidades y agrúpalas por su label_
    pass


def count_entity_types(text: str, lang: str = "es") -> Dict[str, int]:
    """
    Cuenta cuántas entidades de cada tipo hay.

    Ejemplo:
        >>> count_entity_types("Juan y María trabajan en Google y Apple")
        {'PER': 2, 'ORG': 2}

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        Diccionario with withteo por tipo
    """
    # TODO: Cuenta las entidades por tipo
    # Hint: Similar a group_entities_by_type pero solo cuentas
    pass


def find_entity_withtext(
    text: str, entity: str, window: int = 5, lang: str = "es"
) -> str:
    """
    Encuentra el withtext alrededor de una entidad específica.

    Retorna las N words antes y después de la entidad.

    Ejemplo:
        >>> find_entity_withtext("Juan García trabaja en Google desde 2020", "Google", window=2)
        'en Google desde'

    Args:
        text: Text a analizar
        entity: Entidad a buscar
        window: Número de words de withtext (antes y después)
        lang: Language

    Returns:
        Context de la entidad
    """
    # TODO: Implement extracción de withtext
    # Pistas:
    # 1. Procesa el text with spaCy
    # 2. Busca la entidad específica en doc.ents
    # 3. Obtén los tokens alrededor using índices de span
    pass


def visualize_entities(text: str, lang: str = "es") -> str:
    """
    Crea una representación visual de las entidades (simplificada).

    Marca las entidades with corchetes y su tipo.

    Ejemplo:
        >>> visualize_entities("Juan trabaja en Google")
        '[PER: Juan] trabaja en [ORG: Google]'

    Args:
        text: Text a analizar
        lang: Language

    Returns:
        Text with entidades marcadas
    """
    # TODO: Implement visualización simple
    # Hint: Construye un nuevo string insertando marcadores alrededor de las entidades
    pass

