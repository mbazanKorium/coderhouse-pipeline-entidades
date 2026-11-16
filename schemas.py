"""Contrato de salida del pipeline de extracción de entidades técnicas.

Este modelo Pydantic es la "fuente de verdad": LangChain lo convierte en el
JSON Schema que recibe el LLM y, al volver la respuesta, lo usa para validarla.
Si la respuesta no cumple estas reglas, se lanza un ValidationError y la cadena
lo trata como un fallo reintentable.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    """Gravedad del problema o del riesgo descrito en el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log, descripción de arquitectura, etc.)."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos, servicios o "
            "herramientas mencionados explícitamente en el texto. Nunca vacía."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "'alta' si hay caída del servicio, pérdida de datos o riesgo de "
            "seguridad; 'media' si hay degradación o errores parciales; 'baja' "
            "si es informativo o una mejora."
        ),
    )
    # La longitud se controla en el validador (no con min_length/max_length)
    # porque el modo estricto de OpenAI no admite esas claves en strings.
    resumen_tecnico: str = Field(
        ...,
        description="Resumen técnico de una o dos frases, en español (20 a 400 caracteres).",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Quita espacios y vacíos, y elimina duplicados sin distinguir mayúsculas."""
        vistas: set[str] = set()
        limpias: list[str] = []
        for valor in valores:
            nombre = valor.strip()
            if nombre and nombre.lower() not in vistas:
                vistas.add(nombre.lower())
                limpias.append(nombre)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def limpiar_resumen(cls, valor: str) -> str:
        resumen = " ".join(valor.split())
        if not 20 <= len(resumen) <= 400:
            raise ValueError(f"el resumen debe tener entre 20 y 400 caracteres (tiene {len(resumen)})")
        return resumen
