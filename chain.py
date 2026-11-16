"""Pipeline de extracción de entidades técnicas con LangChain (LCEL).

Flujo:

    ChatPromptTemplate
      | modelo.with_structured_output(ExtraccionTecnica, include_raw=True)
      | validar_salida            (revisa finish_reason y errores de parseo)
    .with_retry(...)              (reintenta si el JSON vino mal formado o incompleto)

Configuración por variables de entorno (ver .env.example):
    LLM_PROVIDER    openai | anthropic             (por defecto: openai)
    LLM_MODEL       nombre del modelo              (por defecto según proveedor)
    LLM_MAX_TOKENS  límite de tokens de respuesta  (por defecto: 1024)
    LLM_MAX_INTENTOS intentos totales de la cadena (por defecto: 3)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import ExtraccionTecnica

load_dotenv()

logger = logging.getLogger("pipeline")

MODELOS_POR_DEFECTO = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
}

# Valores de finish_reason (OpenAI) / stop_reason (Anthropic) que indican que el
# modelo cortó la respuesta por falta de tokens.
FINISH_REASONS_TRUNCADOS = {"length", "max_tokens"}


# --------------------------------------------------------------------------- #
# Errores propios: son los que disparan el reintento automático
# --------------------------------------------------------------------------- #
class SalidaIncompletaError(Exception):
    """El LLM cortó la respuesta (finish_reason = length / max_tokens)."""


class SalidaInvalidaError(Exception):
    """El LLM devolvió un JSON mal formado o que no cumple el esquema Pydantic."""


def _errores_reintentables() -> tuple[type[BaseException], ...]:
    errores: list[type[BaseException]] = [
        SalidaIncompletaError,
        SalidaInvalidaError,
        ValidationError,
    ]
    try:
        # El SDK de OpenAI lanza esta excepción él mismo cuando la respuesta
        # estructurada se corta por tokens, antes de llegar a validar_salida.
        from openai import LengthFinishReasonError

        errores.append(LengthFinishReasonError)
    except ImportError:
        pass
    return tuple(errores)


# --------------------------------------------------------------------------- #
# Prompt template (sin f-strings: las variables las gestiona LangChain)
# --------------------------------------------------------------------------- #
INSTRUCCIONES_FORMATO = """\
- "tecnologias": solo nombres propios de tecnologías mencionadas en el texto \
(lenguajes, frameworks, bases de datos, servicios cloud, herramientas). \
No inventes tecnologías que no aparezcan. Usa el nombre oficial (ej. "PostgreSQL", no "postgres").
- "nivel_de_criticidad": exactamente uno de "baja", "media" o "alta".
  * alta: caída del servicio, pérdida o corrupción de datos, riesgo de seguridad.
  * media: degradación de rendimiento, errores intermitentes o parciales.
  * baja: información, advertencias menores, propuestas de mejora.
- "resumen_tecnico": una o dos frases en español (máx. 400 caracteres) que \
expliquen qué pasa y dónde está el problema o el riesgo principal."""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un ingeniero de software senior especializado en análisis de "
            "incidentes y arquitectura. Tu tarea es extraer información técnica "
            "estructurada del texto que te dé el usuario.\n\n"
            "Instrucciones de formato:\n{instrucciones_formato}",
        ),
        ("human", "Texto a analizar:\n<<<\n{texto}\n>>>"),
    ]
).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def crear_modelo() -> BaseChatModel:
    """Crea el cliente del LLM según LLM_PROVIDER (openai o anthropic)."""
    proveedor = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    nombre = os.getenv("LLM_MODEL") or MODELOS_POR_DEFECTO.get(proveedor)
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        modelo: BaseChatModel = ChatOpenAI(
            model=nombre, temperature=0, max_tokens=max_tokens, max_retries=2
        )
    elif proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        modelo = ChatAnthropic(
            model=nombre, temperature=0, max_tokens=max_tokens, max_retries=2
        )
    else:
        raise ValueError(f"LLM_PROVIDER no soportado: {proveedor!r} (usa openai o anthropic)")

    logger.info("Modelo configurado: proveedor=%s modelo=%s max_tokens=%s", proveedor, nombre, max_tokens)
    return modelo


# --------------------------------------------------------------------------- #
# Validación de la salida cruda del LLM
# --------------------------------------------------------------------------- #
def _finish_reason(raw: Any) -> str | None:
    if not isinstance(raw, AIMessage):
        return None
    meta = raw.response_metadata or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def validar_salida(resultado: dict[str, Any]) -> ExtraccionTecnica:
    """Recibe {"raw", "parsed", "parsing_error"} y devuelve el objeto validado.

    Lanza SalidaIncompletaError / SalidaInvalidaError para que .with_retry()
    vuelva a ejecutar la cadena completa.
    """
    raw = resultado.get("raw")
    parsed = resultado.get("parsed")
    error = resultado.get("parsing_error")
    finish_reason = _finish_reason(raw)

    logger.debug("Respuesta recibida: finish_reason=%s", finish_reason)

    # 1) Primero el finish_reason: si se cortó, el objeto está incompleto aunque
    #    por casualidad parezca parseable.
    if finish_reason in FINISH_REASONS_TRUNCADOS:
        logger.warning(
            "Respuesta truncada (finish_reason=%s). Se reintentará; si persiste, "
            "sube LLM_MAX_TOKENS.",
            finish_reason,
        )
        raise SalidaIncompletaError(f"respuesta cortada por tokens (finish_reason={finish_reason})")

    # 2) Error de parseo o de validación Pydantic.
    if error is not None:
        logger.warning("Salida inválida: %s. Se reintentará.", _resumir_error(error))
        raise SalidaInvalidaError(str(error)) from error

    # 3) El modelo no llamó a la herramienta / no devolvió nada.
    if parsed is None:
        logger.warning("El modelo no devolvió un objeto estructurado. Se reintentará.")
        raise SalidaInvalidaError("el modelo no devolvió salida estructurada")

    # Por si el proveedor devuelve un dict en lugar del modelo Pydantic.
    if not isinstance(parsed, ExtraccionTecnica):
        parsed = ExtraccionTecnica.model_validate(parsed)

    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(parsed.tecnologias),
        parsed.nivel_de_criticidad.value,
    )
    return parsed


def _resumir_error(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(raíz)'}: {e['msg']}" for e in error.errors()
        )
    texto = str(error).replace("\n", " ")
    return texto[:300]


def _registrar_error_del_modelo(run: Any) -> None:
    """Algunos errores (ej. LengthFinishReasonError del SDK de OpenAI) se lanzan
    dentro de la llamada al modelo y no llegan a validar_salida: los logueamos aquí."""
    error = str(getattr(run, "error", "") or "").strip().splitlines()
    logger.warning("Error en la llamada al LLM: %s. Se reintentará si es reintentable.", error[-1] if error else "?")


# --------------------------------------------------------------------------- #
# Cadena LCEL
# --------------------------------------------------------------------------- #
def construir_cadena(
    modelo: BaseChatModel | None = None,
    max_intentos: int | None = None,
) -> Runnable[dict[str, str], ExtraccionTecnica]:
    """Compone prompt | LLM estructurado | validación, con reintento automático."""
    modelo = modelo or crear_modelo()
    max_intentos = max_intentos or int(os.getenv("LLM_MAX_INTENTOS", "3"))

    # include_raw=True: además del objeto parseado recibimos el AIMessage crudo,
    # que trae el finish_reason. Los errores de parseo no se lanzan: llegan en
    # "parsing_error" y validar_salida decide qué hacer.
    modelo_estructurado = modelo.with_structured_output(
        ExtraccionTecnica, include_raw=True
    ).with_listeners(on_error=_registrar_error_del_modelo)

    cadena = prompt | modelo_estructurado | RunnableLambda(validar_salida)

    return cadena.with_retry(
        retry_if_exception_type=_errores_reintentables(),
        stop_after_attempt=max_intentos,
        wait_exponential_jitter=True,
    )


_cadena: Runnable[dict[str, str], ExtraccionTecnica] | None = None


def _obtener_cadena() -> Runnable[dict[str, str], ExtraccionTecnica]:
    global _cadena
    if _cadena is None:
        _cadena = construir_cadena()
    return _cadena


async def process_text(
    text: str,
    cadena: Runnable[dict[str, str], ExtraccionTecnica] | None = None,
) -> ExtraccionTecnica:
    """Extrae las entidades técnicas de `text` y devuelve un objeto validado.

    Lanza la última excepción si se agotan los reintentos.
    """
    if not text or not text.strip():
        raise ValueError("el texto de entrada está vacío")

    cadena = cadena or _obtener_cadena()
    vista_previa = " ".join(text.split())[:60]
    logger.info("Procesando texto (%d caracteres): %r...", len(text), vista_previa)

    inicio = time.perf_counter()
    try:
        resultado = await cadena.ainvoke({"texto": text})
    except Exception as exc:
        logger.error(
            "Falló tras agotar los reintentos (%.2fs): %s: %s",
            time.perf_counter() - inicio,
            type(exc).__name__,
            _resumir_error(exc),
        )
        raise

    logger.info("Texto procesado en %.2fs", time.perf_counter() - inicio)
    return resultado
