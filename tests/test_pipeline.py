"""Tests sin API key: un modelo falso simula respuestas buenas, rotas y truncadas
para comprobar la validación y los reintentos de la cadena."""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import (
    SalidaIncompletaError,
    SalidaInvalidaError,
    construir_cadena,
    process_text,
    prompt,
)
from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones.",
}


def respuesta_ok() -> dict:
    return {
        "raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        "parsed": ExtraccionTecnica(**VALIDO),
        "parsing_error": None,
    }


def respuesta_json_roto() -> dict:
    try:
        ExtraccionTecnica.model_validate({"tecnologias": [], "nivel_de_criticidad": "urgente"})
    except ValidationError as exc:
        error = exc
    return {
        "raw": AIMessage(content='{"tecnologias": [', response_metadata={"finish_reason": "stop"}),
        "parsed": None,
        "parsing_error": error,
    }


def respuesta_truncada() -> dict:
    return {
        "raw": AIMessage(content='{"tecnologias": ["Fast', response_metadata={"stop_reason": "max_tokens"}),
        "parsed": None,
        "parsing_error": None,
    }


class ModeloFalso:
    """Imita la parte de BaseChatModel que usa la cadena."""

    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def with_structured_output(self, schema, include_raw=False):
        assert schema is ExtraccionTecnica and include_raw

        def responder(_mensajes):
            self.llamadas += 1
            return self.respuestas.pop(0)()

        return RunnableLambda(responder)


# ---------------------------------------------------------------- esquema ---
def test_esquema_valido_normaliza():
    datos = {**VALIDO, "tecnologias": [" Redis ", "redis", "PostgreSQL", ""]}
    obj = ExtraccionTecnica.model_validate(datos)
    assert obj.tecnologias == ["Redis", "PostgreSQL"]
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA


@pytest.mark.parametrize(
    "cambio",
    [
        {"tecnologias": []},
        {"tecnologias": ["  "]},
        {"nivel_de_criticidad": "critica"},
        {"resumen_tecnico": "corto"},
        {"extra": "no permitido"},
    ],
)
def test_esquema_rechaza_datos_invalidos(cambio):
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, **cambio})


def test_prompt_usa_variables_de_langchain():
    assert prompt.input_variables == ["texto"]
    mensajes = prompt.format_messages(texto="log de prueba")
    assert "log de prueba" in mensajes[-1].content
    assert "nivel_de_criticidad" in mensajes[0].content


# ------------------------------------------------------------ reintentos ---
@pytest.mark.asyncio
async def test_exito_al_primer_intento():
    modelo = ModeloFalso([respuesta_ok])
    resultado = await process_text("texto", cadena=construir_cadena(modelo, max_intentos=3))
    assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
    assert modelo.llamadas == 1


@pytest.mark.asyncio
async def test_reintenta_json_mal_formado_y_se_recupera():
    modelo = ModeloFalso([respuesta_json_roto, respuesta_ok])
    resultado = await process_text("texto", cadena=construir_cadena(modelo, max_intentos=3))
    assert resultado.nivel_de_criticidad is NivelCriticidad.ALTA
    assert modelo.llamadas == 2


@pytest.mark.asyncio
async def test_detecta_respuesta_truncada_y_reintenta():
    modelo = ModeloFalso([respuesta_truncada, respuesta_ok])
    await process_text("texto", cadena=construir_cadena(modelo, max_intentos=3))
    assert modelo.llamadas == 2


@pytest.mark.asyncio
async def test_falla_al_agotar_reintentos():
    modelo = ModeloFalso([respuesta_json_roto, respuesta_truncada, respuesta_json_roto])
    with pytest.raises(SalidaInvalidaError):
        await process_text("texto", cadena=construir_cadena(modelo, max_intentos=3))
    assert modelo.llamadas == 3


@pytest.mark.asyncio
async def test_truncada_en_ultimo_intento():
    modelo = ModeloFalso([respuesta_truncada, respuesta_truncada])
    with pytest.raises(SalidaIncompletaError):
        await process_text("texto", cadena=construir_cadena(modelo, max_intentos=2))


@pytest.mark.asyncio
async def test_rechaza_texto_vacio():
    with pytest.raises(ValueError):
        await process_text("   ", cadena=construir_cadena(ModeloFalso([]), max_intentos=1))
