# Pipeline de Extracción de Entidades Técnicas

Pre-entrega 2 — AI Engineer (Coderhouse).

Recibe un párrafo de texto sin procesar (un log de error, una descripción de arquitectura…) y devuelve un objeto **validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico. Está construido con **LangChain (LCEL)**, usa `.with_structured_output()` y tiene **reintento automático** cuando el LLM devuelve un JSON mal formado o incompleto.

## Estructura

| Archivo | Qué contiene |
|---|---|
| `schemas.py` | Contrato de salida: modelo Pydantic `ExtraccionTecnica` y enum `NivelCriticidad` |
| `chain.py` | `ChatPromptTemplate`, cliente del LLM, cadena LCEL con `.with_retry()`, validación de `finish_reason` y la función asíncrona `process_text()` |
| `prueba_pipeline.py` | Mini-script de prueba asíncrono (4 ejemplos en paralelo, incluidas 2 pruebas de estrés) |
| `tests/test_pipeline.py` | Tests con un modelo falso: validan el esquema y los reintentos **sin gastar API** |

## Cómo funciona

```
ChatPromptTemplate                         ← texto + instrucciones de formato (variables de LangChain, sin f-strings)
  | model.with_structured_output(ExtraccionTecnica, include_raw=True)
  | validar_salida                         ← revisa finish_reason y errores de parseo/validación
.with_retry(stop_after_attempt=3)          ← reintenta ante JSON mal formado o incompleto
```

1. **Contrato (`schemas.py`)**
   - `tecnologias: list[str]`: al menos un elemento; se limpian los espacios y se quitan los duplicados (sin distinguir mayúsculas).
   - `nivel_de_criticidad`: enum `baja | media | alta`.
   - `resumen_tecnico: str`: entre 20 y 400 caracteres.
   - `extra="forbid"`: no se aceptan campos que no estén en el esquema.
2. **Prompt modular**: `ChatPromptTemplate` con dos variables, `{texto}` y `{instrucciones_formato}`. La segunda se inyecta con `.partial()`, así las reglas de formato se cambian sin tocar la cadena.
3. **Salida estructurada**: `with_structured_output(..., include_raw=True)` devuelve el objeto parseado **y** el mensaje crudo del LLM, que es donde viene el `finish_reason`.
4. **Detección de respuestas truncadas**: si el `finish_reason` es `length` (OpenAI) o `max_tokens` (Anthropic), se lanza `SalidaIncompletaError` **antes** de usar el objeto, aunque parezca válido.
5. **Resiliencia**: `.with_retry()` vuelve a ejecutar la cadena completa ante:
   - `SalidaIncompletaError`
   - `SalidaInvalidaError`
   - `pydantic.ValidationError`
   - `openai.LengthFinishReasonError`

   Hace hasta 3 intentos en total, con espera exponencial y *jitter*. Además, el cliente HTTP tiene sus propios `max_retries=2` para errores de red y *rate limit*.
6. **Logs**: cada etapa queda registrada con `logging`: texto recibido, error de validación (con el campo y el motivo), aviso de reintento, validación OK y tiempo total.

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y completa la API key
```

Variables de `.env`:

| Variable | Por defecto | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `LLM_MODEL` | `gpt-4.1-mini` / `claude-haiku-5-5` | Modelo a usar |
| `LLM_MAX_TOKENS` | `1024` | Límite de tokens de respuesta (bájalo, por ejemplo a `30`, para forzar respuestas truncadas y ver los reintentos) |
| `LLM_MAX_INTENTOS` | `3` | Intentos totales de la cadena |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | Clave del proveedor elegido |

## Uso

### Mini-script de prueba

```bash
python prueba_pipeline.py                   # los 4 ejemplos en paralelo
python prueba_pipeline.py "tu texto aquí"   # un texto propio
```

### Desde código

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("ERROR redis ConnectionError en la API FastAPI..."))
print(resultado.model_dump_json(indent=2))
```

### Tests (no necesitan API key)

```bash
pytest -q
```

## Ejemplo de salida

Entrada (log de error):

```
2026-11-16 03:12:44 ERROR [orders-api] FastAPI worker 3: redis.exceptions.ConnectionError:
Error 111 connecting to cache:6379. Fallback a PostgreSQL activado; psycopg.errors.TooManyConnections:
remaining connection slots are reserved. 502 Bad Gateway devuelto por Nginx en el 40% de las
peticiones durante 15 minutos.
```

Salida validada:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "Nginx"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API FastAPI perdió la conexión con Redis y el fallback a PostgreSQL agotó los slots de conexión, provocando errores 502 en Nginx en el 40% de las peticiones."
}
```

Logs de un caso en el que el modelo se recupera tras un reintento:

```
INFO    pipeline: Procesando texto (312 caracteres): '2026-11-16 03:12:44 ERROR [orders-api] FastAPI worker 3: re'...
WARNING pipeline: Salida inválida: tecnologias: List should have at least 1 item after validation, not 0. Se reintentará.
INFO    pipeline: Validación OK: 4 tecnologías, criticidad=alta
INFO    pipeline: Texto procesado en 3.87s
```

## Prueba de estrés

`prueba_pipeline.py` incluye dos textos pensados para poner a prueba el validador:

- **`ambiguo`**: describe lentitud sin nombrar tecnologías concretas. El modelo tiene que inferir con cuidado; si devuelve algo fuera del esquema, el validador lo rechaza y la cadena reintenta.
- **`sin_tecnologias`**: no tiene ninguna tecnología, pero el contrato exige al menos una.
  - Con Anthropic (o si el modelo no respeta `minItems`), la salida no pasa la validación y, tras agotar los reintentos, `process_text` lanza la excepción, que queda registrada en el log.
  - Con el modo estricto de OpenAI, el modelo está obligado a devolver al menos un elemento y puede "inventar" uno. Es una limitación conocida de forzar el esquema: el contrato garantiza la forma de la respuesta, no su veracidad.

Para ver los reintentos por truncado, ejecuta con `LLM_MAX_TOKENS=30`:

- Con Anthropic se registra `Respuesta truncada (finish_reason=max_tokens)` en cada intento, hasta que la cadena falla con `SalidaIncompletaError`.
- Con OpenAI, el propio SDK detecta el corte y lanza `LengthFinishReasonError`, que se registra como `Error en la llamada al LLM` y también se reintenta.
