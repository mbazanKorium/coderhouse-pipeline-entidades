"""Mini-script de prueba asíncrono del pipeline.

Uso:
    python prueba_pipeline.py            # corre los 4 ejemplos en paralelo
    python prueba_pipeline.py "texto"    # procesa un texto propio
"""

import asyncio
import json
import logging
import sys

from chain import process_text

EJEMPLOS = {
    "log_de_error": (
        "2026-11-16 03:12:44 ERROR [orders-api] FastAPI worker 3: "
        "redis.exceptions.ConnectionError: Error 111 connecting to cache:6379. "
        "Fallback a PostgreSQL activado; psycopg.errors.TooManyConnections: "
        "remaining connection slots are reserved. 502 Bad Gateway devuelto por "
        "Nginx en el 40% de las peticiones durante 15 minutos."
    ),
    "arquitectura": (
        "El frontend en React consume una API GraphQL escrita en Node.js con "
        "Apollo Server, desplegada en AWS Lambda. Los datos viven en DynamoDB y "
        "los eventos se publican en Amazon SQS para que un worker en Python los "
        "procese. Proponemos agregar OpenTelemetry para mejorar la trazabilidad."
    ),
    # Prueba de estrés 1: texto ambiguo, sin nombres de tecnologías claros.
    "ambiguo": (
        "Desde ayer el sistema anda medio lento a veces, sobre todo cuando entra "
        "mucha gente. No sabemos si es la base o el servidor."
    ),
    # Prueba de estrés 2: no hay tecnologías. El esquema exige al menos una,
    # así que el validador debería rechazar la salida y la cadena reintentar.
    "sin_tecnologias": "Hoy hizo buen tiempo y la reunión de equipo terminó temprano.",
}


async def procesar(nombre: str, texto: str) -> None:
    try:
        resultado = await process_text(texto)
    except Exception as exc:
        print(f"\n=== {nombre} ===\nNo se pudo validar: {type(exc).__name__}: {exc}")
        return
    print(f"\n=== {nombre} ===")
    print(json.dumps(resultado.model_dump(mode="json"), ensure_ascii=False, indent=2))


async def main() -> None:
    if len(sys.argv) > 1:
        await procesar("texto_propio", " ".join(sys.argv[1:]))
        return
    await asyncio.gather(*(procesar(nombre, texto) for nombre, texto in EJEMPLOS.items()))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar el detalle HTTP de los SDK para que se lean los logs del pipeline.
    for ruidoso in ("httpx", "openai", "anthropic"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    asyncio.run(main())
