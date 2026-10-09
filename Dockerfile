FROM python:3.13-slim

# LIGHTGBM NECESITA LA LIBRERÍA OPENMP DEL SISTEMA
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.11,<0.12"
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy

WORKDIR /app

# INSTALAMOS LAS VERSIONES EXACTAS DEL uv.lock
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY datos ./datos
COPY tasador ./tasador
RUN uv sync --locked --no-dev

# ENTRENAMOS DENTRO DE LA IMAGEN PARA QUE EL CONTENEDOR LLEGUE CON SU MODELO EN PRODUCCIÓN
RUN uv run --no-sync entrenar

# AL ARRANCAR, EL CONTENEDOR SIRVE LA API EN EL PUERTO 8000
ENV HOST_API=0.0.0.0
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "api"]
