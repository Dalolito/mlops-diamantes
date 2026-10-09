from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from tasador.config import HOST_API, PUERTO_API
from tasador.registro import cargar_produccion, configurar_mlflow
from tasador.tasaciones import guardar_tasaciones

RUTA_FORMULARIO = Path(__file__).parent / "formulario.html"
modelo_en_uso = {"tasador": None, "version": None}


class Diamante(BaseModel):
    carat: float = Field(gt=0, le=10, description="Peso en quilates", examples=[1.01])
    cut: Literal["Fair", "Good", "Very Good", "Premium", "Ideal"] = Field(examples=["Ideal"])
    color: Literal["J", "I", "H", "G", "F", "E", "D"] = Field(examples=["G"])
    clarity: Literal["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"] = Field(examples=["VS1"])
    depth: float = Field(gt=40, lt=80, description="Profundidad, % del diámetro", examples=[61.8])
    table: float = Field(gt=40, lt=80, description="Mesa, % del diámetro", examples=[57])
    x: float = Field(ge=0, le=60, description="Largo en mm (0 si no se midió)", examples=[6.42])
    y: float = Field(ge=0, le=60, description="Ancho en mm (0 si no se midió)", examples=[6.46])
    z: float = Field(ge=0, le=60, description="Profundidad en mm (0 si no se midió)", examples=[3.98])


class Tasacion(BaseModel):
    precio_estimado: float
    version_modelo: str
    revision_humana: bool


def cargar_modelo_en_produccion() -> str:
    modelo_en_uso["tasador"], modelo_en_uso["version"] = cargar_produccion()
    return modelo_en_uso["version"]


# AL ARRANCAR, LA API CARGA LA VERSIÓN QUE TENGA EL ALIAS "PRODUCCION" EN MLFLOW
@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    configurar_mlflow()
    cargar_modelo_en_produccion()
    yield


app = FastAPI(
    title="Tasador de diamantes",
    description="Sirve el modelo que está en producción en el registro de MLflow.",
    lifespan=ciclo_de_vida,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def formulario() -> str:
    return RUTA_FORMULARIO.read_text(encoding="utf-8")


@app.get("/salud")
def salud() -> dict:
    return {"estado": "ok", "version_modelo": modelo_en_uso["version"]}


# TASAMOS EL DIAMANTE Y GUARDAMOS LA TASACIÓN EN EL LOG PARA EL MONITOREO
@app.post("/tasar", response_model=Tasacion)
def tasar(diamante: Diamante) -> Tasacion:
    tasador, version = modelo_en_uso["tasador"], modelo_en_uso["version"]
    datos = pd.DataFrame([diamante.model_dump()])
    precio = float(tasador.predict(datos)[0])
    guardar_tasaciones(datos, [precio], version)
    return Tasacion(
        precio_estimado=round(precio, 2),
        version_modelo=version,
        revision_humana=bool(tasador.fuera_de_rango(datos)[0]),
    )


# CAMBIAMOS DE MODELO SIN APAGAR EL SERVICIO
@app.post("/recargar")
def recargar() -> dict:
    return {"version_modelo": cargar_modelo_en_produccion()}


def main() -> None:
    uvicorn.run(app, host=HOST_API, port=PUERTO_API)


if __name__ == "__main__":
    main()
