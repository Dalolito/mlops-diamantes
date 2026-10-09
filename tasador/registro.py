import mlflow
import pandas as pd
from mlflow import MlflowClient

from tasador.config import (
    ALIAS_ANTERIOR, ALIAS_PRODUCCION, CARACTERISTICAS, CARPETA_ARTEFACTOS, EXPERIMENTO,
    NOMBRE_MODELO, OBJETIVO, PARAMETROS_LGB_LINEAL, RUTA_DATOS, URI_MLFLOW,
)
from tasador.modelo import Tasador

# MLFLOW GUARDA EL MODELO EN FORMATO SEGURO (SKOPS) Y SOLO ACEPTA LOS TIPOS QUE DECLARAMOS CONFIABLES
TIPOS_CONFIABLES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMRegressor",
    "tasador.modelo.Tasador",
    "tasador.preparacion.PreparacionDiamantes",
]

def configurar_mlflow() -> None:
    mlflow.set_tracking_uri(URI_MLFLOW)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        mlflow.create_experiment(EXPERIMENTO, artifact_location=CARPETA_ARTEFACTOS)
    mlflow.set_experiment(EXPERIMENTO)


def describir_tasador(nombre: str, tasador: Tasador) -> dict:
    estimador = tasador.pipeline.named_steps["modelo"]
    hiperparametros = {k: v for k, v in estimador.get_params().items() if k in PARAMETROS_LGB_LINEAL}
    return {
        "modelo": nombre,
        "objetivo": "precio por quilate" if tasador.precio_por_quilate else "precio",
        "columnas": len(tasador.pipeline.named_steps["preparacion"].columnas),
        **hiperparametros,
    }


# GUARDAMOS EN LA CORRIDA ACTIVA LOS DATOS USADOS (CON SU HUELLA), LOS PARÁMETROS, LAS MÉTRICAS Y EL MODELO
def guardar_corrida(nombre: str, tasador: Tasador, datos: pd.DataFrame, metricas: dict) -> str:
    dataset = mlflow.data.from_pandas(datos, source=str(RUTA_DATOS), name="diamantes", targets=OBJETIVO)
    mlflow.log_input(dataset, context="entrenamiento")
    mlflow.log_params(describir_tasador(nombre, tasador))
    mlflow.log_metrics(metricas)
    info = mlflow.sklearn.log_model(
        tasador,
        name="tasador",
        input_example=datos[CARACTERISTICAS].head(3),
        skops_trusted_types=TIPOS_CONFIABLES,
    )
    return info.model_uri


def registrar_version(uri_modelo: str, nombre_modelo: str) -> str:
    version = str(mlflow.register_model(uri_modelo, NOMBRE_MODELO).version)
    MlflowClient().set_model_version_tag(NOMBRE_MODELO, version, "modelo", nombre_modelo)
    return version


def version_con_alias(alias: str) -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


# LA VERSIÓN QUE ESTABA EN PRODUCCIÓN QUEDA COMO "ANTERIOR" PARA PODER VOLVER ATRÁS
def promover_a_produccion(version: str) -> None:
    cliente = MlflowClient()
    actual = version_con_alias(ALIAS_PRODUCCION)
    if actual is not None and actual != version:
        cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS_ANTERIOR, actual)
    cliente.set_registered_model_alias(NOMBRE_MODELO, ALIAS_PRODUCCION, version)


def cargar_produccion() -> tuple[Tasador, str]:
    version = version_con_alias(ALIAS_PRODUCCION)
    if version is None:
        raise RuntimeError("No hay un modelo en producción. Ejecuta primero: uv run entrenar")
    return mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@{ALIAS_PRODUCCION}"), version


def detalles_de_version(version: str) -> tuple[str, float]:
    cliente = MlflowClient()
    registro = cliente.get_model_version(NOMBRE_MODELO, version)
    rmse = cliente.get_run(registro.run_id).data.metrics["rmse"]
    return registro.tags["modelo"], rmse
