import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from tasador.config import PARAMETROS_LGB, PARAMETROS_LGB_LINEAL
from tasador.preparacion import COLUMNAS_ARBOL, COLUMNAS_ARBOL_LINEAL, PreparacionDiamantes


# EL TASADOR ENVUELVE LIMPIEZA + MODELO, PREDICE EN DÓLARES Y CONOCE EL RANGO DE PRECIOS CON QUE SE ENTRENÓ
class Tasador(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline: Pipeline, precio_por_quilate: bool = True):
        self.pipeline = pipeline
        self.precio_por_quilate = precio_por_quilate

    def fit(self, diamantes: pd.DataFrame, precios: pd.Series, sample_weight=None):
        divisor = diamantes["carat"] if self.precio_por_quilate else 1
        pesos = {} if sample_weight is None else {"modelo__sample_weight": np.asarray(sample_weight)}
        self.pipeline_ = clone(self.pipeline).fit(diamantes, precios / divisor, **pesos)
        self.precio_minimo_ = float(precios.min())
        self.precio_maximo_ = float(precios.max())
        return self

    def predecir_sin_recorte(self, diamantes: pd.DataFrame) -> np.ndarray:
        multiplicador = diamantes["carat"].to_numpy() if self.precio_por_quilate else 1
        return self.pipeline_.predict(diamantes) * multiplicador

    def predict(self, diamantes: pd.DataFrame) -> np.ndarray:
        return np.clip(self.predecir_sin_recorte(diamantes), self.precio_minimo_, self.precio_maximo_)

    def fuera_de_rango(self, diamantes: pd.DataFrame) -> np.ndarray:
        crudo = self.predecir_sin_recorte(diamantes)
        return (crudo < self.precio_minimo_) | (crudo > self.precio_maximo_)


def crear_baseline_carat() -> Tasador:
    pipeline = Pipeline([("preparacion", PreparacionDiamantes(["carat"])), ("modelo", LinearRegression())])
    return Tasador(pipeline, precio_por_quilate=False)


def crear_lgb_ppc() -> Tasador:
    pipeline = Pipeline([
        ("preparacion", PreparacionDiamantes(COLUMNAS_ARBOL)),
        ("modelo", LGBMRegressor(**PARAMETROS_LGB)),
    ])
    return Tasador(pipeline)


def crear_lgb_lineal_ppc() -> Tasador:
    pipeline = Pipeline([
        ("preparacion", PreparacionDiamantes(COLUMNAS_ARBOL_LINEAL)),
        ("modelo", LGBMRegressor(**PARAMETROS_LGB_LINEAL)),
    ])
    return Tasador(pipeline)


CATALOGO = {
    "Baseline carat": crear_baseline_carat,
    "LGB_ppc": crear_lgb_ppc,
    "LGB_lin_ppc": crear_lgb_lineal_ppc,
}


def calcular_metricas(precios_reales: pd.Series, precios_estimados: np.ndarray) -> dict[str, float]:
    reales = np.asarray(precios_reales, dtype=float)
    error = reales - precios_estimados
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / reales) * 100),
        "r2": float(1 - np.sum(error**2) / np.sum((reales - reales.mean()) ** 2)),
    }
