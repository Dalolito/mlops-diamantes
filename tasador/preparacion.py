import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

DIMENSIONES = ["x", "y", "z"]

ORDEN_CALIDAD = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}

COLUMNAS_ARBOL = [
    "carat", "x", "y", "z", "depth", "table", "densidad_aparente",
    "cut_ord", "color_ord", "clarity_ord",
]
COLUMNAS_ARBOL_LINEAL = COLUMNAS_ARBOL + ["log_carat"]


def ajustar_modelo_log_log(carat: pd.Series, medida: pd.Series) -> LinearRegression:
    return LinearRegression().fit(np.log(carat.to_frame("carat")), np.log(medida))


def estimar_desde_carat(modelo: LinearRegression, carat: pd.Series) -> np.ndarray:
    return np.exp(modelo.predict(np.log(carat.to_frame("carat"))))


# MEDIDAS EN CERO: SE IMPUTAN CON LA GEOMETRÍA DEL DIAMANTE O DESDE SU PESO
def imputar_dimensiones(diamantes: pd.DataFrame, imputadores: dict) -> pd.DataFrame:
    diamantes = diamantes.copy()
    diamantes[DIMENSIONES] = diamantes[DIMENSIONES].replace(0, np.nan)

    diamantes["x"] = diamantes["x"].fillna(diamantes["y"])
    diamantes["y"] = diamantes["y"].fillna(diamantes["x"])

    solo_falta_z = diamantes["z"].isna() & diamantes["x"].notna()
    diametro = (diamantes["x"] + diamantes["y"]) / 2
    diamantes.loc[solo_falta_z, "z"] = diamantes["depth"] / 100 * diametro

    for dimension in DIMENSIONES:
        faltantes = diamantes[dimension].isna()
        if faltantes.any():
            diamantes.loc[faltantes, dimension] = estimar_desde_carat(
                imputadores[dimension], diamantes.loc[faltantes, "carat"]
            )
    return diamantes


# MEDIDAS ATÍPICAS: SE CORRIGEN CON LAS REGLAS FÍSICAS y ≈ x  Y  z ≈ depth × diámetro
def corregir_atipicos(diamantes: pd.DataFrame, modelo_diametro: LinearRegression) -> pd.DataFrame:
    diamantes = diamantes.copy()
    diametro_esperado = estimar_desde_carat(modelo_diametro, diamantes["carat"])

    x_e_y_distintas = ~(diamantes["y"] / diamantes["x"]).between(0.9, 1.1)
    y_es_la_mala = x_e_y_distintas & (
        (diamantes["y"] - diametro_esperado).abs() > (diamantes["x"] - diametro_esperado).abs()
    )
    x_es_la_mala = x_e_y_distintas & ~y_es_la_mala
    diamantes.loc[y_es_la_mala, "y"] = diamantes.loc[y_es_la_mala, "x"]
    diamantes.loc[x_es_la_mala, "x"] = diamantes.loc[x_es_la_mala, "y"]

    diametro = (diamantes["x"] + diamantes["y"]) / 2
    z_no_cuadra = ~(diamantes["z"] / (diamantes["depth"] / 100 * diametro)).between(0.9, 1.1)
    diametro_no_cuadra = (diametro / diametro_esperado - 1).abs() > 0.15
    depth_imposible = ~diamantes["depth"].between(50, 75)

    reconstruir_x_y = z_no_cuadra & diametro_no_cuadra
    recalcular_depth = z_no_cuadra & ~diametro_no_cuadra & depth_imposible
    recalcular_z = z_no_cuadra & ~diametro_no_cuadra & ~depth_imposible

    diametro_desde_z = diamantes["z"] / (diamantes["depth"] / 100)
    diamantes.loc[reconstruir_x_y, "x"] = diametro_desde_z
    diamantes.loc[reconstruir_x_y, "y"] = diametro_desde_z
    diamantes.loc[recalcular_depth, "depth"] = diamantes["z"] / diametro * 100
    diamantes.loc[recalcular_z, "z"] = diamantes["depth"] / 100 * diametro
    return diamantes


def crear_variables(diamantes: pd.DataFrame) -> pd.DataFrame:
    diamantes = diamantes.copy()
    diamantes["densidad_aparente"] = diamantes["carat"] / (diamantes["x"] * diamantes["y"] * diamantes["z"])
    diamantes["log_carat"] = np.log(diamantes["carat"])
    for columna, niveles in ORDEN_CALIDAD.items():
        diamantes[f"{columna}_ord"] = diamantes[columna].map({nivel: i for i, nivel in enumerate(niveles)})
    return diamantes


# TODA LA LIMPIEZA EN UN TRANSFORMADOR: VIAJA DENTRO DEL MODELO Y SE APLICA IGUAL AL ENTRENAR Y AL TASAR
class PreparacionDiamantes(BaseEstimator, TransformerMixin):
    def __init__(self, columnas: list[str] = COLUMNAS_ARBOL_LINEAL):
        self.columnas = columnas

    def fit(self, diamantes: pd.DataFrame, y=None):
        sanos = diamantes[(diamantes[DIMENSIONES] > 0).all(axis=1)]
        self.imputadores_ = {d: ajustar_modelo_log_log(sanos["carat"], sanos[d]) for d in DIMENSIONES}

        imputados = imputar_dimensiones(diamantes, self.imputadores_)
        redondos = imputados[(imputados["y"] / imputados["x"]).between(0.98, 1.02)]
        self.modelo_diametro_ = ajustar_modelo_log_log(redondos["carat"], (redondos["x"] + redondos["y"]) / 2)
        return self

    def transform(self, diamantes: pd.DataFrame) -> pd.DataFrame:
        limpios = corregir_atipicos(imputar_dimensiones(diamantes, self.imputadores_), self.modelo_diametro_)
        return crear_variables(limpios)[self.columnas]
