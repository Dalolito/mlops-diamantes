import pandas as pd
import pytest

from tasador.config import CARACTERISTICAS, OBJETIVO
from tasador.datos import cargar_datos, dividir_entrenamiento_y_validacion, dividir_historia_y_futuro
from tasador.modelo import crear_lgb_lineal_ppc

DIAMANTE_DE_EJEMPLO = {
    "carat": 1.01, "cut": "Ideal", "color": "G", "clarity": "VS1",
    "depth": 61.8, "table": 57.0, "x": 6.42, "y": 6.46, "z": 3.98,
}


@pytest.fixture(scope="session")
def particion() -> tuple[pd.DataFrame, pd.DataFrame]:
    historia, _ = dividir_historia_y_futuro(cargar_datos())
    return dividir_entrenamiento_y_validacion(historia)


@pytest.fixture(scope="session")
def tasador_entrenado(particion):
    entrenamiento, _ = particion
    return crear_lgb_lineal_ppc().fit(entrenamiento[CARACTERISTICAS], entrenamiento[OBJETIVO])
