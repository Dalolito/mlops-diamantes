import pandas as pd
import pytest

from tasador.config import CARACTERISTICAS
from tasador.preparacion import COLUMNAS_ARBOL_LINEAL, PreparacionDiamantes
from tests.conftest import DIAMANTE_DE_EJEMPLO


@pytest.fixture(scope="module")
def preparacion(particion) -> PreparacionDiamantes:
    entrenamiento, _ = particion
    return PreparacionDiamantes().fit(entrenamiento[CARACTERISTICAS])


def test_imputa_las_medidas_en_cero(preparacion):
    diamante = pd.DataFrame([{**DIAMANTE_DE_EJEMPLO, "x": 0, "y": 0, "z": 0}])
    medidas = preparacion.transform(diamante)[["x", "y", "z"]]
    assert (medidas > 0).all(axis=None)


def test_corrige_un_ancho_mal_digitado(preparacion):
    diamante = pd.DataFrame([{**DIAMANTE_DE_EJEMPLO, "y": 64.6}])
    preparado = preparacion.transform(diamante).iloc[0]
    assert preparado["y"] == pytest.approx(preparado["x"])


def test_entrega_las_columnas_que_espera_el_modelo(preparacion, particion):
    _, validacion = particion
    assert list(preparacion.transform(validacion.head()).columns) == COLUMNAS_ARBOL_LINEAL
