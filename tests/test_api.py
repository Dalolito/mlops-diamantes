import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tasador import api, config
from tests.conftest import DIAMANTE_DE_EJEMPLO


# SIN EL BLOQUE `with`, TESTCLIENT NO EJECUTA EL ARRANQUE QUE CARGA EL MODELO DESDE MLFLOW
@pytest.fixture
def cliente(tasador_entrenado, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config, "RUTA_TASACIONES", tmp_path / "tasaciones.csv")
    monkeypatch.setitem(api.modelo_en_uso, "tasador", tasador_entrenado)
    monkeypatch.setitem(api.modelo_en_uso, "version", "prueba")
    return TestClient(api.app)


def test_tasa_un_diamante(cliente):
    respuesta = cliente.post("/tasar", json=DIAMANTE_DE_EJEMPLO)
    assert respuesta.status_code == 200
    assert respuesta.json()["precio_estimado"] > 0
    assert respuesta.json()["version_modelo"] == "prueba"


def test_rechaza_un_diamante_invalido(cliente):
    respuesta = cliente.post("/tasar", json={**DIAMANTE_DE_EJEMPLO, "cut": "Bueno", "carat": -1})
    assert respuesta.status_code == 422


def test_guarda_cada_tasacion_para_el_monitoreo(cliente):
    cliente.post("/tasar", json=DIAMANTE_DE_EJEMPLO)
    cliente.post("/tasar", json=DIAMANTE_DE_EJEMPLO)
    assert len(pd.read_csv(config.RUTA_TASACIONES)) == 2


def test_sirve_el_formulario(cliente):
    respuesta = cliente.get("/")
    assert respuesta.status_code == 200
    assert "Tasador de diamantes" in respuesta.text
