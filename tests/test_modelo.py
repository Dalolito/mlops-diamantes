from tasador.config import CARACTERISTICAS, OBJETIVO, RMSE_MAXIMO_ACEPTABLE
from tasador.modelo import calcular_metricas, crear_baseline_carat


# PUERTA DE CALIDAD: SI EL MODELO EMPEORA, EL BUILD QUEDA EN ROJO Y NO SE PUBLICA
def test_pasa_la_puerta_de_calidad(tasador_entrenado, particion):
    _, validacion = particion
    rmse = calcular_metricas(validacion[OBJETIVO], tasador_entrenado.predict(validacion[CARACTERISTICAS]))["rmse"]
    assert rmse <= RMSE_MAXIMO_ACEPTABLE


def test_le_gana_al_baseline_de_carat(tasador_entrenado, particion):
    entrenamiento, validacion = particion
    baseline = crear_baseline_carat().fit(entrenamiento[CARACTERISTICAS], entrenamiento[OBJETIVO])
    rmse_modelo = calcular_metricas(validacion[OBJETIVO], tasador_entrenado.predict(validacion[CARACTERISTICAS]))["rmse"]
    rmse_baseline = calcular_metricas(validacion[OBJETIVO], baseline.predict(validacion[CARACTERISTICAS]))["rmse"]
    assert rmse_modelo < rmse_baseline / 2


def test_los_precios_quedan_en_el_rango_de_entrenamiento(tasador_entrenado, particion):
    _, validacion = particion
    precios = tasador_entrenado.predict(validacion[CARACTERISTICAS])
    assert precios.min() >= tasador_entrenado.precio_minimo_
    assert precios.max() <= tasador_entrenado.precio_maximo_
