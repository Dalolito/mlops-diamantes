import pandas as pd

from tasador.config import (
    CARACTERISTICAS, CARAT_DIAMANTE_GRANDE, INFLACION_2027, OBJETIVO,
    PROPORCION_GRANDES_2027, SEMILLA, TAMANO_LOTE_2027,
)
from tasador.datos import cargar_datos, dividir_historia_y_futuro
from tasador.registro import cargar_produccion, configurar_mlflow
from tasador.tasaciones import guardar_tasaciones


def crear_lote_2027(futuro: pd.DataFrame) -> pd.DataFrame:
    es_grande = futuro["carat"] >= CARAT_DIAMANTE_GRANDE
    cantidad_grandes = int(TAMANO_LOTE_2027 * PROPORCION_GRANDES_2027)
    grandes = futuro[es_grande].sample(cantidad_grandes, random_state=SEMILLA)
    pequenos = futuro[~es_grande].sample(TAMANO_LOTE_2027 - cantidad_grandes, random_state=SEMILLA)

    lote = pd.concat([grandes, pequenos]).sample(frac=1, random_state=SEMILLA)
    lote[OBJETIVO] = (lote[OBJETIVO] * (1 + INFLACION_2027)).round()
    return lote


def main() -> None:
    # TOMAMOS DIAMANTES QUE EL MODELO NUNCA VIO
    configurar_mlflow()
    historia, futuro = dividir_historia_y_futuro(cargar_datos())

    # SIMULAMOS EL MERCADO DE 2027: MÁS DIAMANTES GRANDES Y PRECIOS MÁS ALTOS
    lote = crear_lote_2027(futuro)

    # EL MODELO EN PRODUCCIÓN LOS TASA, COMO SI LLEGARAN POR LA API, Y DESPUÉS SE VENDEN A SU PRECIO REAL
    tasador, version = cargar_produccion()
    guardar_tasaciones(lote, tasador.predict(lote[CARACTERISTICAS]), version, precios_reales=lote[OBJETIVO])

    grandes_antes = (historia["carat"] >= CARAT_DIAMANTE_GRANDE).mean()
    grandes_ahora = (lote["carat"] >= CARAT_DIAMANTE_GRANDE).mean()
    print(f"Se registraron {len(lote)} tasaciones de 2027 hechas por el modelo versión {version}.")
    print(f"Diamantes de {CARAT_DIAMANTE_GRANDE:g} quilate o más: {grandes_antes:.0%} en la historia, {grandes_ahora:.0%} en 2027.")
    print(f"Los precios reales de venta subieron un {INFLACION_2027:.0%}.")


if __name__ == "__main__":
    main()
