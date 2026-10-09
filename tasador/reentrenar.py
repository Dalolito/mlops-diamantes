import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from tasador.config import CARACTERISTICAS, OBJETIVO, PROPORCION_VENTAS_EVALUACION, SEMILLA
from tasador.datos import cargar_datos, dividir_historia_y_futuro
from tasador.modelo import CATALOGO, calcular_metricas
from tasador.registro import (
    cargar_produccion, configurar_mlflow, detalles_de_version, guardar_corrida,
    promover_a_produccion, registrar_version,
)
from tasador.tasaciones import leer_tasaciones


def main() -> None:
    configurar_mlflow()
    campeon, version_campeon = cargar_produccion()
    nombre_modelo, _ = detalles_de_version(version_campeon)

    # REUNIMOS LAS VENTAS RECIENTES: UNA PARTE PARA ENTRENAR Y OTRA PARA COMPARAR
    vendidos = leer_tasaciones(version_campeon).dropna(subset=["precio_real"])
    if vendidos.empty:
        print("No hay ventas nuevas con precio real: no se puede reentrenar.")
        return
    vendidos_entrenamiento, vendidos_evaluacion = train_test_split(
        vendidos, test_size=PROPORCION_VENTAS_EVALUACION, random_state=SEMILLA
    )

    # ACTUALIZAMOS LOS PRECIOS HISTÓRICOS AL NIVEL DEL MERCADO ACTUAL
    ajuste_de_mercado = float((vendidos_entrenamiento["precio_real"] / vendidos_entrenamiento["precio_estimado"]).median())
    historia, _ = dividir_historia_y_futuro(cargar_datos())
    historia_actualizada = historia.assign(**{OBJETIVO: historia[OBJETIVO] * ajuste_de_mercado})
    ventas_nuevas = vendidos_entrenamiento[CARACTERISTICAS].assign(**{OBJETIVO: vendidos_entrenamiento["precio_real"]})
    datos = pd.concat([historia_actualizada, ventas_nuevas], ignore_index=True)

    # ENTRENAMOS EL RETADOR CON LA MISMA RECETA DEL MODELO EN PRODUCCIÓN
    with mlflow.start_run(run_name=f"{nombre_modelo} reentrenado", tags={"etapa": "reentrenamiento"}):
        retador = CATALOGO[nombre_modelo]().fit(datos[CARACTERISTICAS], datos[OBJETIVO])

        # COMPARAMOS CAMPEÓN Y RETADOR CON VENTAS QUE NINGUNO DE LOS DOS VIO
        precios_reales = vendidos_evaluacion["precio_real"]
        metricas_retador = calcular_metricas(precios_reales, retador.predict(vendidos_evaluacion[CARACTERISTICAS]))
        rmse_campeon = calcular_metricas(precios_reales, campeon.predict(vendidos_evaluacion[CARACTERISTICAS]))["rmse"]

        mlflow.log_params({"version_campeon": version_campeon, "ajuste_de_mercado": round(ajuste_de_mercado, 3)})
        mlflow.log_metric("rmse_campeon", rmse_campeon)
        uri_modelo = guardar_corrida(nombre_modelo, retador, datos, metricas_retador)

    print(f"Ajuste de mercado estimado con las ventas recientes: {ajuste_de_mercado:.3f}")
    print(f"RMSE con ventas recientes: campeón (versión {version_campeon}) {rmse_campeon:.1f}, retador {metricas_retador['rmse']:.1f}.")

    # EL RETADOR SOLO PASA A PRODUCCIÓN SI LE GANA AL CAMPEÓN
    if metricas_retador["rmse"] < rmse_campeon:
        version = registrar_version(uri_modelo, nombre_modelo)
        promover_a_produccion(version)
        print(f"El retador gana: queda en producción como versión {version}; la versión {version_campeon} queda como 'anterior'.")
    else:
        print(f"El campeón sigue en producción (versión {version_campeon}).")


if __name__ == "__main__":
    main()
