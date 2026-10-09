import math

import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset

from tasador import reentrenar
from tasador.config import (
    ALIAS_PRODUCCION, CARACTERISTICAS, DEGRADACION_MAXIMA_RMSE, PROPORCION_MAXIMA_COLUMNAS_CON_DRIFT,
    RUTA_REPORTE_DRIFT,
)
from tasador.datos import cargar_datos, dividir_historia_y_futuro
from tasador.modelo import calcular_metricas
from tasador.registro import configurar_mlflow, detalles_de_version, version_con_alias
from tasador.tasaciones import leer_tasaciones


# EVIDENTLY ELIGE UNA PRUEBA POR COLUMNA: CON P-VALOR HAY DRIFT SI ES BAJO; CON DISTANCIA, SI ES ALTA
def columnas_con_drift(resultado) -> list[str]:
    columnas = []
    for metrica in resultado.dict()["metrics"]:
        configuracion = metrica["config"]
        if "column" not in configuracion:
            continue
        if "p_value" in configuracion["method"]:
            hay_drift = metrica["value"] < configuracion["threshold"]
        else:
            hay_drift = metrica["value"] >= configuracion["threshold"]
        if hay_drift:
            columnas.append(configuracion["column"])
    return columnas


def main() -> None:
    configurar_mlflow()
    version = version_con_alias(ALIAS_PRODUCCION)
    nombre_modelo, rmse_de_referencia = detalles_de_version(version)
    tasaciones = leer_tasaciones(version)
    if tasaciones.empty:
        print(f"Todavía no hay tasaciones del modelo versión {version}.")
        return

    # COMPARAMOS LOS DIAMANTES QUE LLEGAN CON LOS DE ENTRENAMIENTO
    historia, _ = dividir_historia_y_futuro(cargar_datos())
    resultado = Report([DataDriftPreset()]).run(tasaciones[CARACTERISTICAS], historia[CARACTERISTICAS])
    RUTA_REPORTE_DRIFT.parent.mkdir(parents=True, exist_ok=True)
    resultado.save_html(str(RUTA_REPORTE_DRIFT))
    con_drift = columnas_con_drift(resultado)
    proporcion_con_drift = len(con_drift) / len(CARACTERISTICAS)

    # MEDIMOS EL ERROR CON LOS PRECIOS REALES DE LOS DIAMANTES YA VENDIDOS
    vendidos = tasaciones.dropna(subset=["precio_real"])
    rmse_reciente = (
        calcular_metricas(vendidos["precio_real"], vendidos["precio_estimado"].to_numpy())["rmse"]
        if len(vendidos) else math.nan
    )

    # DECIDIMOS SI HAY QUE REENTRENAR
    alerta_por_drift = proporcion_con_drift >= PROPORCION_MAXIMA_COLUMNAS_CON_DRIFT
    alerta_por_error = rmse_reciente > rmse_de_referencia * DEGRADACION_MAXIMA_RMSE

    # DEJAMOS EL MONITOREO REGISTRADO EN MLFLOW JUNTO CON EL REPORTE
    with mlflow.start_run(run_name=f"Monitoreo versión {version}", tags={"etapa": "monitoreo"}):
        mlflow.log_params({
            "version_monitoreada": version,
            "modelo": nombre_modelo,
            "columnas_con_drift": ", ".join(con_drift) or "ninguna",
        })
        mlflow.log_metrics({
            "tasaciones": len(tasaciones),
            "vendidos": len(vendidos),
            "proporcion_columnas_con_drift": proporcion_con_drift,
            "rmse_de_referencia": rmse_de_referencia,
            "rmse_reciente": rmse_reciente,
        })
        mlflow.log_artifact(str(RUTA_REPORTE_DRIFT))

    print(f"Modelo en producción: {nombre_modelo}, versión {version}.")
    print(f"Columnas con drift: {', '.join(con_drift) or 'ninguna'} ({proporcion_con_drift:.0%}).")
    print(f"RMSE: {rmse_de_referencia:.1f} en validación, {rmse_reciente:.1f} con las ventas recientes.")
    print(f"Reporte de drift: {RUTA_REPORTE_DRIFT}")

    # SI HAY ALERTA, EL REENTRENAMIENTO SE DISPARA SOLO
    if alerta_por_drift or alerta_por_error:
        motivos = [m for m, activo in [("drift en los datos", alerta_por_drift), ("aumento del error", alerta_por_error)] if activo]
        print(f"\nALERTA ({' y '.join(motivos)}): se lanza el reentrenamiento.\n")
        reentrenar.main()
    else:
        print("\nSin alertas: el modelo sigue en producción.")


if __name__ == "__main__":
    main()
