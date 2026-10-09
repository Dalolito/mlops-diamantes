import mlflow
import pandas as pd

from tasador.config import CARACTERISTICAS, OBJETIVO, RMSE_MAXIMO_ACEPTABLE
from tasador.datos import cargar_datos, dividir_entrenamiento_y_validacion, dividir_historia_y_futuro
from tasador.modelo import CATALOGO, calcular_metricas
from tasador.registro import configurar_mlflow, guardar_corrida, promover_a_produccion, registrar_version


def entrenar_candidato(nombre: str, crear, entrenamiento: pd.DataFrame, validacion: pd.DataFrame) -> dict:
    with mlflow.start_run(run_name=nombre, tags={"etapa": "entrenamiento"}):
        tasador = crear().fit(entrenamiento[CARACTERISTICAS], entrenamiento[OBJETIVO])
        metricas = calcular_metricas(validacion[OBJETIVO], tasador.predict(validacion[CARACTERISTICAS]))
        uri_modelo = guardar_corrida(nombre, tasador, entrenamiento, metricas)
    return {"modelo": nombre, "uri": uri_modelo, **metricas}


def main() -> None:
    # PREPARAMOS MLFLOW Y LOS DATOS
    configurar_mlflow()
    historia, _ = dividir_historia_y_futuro(cargar_datos())
    entrenamiento, validacion = dividir_entrenamiento_y_validacion(historia)

    # ENTRENAMOS CADA CANDIDATO Y LO DEJAMOS REGISTRADO EN MLFLOW
    resultados = [entrenar_candidato(n, crear, entrenamiento, validacion) for n, crear in CATALOGO.items()]
    tabla = pd.DataFrame(resultados).set_index("modelo")[["rmse", "mae", "mape", "r2"]]
    print("\nResultados en validación:\n" + tabla.round({"rmse": 1, "mae": 1, "mape": 2, "r2": 4}).to_string())

    # PUERTA DE CALIDAD: EL MEJOR SOLO PASA A PRODUCCIÓN SI SU ERROR ES ACEPTABLE
    mejor = min(resultados, key=lambda r: r["rmse"])
    if mejor["rmse"] > RMSE_MAXIMO_ACEPTABLE:
        raise SystemExit(f"Ningún modelo pasa la puerta de calidad (RMSE máximo {RMSE_MAXIMO_ACEPTABLE}).")

    # REGISTRAMOS EL MEJOR Y LO PASAMOS A PRODUCCIÓN
    version = registrar_version(mejor["uri"], mejor["modelo"])
    promover_a_produccion(version)
    print(f"\n{mejor['modelo']} queda en producción como versión {version} (RMSE {mejor['rmse']:.1f}).")


if __name__ == "__main__":
    main()
