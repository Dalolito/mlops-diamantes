import os
from pathlib import Path

# RUTAS DEL PROYECTO
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "datos" / "train.csv"
RUTA_TASACIONES = RAIZ / "registros" / "tasaciones.csv"
RUTA_REPORTE_DRIFT = RAIZ / "reportes" / "drift.html"

# MLFLOW: DÓNDE SE GUARDAN LOS EXPERIMENTOS Y CÓMO SE LLAMA EL MODELO
URI_MLFLOW = f"sqlite:///{(RAIZ / 'mlflow.db').as_posix()}"
CARPETA_ARTEFACTOS = (RAIZ / "mlruns").as_uri()
EXPERIMENTO = "tasador-diamantes"
NOMBRE_MODELO = "tasador-diamantes"
ALIAS_PRODUCCION = "produccion"
ALIAS_ANTERIOR = "anterior"

# DATOS
SEMILLA = 42
PROPORCION_FUTURO = 0.2
PROPORCION_VALIDACION = 0.3
CARACTERISTICAS = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]
OBJETIVO = "price"

# HIPERPARÁMETROS ELEGIDOS EN EL NOTEBOOK DEL RETO 1
PARAMETROS_LGB = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=SEMILLA,
    n_jobs=-1,
    verbose=-1,
)
PARAMETROS_LGB_LINEAL = {**PARAMETROS_LGB, "linear_tree": True, "linear_lambda": 0.01}

# PUERTA DE CALIDAD: UN MODELO CON MÁS ERROR QUE ESTE NO PUEDE LLEGAR A PRODUCCIÓN
RMSE_MAXIMO_ACEPTABLE = 560

# MONITOREO: CUÁNDO SE CONSIDERA QUE EL MODELO NECESITA REENTRENARSE
PROPORCION_MAXIMA_COLUMNAS_CON_DRIFT = 0.5
DEGRADACION_MAXIMA_RMSE = 1.2

# SIMULACIÓN DEL MERCADO DE 2027: MÁS DIAMANTES GRANDES Y PRECIOS MÁS ALTOS
TAMANO_LOTE_2027 = 2000
CARAT_DIAMANTE_GRANDE = 1.0
PROPORCION_GRANDES_2027 = 0.7
INFLACION_2027 = 0.15

# REENTRENAMIENTO: PARTE DE LAS VENTAS RECIENTES SE GUARDA PARA COMPARAR CAMPEÓN Y RETADOR
PROPORCION_VENTAS_EVALUACION = 0.5

# API
HOST_API = os.getenv("HOST_API", "127.0.0.1")
PUERTO_API = int(os.getenv("PUERTO_API", "8000"))
