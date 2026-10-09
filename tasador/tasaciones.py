from datetime import datetime

import numpy as np
import pandas as pd

from tasador import config
from tasador.config import CARACTERISTICAS


# CADA TASACIÓN QUEDA EN UN LOG; CUANDO EL DIAMANTE SE VENDE, SE CONOCE SU PRECIO REAL
def guardar_tasaciones(
    diamantes: pd.DataFrame,
    precios_estimados: np.ndarray,
    version: str,
    precios_reales: pd.Series | None = None,
) -> None:
    registro = diamantes[CARACTERISTICAS].copy()
    registro.insert(0, "fecha", datetime.now().isoformat(timespec="seconds"))
    registro["precio_estimado"] = np.round(precios_estimados, 2)
    registro["version_modelo"] = version
    registro["precio_real"] = np.nan if precios_reales is None else precios_reales.to_numpy()

    ruta = config.RUTA_TASACIONES
    ruta.parent.mkdir(parents=True, exist_ok=True)
    registro.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)


def leer_tasaciones(version: str) -> pd.DataFrame:
    if not config.RUTA_TASACIONES.exists():
        return pd.DataFrame(columns=["fecha", *CARACTERISTICAS, "precio_estimado", "version_modelo", "precio_real"])
    tasaciones = pd.read_csv(config.RUTA_TASACIONES, dtype={"version_modelo": str})
    return tasaciones[tasaciones["version_modelo"] == version]
