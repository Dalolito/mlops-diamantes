import pandas as pd
from sklearn.model_selection import train_test_split

from tasador.config import PROPORCION_FUTURO, PROPORCION_VALIDACION, RUTA_DATOS, SEMILLA


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS)


# LA HISTORIA SE USA PARA ENTRENAR; EL FUTURO SON DIAMANTES QUE LLEGARÁN DESPUÉS Y NUNCA SE ENTRENA CON ELLOS
def dividir_historia_y_futuro(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(datos, test_size=PROPORCION_FUTURO, random_state=SEMILLA)


def dividir_entrenamiento_y_validacion(historia: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(historia, test_size=PROPORCION_VALIDACION, random_state=SEMILLA)
