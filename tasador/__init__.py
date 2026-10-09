import logging
import os
import warnings

# SILENCIAMOS LOS MENSAJES INFORMATIVOS DE MLFLOW PARA QUE LA SALIDA DE CADA PASO SEA LEGIBLE
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_LOGGING_LEVEL", "ERROR")
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=SyntaxWarning)
