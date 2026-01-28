from pathlib import Path
from typing import Optional

class Settings:
    # Datos del modelo
    MODEL_NAME: str = "Franchesco"
    MODEL_VERSION: str = "0.1.0"
    
    # Parametros del moodelo
    VOCAB_SIZE: int = 50257
    D_MODEL: int = 256
    MAX_SEQ_LEN: int = 256
    STRIDE: Optional[int] = 1
    EPOCHS: int = 5
    LEARNING_RATE: float = 3e-4
    N_LAYERS: int = 2
    NUM_HEADS: int = 8
    BATCH_SIZE: int = 8
    
    # Rutas/directorios
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    MODELS_DIR: Path = BASE_DIR / "models"
    CHECKPOINT_DIR: Path = BASE_DIR / "checkpoint"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    ASSETS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)