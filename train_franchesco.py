import logging
from pathlib import Path
from app.trainer import TrainModule
from app.settings.app_settings import Settings
from app.settings.setup_logging import setup_logging

def main():
    # 1. Configuración de logs para ver qué ocurre internamente
    setup_logging()
    logger = logging.getLogger("TestValidation")

    # 2. Definir ruta del dataset pequeño
    dataset_small: Path = Settings.ASSETS_DIR / "recetas.txt"
    
    if not dataset_small.exists():
        logger.error(f"No se encontró el archivo en {dataset_small}")
        return

    # 3. Instanciar el Trainer
    # Usamos la configuración de Config directamente
    trainer = TrainModule(
        config=Settings, 
        dataset_path=dataset_small, 
        model_name=Settings.MODEL_NAME
        )

    # 4. (Opcional) Intentar cargar un checkpoint si quieres probar esa lógica
    # trainer.load_checkpoint(Config.MODEL_DIR / "checkpoint_epoch_2.pth")

    try:
        logger.info("Iniciando prueba de validación corta...")
        # Ejecutamos el entrenamiento
        trainer.train()
        logger.info("¡Validación exitosa! Franchesco gano la Piston Cup.")
        
    except Exception as e:
        logger.error(f"Error durante la validación: {e}", exc_info=True)
    except KeyboardInterrupt:f"validacion interrumpida porque el rayo McQueen le tranco la pista a Franchesco virgolini"

if __name__ == "__main__":
    main()
