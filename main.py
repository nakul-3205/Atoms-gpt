import yaml
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipeline.train_pipeline import TrainPipeline

def load_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    logger.info("========== Atoms-GPT Training Started ==========")

    # Step 1 — Ingestion
    ingestion = DataIngestion(config)
    train_df, val_df = ingestion.initiate()

    # Step 2 — Transformation
    transformation = DataTransformation(config)
    train_loader, val_loader = transformation.get_dataloaders(train_df, val_df)

    # Step 3 — Train
    trainer = TrainPipeline(config)
    model = trainer.run(train_loader, val_loader)

    logger.info("========== Atoms-GPT Training Complete ==========")