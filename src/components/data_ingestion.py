import os
import pandas as pd
from src.logger import logger


class DataIngestion:
    def __init__(self, config: dict):
        self.raw_path = config['paths']['raw_data']
        self.val_split = config['training']['val_split']

    def initiate(self):
        logger.info("=== Data Ingestion Started ===")

        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded CSV: {df.shape[0]} rows")

        # Validate
        assert 'text' in df.columns, "Column 'text' missing!"
        df.dropna(subset=['text'], inplace=True)
        df.drop_duplicates(subset=['text'], inplace=True)
        df['text'] = df['text'].str.strip()
        df = df[df['text'].str.len() > 10]  

        logger.info(f"After validation: {df.shape[0]} rows")

        # Train/val split
        split_idx = int(len(df) * (1 - self.val_split))
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)

        logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")
        logger.info("=== Data Ingestion Complete ===")

        return train_df, val_df