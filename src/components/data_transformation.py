import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from src.logger import logger


class AtomsDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length):
        self.examples = []
        logger.info(f"Tokenizing {len(texts)} samples...")

        for text in texts:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=context_length + 1,
                padding='max_length',
                return_tensors='pt'
            )
            ids = encoded['input_ids'].squeeze()
            if ids.shape[0] == context_length + 1:
                self.examples.append(ids)

        logger.info(f"Total usable samples after tokenization: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]
        # x = all tokens except last, y = all tokens except first (next-token prediction)
        return ids[:-1], ids[1:]


class DataTransformation:
    def __init__(self, config: dict):
        self.context_length = config['model']['context_length']
        self.batch_size = config['training']['batch_size']
        self.processed_path = config['paths']['processed_data']
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)

        logger.info("Loading GPT-2 Tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_dataloaders(self, train_df, val_df):
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist()

        train_dataset = AtomsDataset(train_texts, self.tokenizer, self.context_length)
        val_dataset = AtomsDataset(val_texts, self.tokenizer, self.context_length)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
        return train_loader, val_loader

    def get_tokenizer(self):
        return self.tokenizer