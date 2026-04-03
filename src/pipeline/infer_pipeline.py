import torch
from transformers import GPT2Tokenizer
from src.components.model_factory import AtomsGPT
from src.logger import logger


class InferPipeline:
    def __init__(self, model_path: str, config: dict):
        self.device = torch.device("cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AtomsGPT(config)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

    def chat(self, prompt: str, max_tokens: int = 80):
        encoded = self.tokenizer(prompt, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, max_new_tokens=max_tokens,
                temperature=0.8, top_k=40
            )

        # Decode only newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()