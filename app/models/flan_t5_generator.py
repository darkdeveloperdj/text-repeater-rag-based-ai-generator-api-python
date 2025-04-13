from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import torch

class FlanT5Generator:
    def __init__(self, model_name="google/flan-t5-small"):
        """
        Initialize the google/flan-t5-small model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

    def generate(self, prompt, max_length=200, temperature=0.7, top_p=0.9, system: str = None):
        """
        Generate a text response based on the provided prompt.

        Returns:
          str: The generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,  # enable sampling for variability
                temperature=temperature,
                top_p=top_p,
                num_beams=1,  # beams can be reduced when sampling is on to avoid deterministic beam search effects
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Always return a clean string
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return str(decoded).strip()
