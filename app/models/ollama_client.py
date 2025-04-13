import requests

class OllamaClient:
    #    def __init__(self, model_name="llama3:8b", base_url="http://ollama-container:11434"):
    def __init__(self, model_name="llama3.1:8b", base_url="http://localhost:11434"):
        """
        A local client for interacting with the Ollama API using a model like LLaMA 3.1.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7, top_p: float = 0.9, system: str = None, stream: bool = False) -> str:
        """
        Generate a response from the Ollama API given a prompt.

        Args:
            prompt (str): The user prompt to send.
            max_length (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p nucleus sampling value.
            system (str, optional): Optional system prompt.
            stream (bool, optional): Whether to stream the output.

        Returns:
            str: Generated response text.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_length
            }
        }

        if system:
            payload["system"] = system

        try:
            response = self.session.post(url, headers=self.headers, json=payload)
            response.raise_for_status()

            if stream:
                # Return generator for streaming (not ideal for sync use)
                return (line for line in response.iter_lines(decode_unicode=True) if line)
            else:
                return response.json().get("response", "")

        except requests.exceptions.RequestException as e:
            print(f"[OllamaClient] Error: {e}")
            return ""

    def available_models(self):
        """List all models available in Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"[OllamaClient] Error fetching models: {e}")
            return []

    def pull_model(self, model_name: str):
        """Pull a new model from Ollama registry."""
        try:
            response = self.session.post(f"{self.base_url}/api/pull", json={"name": model_name})
            response.raise_for_status()
            print(f"[OllamaClient] Model '{model_name}' pulled successfully.")
        except requests.exceptions.RequestException as e:
            print(f"[OllamaClient] Error pulling model: {e}")

    def switch_model(self, model_name: str):
        """Change the active model."""
        self.model_name = model_name
