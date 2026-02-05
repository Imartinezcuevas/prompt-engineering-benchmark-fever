import requests
import json
import time

class LocalLLM:
    """
    Wrapper class to interact with local Ollama instace via API.
    Designed to be modular.
    """

    def __init__(self, model_name="llama3.1", temperature=0.0, base_url="http://localhost:11434"):
        """
        Args:
            model_name (str): The model tag in Ollama
            temperature (float): 0.0 for deterministic logic, 0.7 for crative writing.
            base_url (str): The local ollama server URL.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = f"{base_url}/api/generate"
        self.headers = {"Content-Type": "application/json"}

    def generate(self, prompt):
        """
        Sends a prompt to the model and returns the text response.

        Args:
            prompt (str): The input text for the LLM.

        Returns:
            str: The generated text response.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096
            }
        }

        try:
            # We use a timeout to prevent hanging forever if Ollama is down
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=120
            )

            response.raise_for_status()

            #¨ Parse the response
            result = response.json()
            return result.get("response", "").strip()
        
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama."
        except Exception as e:
            return f"Error: {str(e)}"
        
if __name__ == "__main__":
    print(f"Connecting to local Ollama...")

    llm = LocalLLM(model_name="llama3.1", temperature=0.7)

    start_time = time.time()
    reply = llm.generate("Explain in one sentence what Prompt Engineering is.")
    end_time = time.time()

    print(f"\nModel response ({end_time - start_time:.2f}s):")
    print(f"'{reply}'")

