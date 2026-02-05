import pandas as pd
import time
import os
from tqdm import tqdm

from src.core.llm import LocalLLM
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.data.loader import load_fever_data

class ZeroShotPipeline:
    """
    Implements the Zero-Shot prompting strategy.
    The model is asked to classify the claim directly without examples or external retrieval.
    """

    def __init__(self, model_name="llama3.1"):
        """
        Args:
            model_name (str): The Ollama model to use.
        """
        # We use temperature=0.0 for deterministic output
        self.llm = LocalLLM(model_name=model_name, temperature=0.0)

    def construct_prompt(self, claim):
        """
        Construct the Zero-Shot prompt.

        Engineering Note:
        We usa a 'Persona' ("You are a fact checker") to ground the model.
        We explicitly list the allowed labels to constrain the output space.
        """
        return f"""You are an automated fact-checking system.
        Your task is to verify the veracity of the following claim using your internal knowledge.

        Classes:
        - SUPPORTS: The claim is True.
        - REFUTES: The claim is False.
        - NOT ENOUGH INFO: You cannot verify it.

        Claim: "{claim}"

        Respond only with the class label. Don't explain.
        Label:"""
    
    def run(self, limit=10):
        """
        Executes the pipeline on the dataset.

        Args:
            limit (int): number of sampes to process.

        Returns:
            dict: A dictionary containing metrics and the result dataframe.
        """
        # 1. Load data
        df = load_fever_data(limit=limit)
        if df is None:
            return None
        
        print(f"\n Starting Zero-Shot benchmark on {len(df)} samples...")

        predictions = []
        latencies = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            claim = row["claim"]
            prompt = self.construct_prompt(claim)

            start_time = time.time()
            raw_response = self.llm.generate(prompt)
            end_time = time.time()

            latencies.append(end_time - start_time)
            predictions.append(raw_response)

        df['raw_response'] = predictions
        df['predicted_labels'] = [normalize_answer(p) for p in predictions]

        accuracy = evaluate_accuracy(df['predicted_labels'].tolist(), df['label'].tolist())
        avg_latency = sum(latencies) / len(latencies)

        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/zero_shot_{timestamp}.csv"
        df.to_csv(log_file, index=False)

        print("\n" + "="*40)
        print(f"RESULTS: Zero-Shot ({self.llm.model_name})")
        print("="*40)
        print(f" Accuracy:      {accuracy:.2f}%")
        print(f" Avg Latency:   {avg_latency:.2f}s")
        print(f" Log saved to:  {log_file}")
        print("="*40)
        
        return {"accuracy": accuracy, "latency": avg_latency}
    
if __name__ == "__main__":
    pipeline = ZeroShotPipeline(model_name="llama3.1")
    pipeline.run(limit=50)