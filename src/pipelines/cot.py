import pandas
import time
import os
from tqdm import tqdm

from src.core.llm import LocalLLM
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.data.loader import load_fever_data

class CoTPipeline:
    """
    Implements Chain-of-Though prompting strategy.

    Force the model to generate intermediate reasoning steps before answering.
    """

    def __init__(self, model_name="llama3.1"):
        self.llm = LocalLLM(model_name=model_name, temperature=0.0)

    def construct_prompt(self, claim):
        """
        Construct a prompt that triggers reasoning.
        """

        return f"""You are an expert fact-checker.
        Task: Verify the veracity of the following claim.
        Claim: "{claim}"

        Instructions:
        1. Analize the claim concepts.
        2. Think step-by-step about the facts you know related to these concepts.
        3. Check for contradictions or support.
        4. Finally, output the classification label.

        Valid labels:
        - SUPPORTS
        - REFUTES
        - NOT ENOUGH INFO

        Format your response exactly like this:
        Reasoning: [Your step-by-step logic here]
        Label: [One of the valid labels]

        Let's think step by step:"""
    
    def run(self, limit=10):
        """
        Executes CoT pipeline
        """
        df = load_fever_data(limit=limit)
        if df is None: return None
        
        print("\n Starting CoT benchmark on {len(df)} samples...")

        predictions = []
        latencies = []
        raw_outputs = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Reasoning"):
            claim = row['claim']
            prompt = self.construct_prompt(claim)

            start_time = time.time()
            response = self.llm.generate(prompt)
            end_time = time.time()

            latencies.append(end_time - start_time)
            raw_outputs.append(response)

            predictions.append(normalize_answer(response))

        df['cot_reasoning'] = raw_outputs
        df['predicted_label'] = predictions

        accuracy = evaluate_accuracy(df['predicted_label'].to_list(), df['label'].to_list())
        avg_latency = sum(latencies) / len(latencies)

        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/cot_{timestamp}.csv"
        df.to_csv(log_file, index=False)

        print("\n" + "="*40)
        print(f" RESULTS: Chain-of-Thought ({self.llm.model_name})")
        print("="*40)
        print(f" Accuracy:      {accuracy:.2f}%")
        print(f" Avg Latency:   {avg_latency:.2f}s")
        print(f" Log saved to:  {log_file}")
        print("="*40)
        
        return {"accuracy": accuracy, "latency": avg_latency}
    
if __name__ == "__main__":
    pipeline = CoTPipeline(model_name="llama3.1")
    pipeline.run(limit=50)