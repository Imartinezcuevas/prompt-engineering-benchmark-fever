import pandas as pd
import time
import os
from tqdm import tqdm
from collections import Counter

from src.core.llm import LocalLLM
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.data.loader import load_fever_data

class SelfConsistencyPipeline:
    """
    Implements self-consistency
    """
    def __init__(self, model_name="llama3.1", n_samples=3):
        """
        Args:
            n_samples (int): How many times to query the model.
        """
        self.n_samples = n_samples
        self.model_name = model_name
        self.llm = LocalLLM(model_name=model_name, temperature=0.7)

    def construct_prompt(self, claim):
        """
        Same Cot prompt
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
    
    def get_majority_vote(self, answers):
        """
        Selects the most frequent label from a list of answers.
        """
        clean_labels = [normalize_answer(ans) for ans in answers]

        counter = Counter(clean_labels)

        winner, count = counter.most_common(1)[0]
        return winner, clean_labels
    
    def run(self, limit=10):
        """
        Executes Self-Consistency pipeline
        """
        df = load_fever_data(limit=limit)
        if df is None: return None

        print(f"\n Starting Self-Consistency (k = {self.n_samples}) Benchmark on {len(df)} samples...")

        final_predictions = []
        latencies = []
        all_votes_log = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Voting"):
            claim = row['claim']
            prompt = self.construct_prompt(claim)

            sample_answers = []
            start_time = time.time()

            for i in range(self.n_samples):
                response = self.llm.generate(prompt)
                sample_answers.append(response)

            end_time = time.time()
            latencies.append(end_time - start_time)

            winner, votes = self.get_majority_vote(sample_answers)
            final_predictions.append(winner)
            all_votes_log.append(votes)

        df['predicted_label'] = final_predictions
        df['votes'] = all_votes_log
        df['latency_seconds'] = latencies

        accuracy = evaluate_accuracy(df['predicted_label'].tolist(), df['label'].tolist())
        avg_latency = sum(latencies) / len(latencies)

        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/self_consistency_{timestamp}.csv"
        df.to_csv(log_file, index=False)
        
        print("\n" + "="*40)
        print(f" RESULTS: Self-Consistency (k={self.n_samples})")
        print("="*40)
        print(f" Accuracy:      {accuracy:.2f}%")
        print(f" Avg Latency:   {avg_latency:.2f}s (Total for {self.n_samples} calls)")
        print(f" Log saved to:  {log_file}")
        print("="*40)
        
        return {"accuracy": accuracy, "latency": avg_latency}
    
if __name__ == "__main__":
    pipeline = SelfConsistencyPipeline(model_name="llama3.1", n_samples=5)
    pipeline.run(limit=50)