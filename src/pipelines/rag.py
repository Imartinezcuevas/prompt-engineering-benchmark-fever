import pandas as pd
import time
from tqdm import tqdm
import os

from src.core.llm import LocalLLM
from src.core.retriever import WebRetriever
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.data.loader import load_fever_data

class RAGPipeline:
    """
    Implements Retrieval-Augmented Generation (RAG).

    Workflow:
    1. Retrieve: Search the wev for evidence related to the claim.
    2. Augment: Inject that evidence into the prompt.
    3. Generate: Ask the LLM to verify the claim based on that evidence.
    """

    def __init__(self, model_name="llama3.1"):
        self.llm = LocalLLM(model_name=model_name, temperature=0.0)
        self.retriever = WebRetriever(max_results=1)

    def construct_prompt(self, claim, evidence):
        """
        Builds a prompt that grounds the model in the retrieved evidence.
        """
        return f"""You are an expert fact-checker.
        GOAL: Verify the claim using the provided EVIDENCE. 

        RULES:
        1. PRIORITY: If the EVIDENCE confirms or contradicts the claim, base your answer on it.
        2. FALLBACK: If the EVIDENCE is irrelevant or incomplete, verify the claim using your INTERNAL KNOWLEDGE.
        3. SYNONYMS: Use common sense (e.g., "Youtuber" = "Content Creator").

        RETRIEVED EVIDENCE:
        '''
        {evidence}
        '''

        CLAIM:
        "{claim}"

        Based on the evidence (or your knowledge if evidence is missing), what is the verdict?
        Format:
        Reasoning: [Explain your logic briefly]
        Label: [SUPPORTS, REFUTES, or NOT ENOUGH INFO]"""
    
    def run(self, limit=10):
        df = load_fever_data(limit=limit)
        if df is None: return None

        print(f"\n Starting RAG pipeline on {len(df)} samples...")

        predictions = []
        latencies = []
        retrieved_contexts = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
            claim = row['claim']

            start_time = time.time()
            evidence_text = self.retriever.search(claim)
            prompt = self.construct_prompt(claim, evidence_text)
            response = self.llm.generate(prompt)
            end_time = time.time()

            latencies.append(end_time - start_time)
            retrieved_contexts.append(evidence_text)
            predictions.append(normalize_answer(response))

            # Delay to avoid getting banned by DuckDuckGo
            time.sleep(5.0)

        df['rag_evidence'] = retrieved_contexts
        df['predicted_label'] = predictions
        df['latency_seconds'] = latencies

        accuracy = evaluate_accuracy(df['predicted_label'].tolist(), df['label'].tolist())
        avg_latency = sum(latencies) / len(latencies)
        
        # Export
        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/rag_{timestamp}.csv"
        df.to_csv(log_file, index=False)
        
        print("\n" + "="*40)
        print(f" RESULTS: RAG + Web Search ({self.llm.model_name})")
        print("="*40)
        print(f" Accuracy:      {accuracy:.2f}%")
        print(f" Avg Latency:   {avg_latency:.2f}s")
        print(f" Log saved to:  {log_file}")
        print("="*40)
        
        return {"accuracy": accuracy, "latency": avg_latency}
    
if __name__ == "__main__":
    pipeline = RAGPipeline(model_name="llama3.1")
    pipeline.run(limit=50)