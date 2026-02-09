# src/pipelines/decomposition.py
import json
import time
import os
import re
import warnings
import pandas as pd
from tqdm import tqdm

from bs4 import GuessedAtParserWarning
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)

from src.core.llm import LocalLLM
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.core.retriever import WebRetriever
from src.data.loader import load_fever_data

class DecompositionPipeline:
    """
    Implements 'Divide and Conquer' with GLOBAL CONTEXT ANCHORING.
    Strategy: 
    1. Search the full claim first to establish context (The "Anchor").
    2. Decompose into atoms.
    3. Verify atoms using Anchor + New Search.
    """

    def __init__(self, model_name="llama3.1"):
        self.llm = LocalLLM(model_name=model_name, temperature=0.0)
        self.retriever = WebRetriever(max_results=3)

    def decompose_claim(self, claim):
        """
        Step 1: Break claim into atomic facts.
        """
        prompt = f"""Task: Split the CLAIM into verifiable atomic facts.
        Rules:
        1. Output strictly a JSON list of strings.
        2. MAX 3 items.
        3. Keep the full entity names (e.g. instead of "He", say "Brad Pitt").

        ### EXAMPLES ###
        Input: "The film Titanic, directed by Cameron, came out in 1997"
        Output: ["Titanic was directed by James Cameron", "Titanic was released in 1997"]

        Input: "Neal Schon was named in 1954"
        Output: ["Neal Schon was named in 1954"]

        ### YOUR TASK ###
        Input: "{claim}"
        Output:"""
    
        try:
            response = self.llm.generate(prompt)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\[.*\]', clean_json, re.DOTALL)
            if match:
                clean_json = match.group(0)
                sub_claims = json.loads(clean_json)
                
                if "Titanic" in str(sub_claims) and "Titanic" not in claim:
                    return [claim]
                    
                if isinstance(sub_claims, list):
                    return sub_claims[:3]
            return [claim]
        except Exception:
            return [claim]
        
    def generate_search_query(self, sub_claim):
        """
        Step 2: Generate simple keyword query (Text-based, no JSON).
        """
        prompt = f"""Task: Convert Fact to Search Query.
        Rules: Keywords only. No conversational filler.

        Fact: Stranger Things is set in Bloomington.
        Query: Stranger Things filming location setting

        Fact: Nikolaj Coster-Waldau worked with Fox.
        Query: Nikolaj Coster-Waldau Fox Broadcasting projects

        Fact: {sub_claim}
        Query:"""

        try:
            response = self.llm.generate(prompt).strip()
            if ":" in response:
                response = response.split(":")[-1].strip()
            query = response.split("\n")[0].replace('"', '').replace("'", "").strip()
            return query
        except:
            return sub_claim

    def verify_sub_claim(self, sub_claim, global_context, previous_memory):
        """
        Step 3: Verify using GLOBAL Context + SPECIFIC Search.
        """
        # A. Specific Search
        query = self.generate_search_query(sub_claim)
        local_evidence = self.retriever.search(query)
        
        # B. Combine Evidence
        combined_evidence = f"""
        --- GLOBAL CONTEXT (From original claim) ---
        {global_context}

        --- PREVIOUS STEPS ---
        {previous_memory}

        --- NEW EVIDENCE (Query: {query}) ---
        {local_evidence}
        """

        # C. Verify
        prompt = f"""You are an expert fact-checker.
        GOAL: Verify the claim using the EVIDENCE.

        RULES:
        1. Focus on the 'NEW EVIDENCE' but use 'GLOBAL CONTEXT' if needed.
        2. If the claim implies a specific entity type (e.g. "The film X") and evidence shows it's a "Song", output REFUTES.
        3. If dates/numbers mismatch, output REFUTES.
        4. If unknown, output NOT ENOUGH INFO.

        EVIDENCE:
        {combined_evidence}

        CLAIM: "{sub_claim}"

        Label (SUPPORTS / REFUTES / NOT ENOUGH INFO):"""

        response = self.llm.generate(prompt)
        return normalize_answer(response), local_evidence, query

    def run(self, limit=10):
        df = load_fever_data(limit=limit)
        if df is None: return None

        print(f"\n Starting decomposition pipeline on {len(df)} samples...")

        predictions = []
        latencies = []
        full_logs = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Decomposing"):
            claim = row['claim']
            start_time = time.time()

            global_context = self.retriever.search(claim)
            sub_claims = self.decompose_claim(claim)
            
            sub_results = []
            votes = []
            accumulated_memory = ""

            for sub in sub_claims:
                label, ev, q = self.verify_sub_claim(sub, global_context, accumulated_memory)
                accumulated_memory += f"- Checked: {sub} -> {label}\n"     
                sub_results.append({"sub": sub, "label": label, "query": q})
                votes.append(label)

            if "REFUTES" in votes:
                final_label = "REFUTES"
            elif "SUPPORTS" in votes:
                final_label = "SUPPORTS"
            else:
                final_label = "NOT ENOUGH INFO"

            end_time = time.time()
            latencies.append(end_time - start_time)
            predictions.append(final_label)
            full_logs.append(sub_results)
            
            time.sleep(1.0)

        # Save
        df['predicted_label'] = predictions
        df['latency_seconds'] = latencies
        df['decomposition_traces'] = [json.dumps(x) for x in full_logs]

        accuracy = evaluate_accuracy(df["predicted_label"].tolist(), df['label'].tolist())
        avg_latency = sum(latencies) / len(latencies)

        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/decomposition_{timestamp}.csv"
        df.to_csv(log_file, index=False)
        
        print("\n" + "="*40)
        print(f" RESULTS: Decomposition ({self.llm.model_name})")
        print("="*40)
        print(f" Accuracy:      {accuracy:.2f}%")
        print(f" Avg Latency:   {avg_latency:.2f}s")
        print(f" Log saved to:  {log_file}")
        print("="*40)
        
        return {"accuracy": accuracy, "latency": avg_latency}

if __name__ == "__main__":
    pipeline = DecompositionPipeline(model_name="llama3.1")
    pipeline.run(limit=50)