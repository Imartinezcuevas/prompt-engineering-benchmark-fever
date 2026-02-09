import pandas as pd
import time
from tqdm import tqdm
import os
import re

from src.core.llm import LocalLLM
from src.core.tools import SearchTool
from src.core.evaluator import evaluate_accuracy, normalize_answer
from src.data.loader import load_fever_data

class Agent:
    """
    Agent
    """
    
    def __init__(self, model_name="llama3.1", max_iterations=3):
        self.llm = LocalLLM(model_name=model_name, temperature=0.0)
        self.tools = {"Search": SearchTool(max_results=2)}
        self.max_iterations = max_iterations
    
    def construct_react_prompt(self, claim, history="", iteration_num=1):
        """Builds the ReAct prompt with adaptive guidance based on iteration."""
        
        tools_desc = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        # Add stronger guidance after first iteration
        evidence_guidance = ""
        if iteration_num > 1:
            evidence_guidance = """
⚠️ IMPORTANT - INTERPRETING EVIDENCE:
- "Worked with Company X" means appearing in their productions, projects, or collaborations
- An actor appearing in a TV show on Fox = worked with Fox Broadcasting Company
- A musician releasing an album on Sony = worked with Sony Music
- You DON'T need to find evidence of direct employment or formal contracts
- If Wikipedia or reliable sources confirm the connection, you have enough evidence

If you already found relevant evidence in previous searches, use Action: Finish now.
"""
        
        return f"""You are an expert fact-checking agent that verifies claims using systematic reasoning.

AVAILABLE TOOLS:
{tools_desc}

SEARCH QUERY BEST PRACTICES:
✓ GOOD: "Nikolaj Coster-Waldau Fox" (entity + organization)
✓ GOOD: "Cristiano Ronaldo Ballon d'Or" (entity + fact)
✗ BAD: "early career credits" (too vague)
✗ BAD: "employee records" (too specific - actors aren't employees)

REASONING FORMAT:
Thought: [Your reasoning about what information you have and what you need]
Action: [Either "Search" or "Finish"]
Action Input: [If Search: specific query. If Finish: ONLY the verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO)]

DECISION RULES:
1. If you found evidence that CONFIRMS the claim → Action: Finish with SUPPORTS
2. If you found evidence that CONTRADICTS the claim → Action: Finish with REFUTES
3. If after {self.max_iterations} searches you found NO relevant evidence → Action: Finish with NOT ENOUGH INFO
4. Don't keep searching if you already have a clear answer

INTERPRETATION GUIDE:
- "Person X worked with Company Y" is SUPPORTED if they appeared in Y's productions, projects, or media
- "Person X won N awards" is SUPPORTED if reliable sources confirm the exact number
- "Event X happened in Year Y" is SUPPORTED if dates match in reliable sources
{evidence_guidance}

CLAIM TO VERIFY:
"{claim}"

{history}

Iteration {iteration_num}/{self.max_iterations + 1}
Thought:"""

    def parse_action(self, llm_output):
        """Extracts Action and Action Input from LLM response."""
        action_match = re.search(r'Action:\s*(\w+)', llm_output, re.IGNORECASE)
        input_match = re.search(r'Action Input:\s*(.+?)(?:\n|$)', llm_output, re.IGNORECASE | re.DOTALL)
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip().strip('"').strip("'").strip()
            return action, action_input
        
        return None, None
    
    def run_iteration(self, claim):
        """Runs the ReAct loop for a single claim."""
        history = ""
        reasoning_trace = []
        search_count = 0
        
        for i in range(self.max_iterations + 1):
            # Generate prompt with iteration-specific guidance
            prompt = self.construct_react_prompt(claim, history, iteration_num=i+1)
            
            # Get LLM response
            response = self.llm.generate(prompt, stop=["Observation:"])
            reasoning_trace.append(f"Iteration {i+1}:\n{response}")
            
            # Parse the action
            action, action_input = self.parse_action(response)
            
            if action is None:
                # Try to extract verdict if parsing fails
                verdict = normalize_answer(response)
                if verdict in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
                    return verdict, "\n\n".join(reasoning_trace)
                history += f"\n{response}\nObservation: Please provide valid Action (Search or Finish).\n"
                continue
            
            # Check if agent wants to finish
            if action.lower() == "finish":
                final_answer = normalize_answer(action_input)
                return final_answer, "\n\n".join(reasoning_trace)
            
            # Execute Search
            if action.lower() == "search":
                if search_count >= self.max_iterations:
                    observation = f"Observation: Maximum searches ({self.max_iterations}) reached. You must finish now with your best verdict based on evidence found."
                    history += f"\n{response}\n{observation}\n"
                    reasoning_trace.append(observation)
                    continue
                
                search_count += 1
                tool_output = self.tools["Search"].run(action_input)
                observation = f"Observation: {tool_output}"
                
                history += f"\n{response}\n{observation}\n"
                reasoning_trace.append(observation)
            else:
                observation = f"Observation: Unknown action '{action}'. Use 'Search' or 'Finish'."
                history += f"\n{response}\n{observation}\n"
                reasoning_trace.append(observation)
        
        # Force final decision with all context
        final_prompt = f"""You have completed {self.max_iterations} searches for this claim:
"{claim}"

Based on ALL the evidence you gathered, what is your final verdict?

CRITICAL: Review all observations above. If ANY observation provided evidence confirming or contradicting the claim, use that evidence.

Action: Finish
Action Input:"""
        
        final_response = self.llm.generate(final_prompt)
        reasoning_trace.append(f"Forced Final Decision:\n{final_response}")
        
        final_answer = normalize_answer(final_response)
        return final_answer, "\n\n".join(reasoning_trace)


class AgentPipeline:
    """Agent-based pipeline."""
    
    def __init__(self, model_name="llama3.1", max_iterations=3):
        self.agent = Agent(model_name=model_name, max_iterations=max_iterations)
        self.model_name = model_name
    
    def run(self, limit=10, sleep_time=3.0):
        df = load_fever_data(limit=limit)
        if df is None:
            return None
        
        print(f"\n Starting ReAct agent on {len(df)} samples...")
        print(f"   Max iterations: {self.agent.max_iterations}")
        
        predictions = []
        latencies = []
        reasoning_traces = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
            claim = row['claim']
            
            start_time = time.time()
            prediction, trace = self.agent.run_iteration(claim)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
            predictions.append(prediction)
            reasoning_traces.append(trace)
            
            if idx < len(df) - 1:
                time.sleep(sleep_time)
        
        df['predicted_label'] = predictions
        df['reasoning_trace'] = reasoning_traces
        df['latency_seconds'] = latencies
        
        # Calculate metrics
        accuracy = evaluate_accuracy(df['predicted_label'].tolist(), df['label'].tolist())
        avg_latency = sum(latencies) / len(latencies)
        pred_counts = df['predicted_label'].value_counts()
        
        # Calculate per-label accuracy
        label_accuracy = {}
        for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            label_df = df[df['label'] == label]
            if len(label_df) > 0:
                correct = (label_df['predicted_label'] == label_df['label']).sum()
                label_accuracy[label] = (correct / len(label_df)) * 100
        
        # Export results
        os.makedirs("results/logs", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = f"results/logs/agent_fixed_{timestamp}.csv"
        df.to_csv(log_file, index=False)
        
        # Print results
        print("\n" + "="*70)
        print(f" RESULTS: agent ({self.model_name})")
        print("="*70)
        print(f" Overall Accuracy: {accuracy:.2f}%")
        print(f" Avg Latency:      {avg_latency:.2f}s")
        print(f" Max Iterations:   {self.agent.max_iterations}")
        print(f"\n Per-Label Accuracy:")
        for label, acc in label_accuracy.items():
            print(f"   {label:20s}: {acc:5.1f}%")
        print("="*70)
        
        return {
            "accuracy": accuracy,
            "latency": avg_latency,
            "predictions": predictions,
            "traces": reasoning_traces,
            "distribution": pred_counts.to_dict(),
            "per_label_accuracy": label_accuracy
        }


if __name__ == "__main__":
    pipeline = AgentPipeline(model_name="llama3.1", max_iterations=3)
    results = pipeline.run(limit=20, sleep_time=3.0)