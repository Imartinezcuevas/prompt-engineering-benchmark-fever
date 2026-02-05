import re

def normalize_answer(text):
    """
    Cleans and extracts the classification label from the LLM's raw response.

    The FEVER dataset uses three specific labels:
    1. SUPPORTS
    2. REFUTES
    3. NOT ENOUGH INFO

    This functions handles 'chatty' responses (e.g., "The answer is SUPPORTS")
    and maps common synonyms (TRUE -> SUPPORTS).

    Args:
        text (str): The raw text output from the LLM.
    
    Returns:
        str: The normalized label or 'UNKNOWN'
    """
    if not isinstance(text, str):
        return "UNKNOWN"
    
    # 1. Basic cleaning: Uppercase and remove surrounding whitespace
    text = text.upper().strip()

    # 2. Remove punctuation that might interfere
    text = text.replace(".", "").replace("*", "").replace('!', "")

    # 3. Priority matching
    if "NOT ENOUGH INFO" in text or "NEI" in text:
        return "NOT ENOUGH INFO"
    
    if "SUPPORTS" in text:
        return "SUPPORTS"
    
    if "REFUTES" in text:
        return "REFUTES"
    
    # 4. Heuristic matching
    if "FALSE" in text or "INCORRECT" in text or "FAKE" in text:
        return "REFUTES"
    
    if "TRUE" in text or "CORRECT" in text or "VALID" in text:
        return "SUPPORTS"
    
    return "UNKNOWN"

def evaluate_accuracy(predictions, references):
    """
    Computes the accuracy metric for the benchmark.

    Args:
        predictions (list): List of raw output strigns from the LLM.
        references (list): List of ground truth from the dataset.

    Returns:
        float: The accuracy percentage (0.0 to 100.0)
    """
    if not predictions or not references:
        print(f" Warning: Empty prediction or reference list.")
        return 0.0
    
    # Safety check
    if len(predictions) != len(references):
        print(f"Warning: length missmatch (Preds: {len(predictions)}, Refs: {len(references)}). Truncating to minimum.")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]

    correct_count = 0
    total_count = len(references)

    for pred, ref in zip(predictions, references):
        # Normalize
        pred_label = normalize_answer(pred)

        # Ensure reference is also clean
        ref_label = str(ref).upper().strip()

        # Exact string match check
        if pred_label == ref_label:
            correct_count += 1
        
    accuracy = (correct_count / total_count) * 100.0
    return accuracy

if __name__ == "__main__":
    print(f"Running evaluator unit tests...")

    # Test data
    test_preds = [
        "Based on the evidence, the label is SUPPORTS.",
        "REFUTES",
        "I think is is false.",
        "Not enough info provided.",
        "Cheese.",
        "TRUE"
    ]

    test_refs = [
        "SUPPORTS",
        "REFUTES",
        "REFUTES",
        "NOT ENOUGH INFO",
        "SUPPORTS",
        "SUPPORTS"
    ]

    score = evaluate_accuracy(test_preds, test_refs)
    print(f"Test accuracy: {score:2f}%")

    if 83.0 < score < 84.0:
        print("Evaluator logic is working correctly.")
    else:
        print("Evaluator logic failed.")
