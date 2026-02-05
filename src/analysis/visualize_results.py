# src/analysis/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Style configuration
sns.set_theme(style="whitegrid")
RESULTS_DIR = "results/logs"
OUTPUT_DIR = "results/reports"

def get_latest_log(strategy_prefix):
    """
    Finds the most recent CSV log file for a given strategy.
    
    Args:
        strategy_prefix (str): The prefix of the filename (e.g., 'zero_shot').
        
    Returns:
        str: Path to the latest file, or None if not found.
    """
    pattern = os.path.join(RESULTS_DIR, f"{strategy_prefix}_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time (newest last)
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def calculate_metrics(file_path, strategy_name):
    """
    Parses a log file and extracts Accuracy and Average Latency.
    """
    try:
        df = pd.read_csv(file_path)
        
        # --- DEBUG: Verify required columns ---
        if 'predicted_label' not in df.columns:
            print(f"  [Skipping] File {os.path.basename(file_path)} is missing column 'predicted_label'.")
            print(f"   Available columns: {list(df.columns)}")
            return None
            
        # Calculate Accuracy
        correct = (df['predicted_label'] == df['label']).sum()
        total = len(df)
        accuracy = (correct / total) * 100
        
        # Calculate Latency
        # If the pipeline saved 'latency_seconds', use the mean.
        # Otherwise, default to 0.0 (fallback).
        avg_latency = 0.0
        if 'latency_seconds' in df.columns:
             avg_latency = df['latency_seconds'].mean()
        
        return {
            "Strategy": strategy_name,
            "Accuracy": accuracy,
            "Latency": avg_latency,
            "File": os.path.basename(file_path)
        }
    except Exception as e:
        print(f" Error reading {file_path}: {e}")
        return None

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    strategies = [
        ("zero_shot", "Zero-Shot"),
        ("cot", "Chain-of-Thought"),
        ("self_consistency", "Self-Consistency")
    ]
    
    data = []
    
    print(f" Scanning logs in: {os.path.abspath(RESULTS_DIR)}\n")
    
    for prefix, name in strategies:
        file_path = get_latest_log(prefix)
        if file_path:
            metrics = calculate_metrics(file_path, name)
            if metrics:
                data.append(metrics)
                print(f" Loaded: {name:<20} | File: {metrics['File']}")
        else:
            print(f"  No log found for: {name}")

    if not data:
        print("\n No valid data found to visualize.")
        return

    df_results = pd.DataFrame(data)

    # --- CHART 1: Accuracy Comparison ---
    plt.figure(figsize=(10, 6))
    
    # Note: hue="Strategy" and legend=False prevents Seaborn FutureWarning
    barplot = sns.barplot(
        data=df_results, 
        x="Strategy", 
        y="Accuracy", 
        hue="Strategy", 
        palette="viridis", 
        legend=False
    )
    
    plt.title("Accuracy Comparison: Zero-Shot vs Reasoning", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels on top of bars
    for p in barplot.patches:
        height = p.get_height()
        if height > 0: 
            barplot.annotate(f'{height:.1f}%', 
                             (p.get_x() + p.get_width() / 2., height), 
                             ha='center', va='center', 
                             xytext=(0, 9), 
                             textcoords='offset points',
                             fontsize=12, fontweight='bold')
    
    save_path_acc = f"{OUTPUT_DIR}/comparison_accuracy.png"
    plt.savefig(save_path_acc)
    print(f"\n Accuracy Chart saved: {save_path_acc}")

    # --- CHART 2: Trade-off Analysis (Scatter Plot) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_results, 
        x="Latency", 
        y="Accuracy", 
        hue="Strategy", 
        s=200, 
        style="Strategy"
    )
    
    # Draw a line connecting the points to visualize the trend
    plt.plot(df_results["Latency"], df_results["Accuracy"], linestyle='--', color='gray', alpha=0.5)
    
    plt.title("Trade-off: Computational Cost vs Accuracy", fontsize=16)
    plt.xlabel("Average Latency (seconds)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True)
    
    save_path_trade = f"{OUTPUT_DIR}/tradeoff_analysis.png"
    plt.savefig(save_path_trade)
    print(f" Trade-off Chart saved: {save_path_trade}")

if __name__ == "__main__":
    main()