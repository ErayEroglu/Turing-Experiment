import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def load_results(results_file: str) -> Dict[str, Any]:
    """Load results from a JSON file with the new nested structure."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data has the expected structure
        if "results" in data:
            return data
        else:
            print(f"Warning: Results file doesn't have the expected structure with 'results' key")
            return data
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return {}

def extract_model_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract aggregated results per model from the nested structure."""
    model_results = {}
    
    # Handle the nested structure with "results" key
    if "results" in results:
        file_results_dict = results["results"]
    else:
        # Fallback to the old structure for backward compatibility
        file_results_dict = results
    
    # First, identify all unique models
    all_models = set()
    for file_path, file_results in file_results_dict.items():
        model = file_results.get('model')
        if model:
            all_models.add(model)
    
    # If no models found in file results, check the summary
    if not all_models and "summary" in results and "model" in results["summary"]:
        all_models.add(results["summary"]["model"])
    
    # Initialize data structures for each model
    for model in all_models:
        model_results[model] = {
            'total_games': 0,
            'correct_predictions': 0,
            'accuracy': 0,
            'file_accuracies': {},
            'game_results': []
        }
    
    # Populate with data from each file
    for file_path, file_results in file_results_dict.items():
        model = file_results.get('model')
        if not model:
            continue
        
        file_name = os.path.basename(file_path)
        
        model_results[model]['total_games'] += file_results.get('total_games', 0)
        model_results[model]['correct_predictions'] += file_results.get('correct_predictions', 0)
        
        file_accuracy = file_results.get('accuracy', 0)
        model_results[model]['file_accuracies'][file_name] = file_accuracy
        
        model_results[model]['game_results'].extend(file_results.get('game_results', []))
    
    # Calculate accuracy for each model
    for model, data in model_results.items():
        if data['total_games'] > 0:
            data['accuracy'] = data['correct_predictions'] / data['total_games']
    
    return model_results

def plot_overall_accuracy(model_results: Dict[str, Any], output_file: str = None):
    """Plot overall accuracy for each model."""
    models = list(model_results.keys())
    accuracies = [data['accuracy'] * 100 for data in model_results.values()]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color='skyblue')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{acc:.1f}%",
                 ha='center', va='bottom')

    plt.axhline(y=33.33, color='r', linestyle='--', label='Random Guess (33.33%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Bot Detection Accuracy by Model')
    plt.ylim(0, max(100, max(accuracies) + 10))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()

    if output_file:
        plt.savefig(output_file)
        print(f"Saved overall accuracy plot to {output_file}")
    else:
        plt.show()

def plot_file_accuracies(model_results: Dict[str, Any], output_file: str = None):
    """Plot accuracy by file for each model."""
    df_data = {}
    for model, data in model_results.items():
        for file_name, accuracy in data['file_accuracies'].items():
            if file_name not in df_data:
                df_data[file_name] = {}
            df_data[file_name][model] = accuracy * 100

    df = pd.DataFrame(df_data)

    plt.figure(figsize=(12, 7))
    df.plot(kind='bar', ax=plt.gca())

    plt.axhline(y=33.33, color='r', linestyle='--', label='Random Guess (33.33%)')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model')
    plt.title('Bot Detection Accuracy by Model and File')
    plt.legend(title='File')
    plt.ylim(0, 100)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Saved file accuracy plot to {output_file}")
    else:
        plt.show()

def analyze_game_difficulty(model_results: Dict[str, Any], output_file: str = None):
    """Analyze which games were most difficult across models."""
    all_games = {}
    for model, data in model_results.items():
        for game_result in data['game_results']:
            game_id = game_result.get('game_id')
            if not game_id:
                continue
                
            correct_bot_user = game_result.get('correct_bot_user')
            if not correct_bot_user:
                continue

            if game_id not in all_games:
                all_games[game_id] = {
                    'predictions': {},
                    'correct_bot_user': correct_bot_user,
                    'success_count': 0,
                    'total_predictions': 0
                }

            # Use 'is_correct' instead of 'correct_bot'
            all_games[game_id]['predictions'][model] = game_result.get('is_correct', False)
            all_games[game_id]['total_predictions'] += 1
            if game_result.get('is_correct', False):
                all_games[game_id]['success_count'] += 1

    for game_id, game_data in all_games.items():
        game_data['success_rate'] = (game_data['success_count'] / game_data['total_predictions'] * 100) if game_data['total_predictions'] > 0 else 0

    # If no games were processed, return empty results
    if not all_games:
        print("No game data available for analysis")
        return {}

    df = pd.DataFrame([{
        'game_id': game_id,
        'success_rate': data['success_rate'],
        'correct_bot_user': data['correct_bot_user']
    } for game_id, data in all_games.items()])

    if df.empty:
        print("No game data available for analysis")
        return {}

    df_sorted = df.sort_values('success_rate')

    plt.figure(figsize=(14, 7))
    bars = plt.bar(df_sorted['game_id'].astype(str), df_sorted['success_rate'], color='lightgreen')

    plt.axhline(y=50, color='orange', linestyle='--', label='50% Success Rate')
    plt.axhline(y=33.33, color='r', linestyle='--', label='33.33% Success Rate')
    plt.text(len(df_sorted) - 1, 51, '50% Success Rate', ha='right', va='bottom', color='orange')
    plt.text(len(df_sorted) - 1, 34.33, 'Random Guess (33.33%)', ha='right', va='bottom', color='r')

    plt.ylabel('Success Rate (%)')
    plt.xlabel('Game ID')
    plt.title('Bot Detection Success Rate by Game')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()

    if output_file:
        plt.savefig(output_file)
        print(f"Saved game difficulty plot to {output_file}")
    else:
        plt.show()

    difficult_games = df_sorted[df_sorted['success_rate'] < 33.33]['game_id'].tolist()
    easy_games = df_sorted[df_sorted['success_rate'] > 66.67]['game_id'].tolist()

    return {
        'difficult_games': difficult_games[:5] if len(difficult_games) >= 5 else difficult_games,
        'easy_games': easy_games[-5:] if len(easy_games) >= 5 else easy_games
    }

def generate_report(model_results: Dict[str, Any], game_analysis: Dict, output_file: str = None):
    """Generate a text report summarizing the experiment results."""
    report = ["# Turing Game Experiment Results\n", "## Overall Model Performance\n"]

    for model, data in model_results.items():
        report.append(f"- **{model}**: {data['accuracy']*100:.1f}% ({data['correct_predictions']}/{data['total_games']})")

    report.append("\n## Performance by File\n")
    for model, data in model_results.items():
        report.append(f"\n### {model}\n")
        for file_name, accuracy in data['file_accuracies'].items():
            report.append(f"- **{file_name}**: {accuracy*100:.1f}% accuracy")

    if game_analysis.get('difficult_games'):
        report.append("\n## Most Challenging Games\n")
        report.append("The following games were the most difficult for models to correctly identify the bot:")
        for game_id in game_analysis['difficult_games']:
            report.append(f"- Game {game_id}")

    if game_analysis.get('easy_games'):
        report.append("\n## Easiest Games\n")
        report.append("The following games were the easiest for models to correctly identify the bot:")
        for game_id in game_analysis['easy_games']:
            report.append(f"- Game {game_id}")

    report.append("\n## Conclusion\n")
    avg_accuracies = [data['accuracy'] for data in model_results.values()]
    avg_accuracy = np.mean(avg_accuracies) * 100 if avg_accuracies else 0

    if avg_accuracy > 70:
        report.append("The models performed significantly better than random chance (33.33%) at detecting bots in the Turing Game.")
    elif avg_accuracy > 40:
        report.append("The models performed better than random chance (33.33%) at detecting bots, but with moderate success.")
    else:
        report.append("The models struggled to reliably detect bots beyond random chance (33.33%).")

    report_text = '\n'.join(report)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Saved report to {output_file}")

    return report_text

def main():
    parser = argparse.ArgumentParser(description='Analyze Turing Game Experiment Results')
    parser.add_argument('--results', required=True, help='Results JSON file to analyze')
    parser.add_argument('--output-dir', default='analysis', help='Output directory for analysis files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)

    if not results:
        print("Error: No valid results found in the file.")
        return
        
    if args.verbose:
        print(f"Results structure: {list(results.keys())}")
        if "summary" in results:
            print(f"Summary: {results['summary']}")
        if "results" in results:
            print(f"Found results for {len(results['results'])} files")

    print("Extracting model results...")
    model_results = extract_model_results(results)

    if not model_results:
        print("Error: No model results could be extracted from the data.")
        print("Please check that your results file has the correct structure.")
        return
        
    print(f"Found results for models: {list(model_results.keys())}")
    
    for model, data in model_results.items():
        print(f"Model: {model}")
        print(f"  Total games: {data['total_games']}")
        print(f"  Correct predictions: {data['correct_predictions']}")
        print(f"  Accuracy: {data['accuracy']*100:.1f}%")
        print(f"  Files: {list(data['file_accuracies'].keys())}")

    print("\nGenerating plots and reports...")
    
    # Generate overall accuracy plot
    plot_overall_accuracy(model_results, os.path.join(args.output_dir, 'overall_accuracy.png'))

    # Generate file accuracies plot if there are multiple files
    any_multiple_files = any(len(data['file_accuracies']) > 1 for data in model_results.values())
    if any_multiple_files:
        plot_file_accuracies(model_results, os.path.join(args.output_dir, 'file_accuracies.png'))
    else:
        print("Skipping file accuracies plot as there is only one file per model")

    # Analyze game difficulty
    try:
        game_analysis = analyze_game_difficulty(model_results, os.path.join(args.output_dir, 'game_difficulty.png'))
    except Exception as e:
        print(f"Error analyzing game difficulty: {e}")
        game_analysis = {}

    # Generate report
    try:
        report = generate_report(model_results, game_analysis, os.path.join(args.output_dir, 'experiment_report.md'))
        print("\nReport Summary:")
        print("\n".join(report.split("\n")[:10]) + "\n...")
    except Exception as e:
        print(f"Error generating report: {e}")
        
    print(f"\nAnalysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()