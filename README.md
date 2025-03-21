# Turing Game Experiment

This project implements an automated evaluation framework for a Turing Game experiment. It processes chat logs where two humans and one AI bot interact, anonymizes the data, and asks various LLM models to detect which participant is the AI bot.

## Project Structure

- `turing_game_experiment.py`: Core framework for parsing log files, running experiments, and processing multiple files
- `analyze_results.py`: Tool to visualize and analyze experiment results from multiple files and models

## Experiment Framework (`turing_game_experiment.py`)

The experiment script is the core component that:

1. **Parses chat logs** from Markdown files containing Turing Game conversations
2. **Anonymizes users** by converting colored identifiers (Red, Blue, etc.) to generic labels (user1, user2, user3)
3. **Queries LLM models** to predict which user is the AI bot
4. **Evaluates predictions** against the ground truth
5. **Processes directories of files** rather than individual files
6. **Saves results** to a structured JSON file for later analysis

## Setup

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt 
```

3. Set up your API keys as environment variables. Create a .env file in the project directory with the following content:

```bash
ANTHROPIC_API_KEY=
```

### Usage

Then you may use the following command to start the experiment:

```bash
python3 turing_game_experiment.py --input-dir inputs --output claude-sonnet-results.json --model claude-3-5-sonnet-20240620
```

Arguments:

- `--input-dir`: Directory containing log files to process
- `--output`: Output JSON file for results (default: results.json)
- `--model`: LLM model to use (e.g., gpt-4, gemini-1.5-pro)
- `--max-retries`: Maximum number of retries for API calls (default: 3)

## Analysis Tool (`analyze_results.py`)

The analysis script processes the JSON results from the experiment to generate visualizations and reports:

1. **Loads and aggregates results** from multiple files and models
2. **Visualizes overall accuracy** of each model
3. **Compares accuracy by file** for each model
4. **Analyzes game difficulty** to identify which games were hardest to detect
5. **Generates a comprehensive report** with insights and statistics

### Usage

```bash
python analyze_results.py --results path/to/results.json --output-dir analysis_output
```

Arguments:
- `--results`: Results JSON file to analyze
- `--output-dir`: Directory where analysis artifacts will be saved (default: analysis)

### Analysis Outputs

The analysis tool generates several files:
- `overall_accuracy.png`: Bar chart comparing model accuracies
- `file_accuracies.png`: Comparison of accuracy by input file for each model
- `game_difficulty.png`: Analysis of which games were easiest/hardest to detect
- `experiment_report.md`: Comprehensive text report with statistics and insights
