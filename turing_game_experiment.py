import re
import json
import os
import time
from typing import Dict, List, Tuple, Optional
import argparse
import random
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class Game:
    id: str
    users: Dict[str, str]
    chat: List[Tuple[str, str]]
    accusations: Dict[str, bool]

def parse_log_file(filename: str) -> List[Game]:
    """Parse the log file and extract games."""
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    game_blocks = re.findall(r'<details>\s*<summary>Game \d+: \(ID: (\d+)\)</summary>(.*?)</details>',
                             content, re.DOTALL)

    games = []
    for game_id, game_content in game_blocks:
        user_table = re.search(r'\| User \| Color \|\s*\| ---- \| ----- \|(.*?)\s*###',
                               game_content, re.DOTALL)

        if not user_table:
            continue  # Skip if user table not found

        users = {}
        emoji_to_color = {}  # Map emoji to color for easier lookup
        user_roles = {}  # Map roles like "You" to color

        for line in user_table.group(1).strip().split('\n'):
            if '**' in line:
                parts = line.split('|')
                if len(parts) < 3:
                    continue

                user_role = parts[1].strip()
                color_match = re.search(r'\*\*(.+?) (.+?)\*\*', parts[2].strip())

                if color_match:
                    emoji, color = color_match.groups()
                    users[color] = 'bot' if 'Bot' in user_role else 'human'
                    emoji_to_color[emoji] = color  # Store mapping from emoji to color

                    # Store mapping from role to color for accusations later
                    if user_role == "You" or user_role == "Other human":
                        user_roles[user_role] = color

        chat_section = re.search(r'### The Chat:(.*?)### The Accusations:',
                                 game_content, re.DOTALL)

        if not chat_section:
            continue  # Skip if chat section not found

        chat = []
        for line in chat_section.group(1).strip().split('\n'):
            if '): **' in line:
                message_match = re.search(r'\((.+?)\): \*\*(.+?)\*\*', line)
                if message_match:
                    emoji, message = message_match.groups()
                    if emoji in emoji_to_color:
                        color = emoji_to_color[emoji]
                        chat.append((color, message))

        # Extract accusations
        accusations_section = re.search(r'### The Accusations:(.*?)$',
                                        game_content, re.DOTALL)

        if not accusations_section:
            continue  # Skip if accusations section not found

        accusations = {}
        for line in accusations_section.group(1).strip().split('\n'):
            if '**' in line:
                parts = line.split('|')
                if len(parts) < 3:
                    continue

                user_role = parts[1].strip()
                result = parts[2].strip()

                # Only process accusations for human users
                if user_role in user_roles and users.get(user_roles[user_role]) == 'human':
                    color = user_roles[user_role]

                    # Determine accusation result
                    if "Correct accusation" in result:
                        accusations[color] = True
                    elif "Incorrect accusation" in result:
                        accusations[color] = False
                    elif "No accusation" in result:
                        accusations[color] = None

        games.append(Game(
            id=game_id,
            users=users,
            chat=chat,
            accusations=accusations
        ))

    return games

def anonymize_game(game: Game) -> Dict:
    """Convert a game to an anonymized format for LLM input."""
    color_to_user = {}
    for i, color in enumerate(game.users.keys(), 1):
        color_to_user[color] = f"user{i}"

    anonymized_chat = []
    for color, message in game.chat:
        anonymized_chat.append((color_to_user[color], message))

    return {
        "game_id": game.id,
        "color_to_user": color_to_user,
        "user_to_color": {v: k for k, v in color_to_user.items()},
        "anonymized_chat": anonymized_chat,
        "original_users": game.users,
        "correct_bot_color": next(color for color, role in game.users.items() if role == 'bot'),
    }

def format_prompt(anonymized_game: Dict) -> str:
    """Format the anonymized game as a prompt for the LLM."""
    prompt = f"You are participating in a Turing test experiment. Below is a chat log with three participants: user1, user2, and user3.\n\n"
    prompt += f"One of these three users is an AI bot, while the other two are humans.\n\n"
    prompt += f"Chat log:\n"

    for user, message in anonymized_game["anonymized_chat"]:
        prompt += f"{user}: {message}\n"

    prompt += f"\nBased on the chat log above, which user do you think is the AI bot? Please respond with only 'user1', 'user2', or 'user3'."

    return prompt

def query_llm(prompt: str, api_key: str, model: str = "gpt-4") -> str:
    """Send a prompt to the LLM and get a response."""
    load_dotenv()
    if model.startswith("gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        return query_gemini(prompt, api_key, model)
    elif model.startswith("gpt"):
        # OpenAI implementation
        # For OpenAI example:
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0
        # )
        # return response.choices[0].message.content.strip()
        pass

    # For demonstration or fallback, return a random prediction
    return f"user{random.randint(1, 3)}"

def query_gemini(prompt: str, api_key: str,  model: str = "gemini-1.5-pro") -> str:
    """Send a prompt to Google's Gemini model and get a response."""
    try:
        import google.generativeai as genai

        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini")

        genai.configure(api_key=api_key)

        model_instance = genai.GenerativeModel(model)

        response = model_instance.generate_content(prompt)

        prediction = response.text.strip().lower()

        if "user1" in prediction:
            return "user1"
        elif "user2" in prediction:
            return "user2"
        elif "user3" in prediction:
            return "user3"
        else:
            print(f"Warning: Unexpected response format from Gemini: {prediction}")
            return prediction

    except ImportError:
        print("Error: google-generativeai package not installed.")
        print("Install it with: pip install google-generativeai")
        raise
    except Exception as e:
        print(f"Error querying Gemini: {e}")
        raise

def evaluate_prediction(anonymized_game: Dict, prediction: str) -> bool:
    correct_bot_color = anonymized_game["correct_bot_color"]
    predicted_color = anonymized_game["user_to_color"].get(prediction)
    return predicted_color == correct_bot_color

def run_experiment(log_file: str, model: str = "gpt-4", max_retries: int = 3) -> dict:
    """Run the full experiment and save results."""
    print(f"Parsing log file: {log_file}")
    games = parse_log_file(log_file)
    print(f"Found {len(games)} games")

    results = []
    correct_predictions = 0

    for i, game in enumerate(games):
        print(f"Processing game {i+1}/{len(games)} (ID: {game.id})")
        anonymized_game = anonymize_game(game)
        prompt = format_prompt(anonymized_game)

        print(f"Sending query to LLM ({model})...")

        prediction = None
        for attempt in range(max_retries):
            try:
                prediction = query_llm(prompt, model)
                break
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("Failed after multiple retries, skipping this game")
                    prediction = "error"

        print(f"LLM prediction: {prediction}")

        is_correct = evaluate_prediction(anonymized_game, prediction) if prediction != "error" else False
        if is_correct:
            correct_predictions += 1

        results.append({
            "game_id": game.id,
            "anonymized_game": anonymized_game,
            "prompt": prompt,
            "prediction": prediction,
            "correct_bot": anonymized_game["user_to_color"].get(prediction) == anonymized_game["correct_bot_color"] if prediction != "error" else False,
            "correct_bot_user": next(user for user, color in anonymized_game["user_to_color"].items()
                                     if color == anonymized_game["correct_bot_color"])
        })

        print(f"Prediction correct: {is_correct}")

        if i < len(games) - 1:
            time.sleep(1)

        print("-" * 40)

    accuracy = correct_predictions / len(games) if games else 0
    print(f"\nExperiment complete. Accuracy: {accuracy:.2%} ({correct_predictions}/{len(games)})")

    return {
        "total_games": len(games),
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "model": model,
        "game_results": results
    }

def main():
    parser = argparse.ArgumentParser(description='Run Turing Game Experiment')
    parser.add_argument('--input-dir', required=True, help='Directory containing log files to process')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    parser.add_argument('--model', default='gpt-4',
                        help='LLM model to use (e.g., gpt-4, gemini-1.5-pro, gemini-1.5-flash)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API calls')

    args = parser.parse_args()

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if os.path.isfile(os.path.join(args.input_dir, f))]

    output_file = args.output

    if not input_files:
        print(f"No files found in directory: {args.input_dir}")
        return

    print(f"Found {len(input_files)} files to process")
    outputs = {}
    for i, log_file in enumerate(input_files, 1):
        print(f"\nProcessing file {i}/{len(input_files)}: {log_file}")
        try:
            outputs[log_file] = run_experiment(log_file, args.model, args.max_retries)
        except Exception as e:
            print(f"Error processing file {log_file}: {e}")
            continue

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2,separators=(',', ': '))
        print(f"Completed processing {log_file}")

if __name__ == "__main__":
    main()