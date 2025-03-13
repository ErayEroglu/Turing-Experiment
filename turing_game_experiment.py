import re
import json
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse
import random
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Game:
    id: str
    users: Dict[str, str]
    chat: List[Tuple[str, str]]

    def to_dict(self):
        return asdict(self)

def parse_log_file(filename: str) -> List[Game]:
    """Parse the log file and extract games."""
    logger.info(f"Parsing file: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Failed to read file {filename}: {e}")
        return []

    game_blocks = re.findall(r'<details>\s*<summary>Game \d+: \(ID: (\d+)\)</summary>(.*?)</details>',
                             content, re.DOTALL)

    logger.info(f"Found {len(game_blocks)} game blocks in {filename}")

    games = []
    for game_id, game_content in game_blocks:
        try:
            user_table = re.search(r'\| User \| Color \|\s*\| ---- \| ----- \|(.*?)\s*###',
                                   game_content, re.DOTALL)

            if not user_table:
                logger.warning(f"Game {game_id}: User table not found, skipping")
                continue

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
                        if user_role in ["You", "Other human"]:
                            user_roles[user_role] = color

            if len(users) != 3 or list(users.values()).count('bot') != 1:
                logger.warning(f"Game {game_id}: Expected 3 users with 1 bot, found {len(users)} users with {list(users.values()).count('bot')} bots. Skipping.")
                continue

            chat_section = re.search(r'### The Chat:(.*?)### The Accusations:',
                                     game_content, re.DOTALL)

            if not chat_section:
                logger.warning(f"Game {game_id}: Chat section not found, skipping")
                continue

            chat = []
            for line in chat_section.group(1).strip().split('\n'):
                if '): **' in line:
                    message_match = re.search(r'\((.+?)\): \*\*(.+?)\*\*', line)
                    if message_match:
                        emoji, message = message_match.groups()
                        if emoji in emoji_to_color:
                            color = emoji_to_color[emoji]
                            chat.append((color, message))
                        else:
                            logger.warning(f"Game {game_id}: Could not map emoji '{emoji}' to a color")

            if len(chat) < 2:
                logger.warning(f"Game {game_id}: Not enough chat messages (found {len(chat)}), skipping")
                continue

            accusations_section = re.search(r'### The Accusations:(.*?)$',
                                            game_content, re.DOTALL)

            if not accusations_section:
                logger.warning(f"Game {game_id}: Accusations section not found, skipping")
                continue

            games.append(Game(
                id=game_id,
                users=users,
                chat=chat,
            ))
            logger.info(f"Game {game_id}: Successfully parsed")

        except Exception as e:
            logger.error(f"Error parsing game {game_id}: {e}")
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Successfully parsed {len(games)}/{len(game_blocks)} games from {filename}")
    return games

def anonymize_game(game: Game) -> Dict:
    """Convert a game to an anonymized format for LLM input."""
    # First, ensure consistent ordering
    all_colors = list(game.users.keys())
    random.shuffle(all_colors)  # Randomize to avoid position bias

    color_to_user = {}
    for i, color in enumerate(all_colors, 1):
        color_to_user[color] = f"user{i}"

    anonymized_chat = []
    for color, message in game.chat:
        if color in color_to_user:
            anonymized_chat.append((color_to_user[color], message))
        else:
            logger.warning(f"Game {game.id}: Color '{color}' not found in color_to_user mapping")

    # Validate that all three users appear in the chat
    users_in_chat = set(user for user, _ in anonymized_chat)
    if len(users_in_chat) < 3:
        logger.warning(f"Game {game.id}: Not all users appear in chat (only {users_in_chat})")

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

def query_llm(prompt: str, model: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Send a prompt to the LLM and get a response.

    Returns:
        Dict with keys:
        - prediction: the user number predicted
        - raw_response: the full text response
        - success: boolean indicating if the call succeeded
        - error: error message if any
    """
    start_time = time.time()

    # Default return structure
    result = {
        "prediction": None,
        "raw_response": None,
        "success": False,
        "error": None,
        "latency": 0
    }

    try:
        if model.startswith("claude") and config.get("anthropic_api_key"):
            response = query_anthropic(prompt, config.get("anthropic_api_key"), model)
            result.update(response)

        elif model.startswith("gpt") and config.get("openai_api_key"):
            response = query_openai(prompt, config.get("openai_api_key"), model)
            result.update(response)

    except Exception as e:
        logger.error(f"Error in query_llm for {model}: {e}")
        logger.debug(traceback.format_exc())
        result["error"] = str(e)

    # Record latency
    result["latency"] = time.time() - start_time

    return result

def query_openai(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    """Query OpenAI API (supports GPT and Claude models via OpenAI API)."""
    if not api_key:
        raise ValueError("OpenAI API key not provided")

    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100
        )

        raw_response = response.choices[0].message.content.strip()

        # Extract prediction using regex to be more robust
        import re
        user_match = re.search(r'user\s*(\d+)', raw_response.lower())

        if user_match:
            prediction = f"user{user_match.group(1)}"
        else:
            if "user1" in raw_response.lower():
                prediction = "user1"
            elif "user2" in raw_response.lower():
                prediction = "user2"
            elif "user3" in raw_response.lower():
                prediction = "user3"
            else:
                prediction = None
                logger.warning(f"Could not extract prediction from response: {raw_response}")

        return {
            "prediction": prediction,
            "raw_response": raw_response,
            "success": prediction is not None
        }

    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

def query_anthropic(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    """Send a prompt to Anthropic's Claude model and get a response."""
    if not api_key:
        raise ValueError("Anthropic API key not provided")

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw_response = response.content[0].text.strip()
        logger.info(f"Raw response: {raw_response}")
        
        # Extract prediction using regex to be more robust
        import re
        user_match = re.search(r'user\s*(\d+)', raw_response.lower())
        
        if user_match:
            prediction = f"user{user_match.group(1)}"
        else:
            # Fallback to exact matching if regex fails
            if "user1" in raw_response.lower():
                prediction = "user1"
            elif "user2" in raw_response.lower():
                prediction = "user2"
            elif "user3" in raw_response.lower():
                prediction = "user3"
            else:
                prediction = None
                logger.warning(f"Could not extract prediction from response: {raw_response}")
                
        return {
            "prediction": prediction,
            "raw_response": raw_response,
            "success": prediction is not None
        }

    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

def evaluate_prediction(anonymized_game: Dict, prediction: str) -> bool:
    """Check if the prediction correctly identified the bot."""
    if not prediction or prediction not in anonymized_game["user_to_color"]:
        return False

    correct_bot_color = anonymized_game["correct_bot_color"]
    predicted_color = anonymized_game["user_to_color"].get(prediction)
    return predicted_color == correct_bot_color

def run_experiment(log_file: str, model: str, config: Dict[str, Any], max_retries: int = 3) -> dict:
    """Run the full experiment and return results."""
    logger.info(f"Starting experiment on {log_file} with model {model}")

    games = parse_log_file(log_file)
    logger.info(f"Found {len(games)} valid games")

    if not games:
        logger.warning(f"No valid games found in {log_file}, skipping experiment")
        return {
            "total_games": 0,
            "correct_predictions": 0,
            "accuracy": 0,
            "model": model,
            "game_results": [],
            "status": "failed",
            "error": "No valid games found"
        }

    results = []
    correct_predictions = 0
    total_latency = 0
    api_errors = 0

    for i, game in enumerate(games):
        game_start_time = time.time()
        logger.info(f"Processing game {i+1}/{len(games)} (ID: {game.id})")

        try:
            anonymized_game = anonymize_game(game)
            prompt = format_prompt(anonymized_game)

            # Track API calls for rate limiting/budgeting
            logger.info(f"Sending query to LLM ({model})...")

            llm_result = None
            for attempt in range(max_retries):
                try:
                    llm_result = query_llm(prompt, model, config)

                    if llm_result["success"]:
                        break

                    logger.warning(f"Unsuccessful query on attempt {attempt+1}/{max_retries}: {llm_result.get('error')}")

                    if attempt < max_retries - 1:
                        wait_time = min(30, 10 * (attempt + 1))
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)

                except Exception as e:
                    logger.error(f"Error on attempt {attempt+1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = min(30, 2 ** attempt)
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Failed after {max_retries} retries")
                        api_errors += 1

            if not llm_result or not llm_result["success"]:
                logger.error(f"Failed to get valid prediction for game {game.id}")
                prediction = None
                raw_response = llm_result.get("raw_response") if llm_result else None
                error_msg = llm_result.get("error") if llm_result else "Unknown error"
            else:
                prediction = llm_result["prediction"]
                raw_response = llm_result["raw_response"]
                error_msg = None
                total_latency += llm_result.get("latency", 0)

            logger.info(f"LLM prediction: {prediction}")

            is_correct = evaluate_prediction(anonymized_game, prediction) if prediction else False
            if is_correct:
                correct_predictions += 1
                logger.info(f"✅ Correct prediction")
            else:
                logger.info(f"❌ Incorrect prediction")

            results.append({
                "game_id": game.id,
                "prediction": prediction,
                "raw_response": raw_response,
                "error": error_msg,
                "is_correct": is_correct,
                "processing_time": time.time() - game_start_time,
                "correct_bot_user": next(user for user, color in anonymized_game["user_to_color"].items()
                                         if color == anonymized_game["correct_bot_color"])
            })

        except Exception as e:
            logger.error(f"Error processing game {game.id}: {e}")
            logger.debug(traceback.format_exc())
            results.append({
                "game_id": game.id,
                "error": str(e),
                "is_correct": False
            })

        if i < len(games) - 1:
            time.sleep(10)

    # Calculate metrics
    completed_games = len([r for r in results if "error" not in r or not r["error"]])
    accuracy = correct_predictions / len(games) if games else 0
    avg_latency = total_latency / completed_games if completed_games else 0

    logger.info(f"Experiment complete.")
    logger.info(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{len(games)})")
    logger.info(f"API Errors: {api_errors}")
    logger.info(f"Average latency: {avg_latency:.2f}s")

    return {
        "total_games": len(games),
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "model": model,
        "avg_latency": avg_latency,
        "api_errors": api_errors,
        "game_results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()

    keys = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY")
    }

    available_keys = [k for k, v in keys.items() if v]
    if not available_keys:
        logger.warning("No API keys found in environment variables")
    else:
        logger.info(f"Found API keys for: {', '.join(k.replace('_api_key', '') for k in available_keys)}")

    return keys

def main():
    parser = argparse.ArgumentParser(description='Run Turing Game Experiment')
    parser.add_argument('--input-dir', required=True, help='Directory containing log files to process')
    parser.add_argument('--output', default='results.json', help='Output file for results')
    parser.add_argument('--model', default='gpt-4',
                        help='LLM model to use (e.g., gpt-4, claude-3-opus-20240229)')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries for API calls')
    parser.add_argument('--mock', action='store_true', help='Use mock responses for testing without API calls')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Delay between API calls in seconds')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    api_keys = load_api_keys()

    config = {
        **api_keys,
        "allow_mock": args.mock,
        "rate_limit_delay": args.rate_limit
    }

    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return

    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                   if os.path.isfile(os.path.join(args.input_dir, f)) and f.endswith(('.md', '.txt'))]

    if not input_files:
        logger.error(f"No .md or .txt files found in directory: {args.input_dir}")
        return

    logger.info(f"Found {len(input_files)} files to process")

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results = {}
    summary = {
        "total_files": len(input_files),
        "files_processed": 0,
        "total_games": 0,
        "correct_predictions": 0,
        "model": args.model,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "average_accuracy": 0.0
    }

    for i, log_file in enumerate(input_files, 1):
        logger.info(f"\nProcessing file {i}/{len(input_files)}: {log_file}")
        try:
            file_results = run_experiment(log_file, args.model, config, args.max_retries)
            file_basename = os.path.basename(log_file)
            all_results[file_basename] = file_results

            summary["files_processed"] += 1
            summary["total_games"] += file_results.get("total_games", 0)
            summary["correct_predictions"] += file_results.get("correct_predictions", 0)

            combined_results = {
                "summary": summary,
                "results": all_results
            }

            # Calculate average accuracy
            if summary["total_games"] > 0:
                summary["average_accuracy"] = summary["correct_predictions"] / summary["total_games"]

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, indent=2, separators=(',', ': '))

            logger.info(f"Saved intermediate results to {args.output}")

        except Exception as e:
            logger.error(f"Error processing file {log_file}: {e}")
            logger.debug(traceback.format_exc())

    summary["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if summary["total_games"] > 0:
        summary["average_accuracy"] = summary["correct_predictions"] / summary["total_games"]

    combined_results = {
        "summary": summary,
        "results": all_results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, separators=(',', ': '))

    logger.info(f"\nExperiment complete!")
    logger.info(f"Processed {summary['files_processed']}/{len(input_files)} files")
    logger.info(f"Total games: {summary['total_games']}")
    logger.info(f"Overall accuracy: {summary['average_accuracy']:.2%} ({summary['correct_predictions']}/{summary['total_games']})")
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()