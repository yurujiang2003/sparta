import os
import sys
import json
import random
import logging
import argparse
from typing import List, Dict
from datetime import datetime
from time import sleep

import torch
import torch.multiprocessing as mp
from datasets import Dataset
from multiprocessing import Pool


from model_init import ModelInit, ModelInitFair
from rating_tracker import RatingTracker, StopCriteria
from rating_system import RatingSystem, RatingSystemDynamicWeighted, RatingSystemStaticWeighted
from judge import Judge, run_judges, calculate_judge_averages
from inference import Inference


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_model_group(args):
    """
    process the model group on a single GPU
    """
    group, model_tasks, gpu_id, model_paths, model_types = args  # model_types: dict
    model_responses = {}
    
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

        for model_name in group:
            if model_name not in model_tasks or not model_tasks[model_name]:
                continue
                
            unique_tasks = list(set(model_tasks[model_name]))
            print(f"Processing model {model_name} on GPU {gpu_id}")
            
            try:
                inference = Inference(
                    model_name=model_name,
                    gpu_id=gpu_id,
                    model_path=model_paths[model_name],
                    base_model=model_types[model_name]  # 用各自的model_type
                )
                
                result = inference.batch_generate_responses(
                    instructions=unique_tasks,  
                    batch_size=24, 
                    max_new_tokens=512,
                    use_chat_template=True
                )
                
                model_responses[model_name] = {
                    instruction: response
                    for instruction, response in zip(unique_tasks, result)
                }
                
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                continue
                
            finally:

                if 'inference' in locals():
                    del inference.model
                    del inference
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sleep(3)  
        
        return model_responses
        
    except Exception as e:
        print(f"Error in process_model_group: {e}")
        return {}
        
    finally:
        cleanup_resources()

class Competition:
    def __init__(self, model_configs, model_info, num_opponents, random_match_prob, is_random_select=True):
        self.model_configs = model_configs
        self.model_names = [config["name"] for config in model_configs]
        self.model_paths = {config["name"]: config["path"] for config in model_configs}
        self.model_types = {config["name"]: config["model_type"] for config in model_configs}  # 新增
        self.model_info = model_info
        self.num_opponents = num_opponents
        self.random_match_prob = random_match_prob
        self.is_random_select = is_random_select

    def generate_idx(self, model_name: str, instructions: List[str], gpu_id: int) -> List[str]:
        inference = None
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, -1))

            device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()

            model_path = self.model_paths[model_name]
            model_type = self.model_types[model_name]  # 用各自的model_type
            inference = Inference(
                model_name=model_name,
                gpu_id=gpu_id,
                model_path=model_path,
                base_model=model_type
            )
            
            responses = inference.batch_generate_responses(
                instructions=instructions,
                batch_size=12,  
                max_new_tokens=512,
                use_chat_template=True
            )
            
            return responses
            
        except Exception as e:
            print(f"Generate error for {model_name}: {e}")
            return [f"Error: {str(e)}"] * len(instructions)
            
        finally:
            if inference is not None:
                del inference.model
                del inference
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_opponent(self, current_model: str) -> str:
        """choose the opponent based on the score"""
        current_score = self.model_info[current_model]['score']

        if random.random() < self.random_match_prob:
            opponents = [m for m in self.model_names if m != current_model]
            return random.choice(opponents)

        potential_opponents = []
        for other_model in self.model_names:
            if other_model != current_model:
                score_diff = abs(current_score - self.model_info[other_model]['score'])
                potential_opponents.append((other_model, score_diff))

        potential_opponents.sort(key=lambda x: x[1])
        print(f"random_match_prob: {self.random_match_prob}")
        return random.choice(potential_opponents[:self.num_opponents])[0]

    def run(self, instructions: List[str]) -> List[Dict]:
        model_tasks = {model: [] for model in self.model_names}
        instruction_pairs = []

        if self.is_random_select:
            # Calculate number of complete loops
            num_loops = len(instructions) // len(self.model_names)
            
            # Process each complete loop
            for loop_idx in range(num_loops):
                # Get instructions for this loop
                start_idx = loop_idx * len(self.model_names)
                end_idx = start_idx + len(self.model_names)
                loop_instructions = instructions[start_idx:end_idx]
                
                # Randomly shuffle models for this loop
                loop_models = random.sample(self.model_names, len(self.model_names))
                
                # Create pairs for each model in the shuffled order
                for model_idx, first_model in enumerate(loop_models):
                    instruction = loop_instructions[model_idx]
                    opponent_model = self.get_opponent(first_model)
                    
                    instruction_pairs.append((instruction, first_model, opponent_model))
                    model_tasks[first_model].append(instruction)
                    model_tasks[opponent_model].append(instruction)
            
            # Handle remaining instructions
            remaining_start = num_loops * len(self.model_names)
            for idx, instruction in enumerate(instructions[remaining_start:]):
                first_model = random.choice(self.model_names)
                opponent_model = self.get_opponent(first_model)
                
                instruction_pairs.append((instruction, first_model, opponent_model))
                model_tasks[first_model].append(instruction)
                model_tasks[opponent_model].append(instruction)
        else:
            # Original sequential selection logic
            for idx, instruction in enumerate(instructions):
                first_model = self.model_names[idx % len(self.model_names)]
                opponent_model = self.get_opponent(first_model)

                instruction_pairs.append((instruction, first_model, opponent_model))
                model_tasks[first_model].append(instruction)
                model_tasks[opponent_model].append(instruction)

        # Process model responses in groups
        model_groups = []
        models_list = list(self.model_names)
        for i in range(0, len(models_list), 2):
            if i + 1 < len(models_list):
                model_groups.append((models_list[i], models_list[i+1]))
            else:
                model_groups.append((models_list[i],))

        model_responses = {}
        with Pool(processes=len(model_groups)) as pool:
            process_args = [
                (group, model_tasks, gpu_id, self.model_paths, self.model_types)  # 传入model_types
                for gpu_id, group in enumerate(model_groups)
            ]

            results = pool.map(process_model_group, process_args)
            
            for group_responses in results:
                model_responses.update(group_responses)

        # Generate raw pairs
        raw_pairs = []
        for instruction, model_a, model_b in instruction_pairs:
            if (model_a in model_responses and 
                model_b in model_responses and
                instruction in model_responses[model_a] and
                instruction in model_responses[model_b]):
                
                response_list = [
                    {
                        'model_name': model_a,
                        'response': model_responses[model_a][instruction]
                    },
                    {
                        'model_name': model_b,
                        'response': model_responses[model_b][instruction]
                    }
                ]
                raw_pairs = self.pair(raw_pairs, instruction, response_list)
        
        return raw_pairs

    def pair(self, raw_pairs: List[Dict], instruction: str, response_list: List[Dict]) -> List[Dict]:
        """
        organize the responses into pairs
        
        Args:
            raw_pairs (List[Dict]): existing pairs list
            instruction (str): current instruction
            response_list (List[Dict]): list of responses, each element is a dictionary containing 'model_name' and 'response'
        
        Returns:
            List[Dict]: updated pairs list
        """
        # check if the response list is valid
        if len(response_list) != 2:
            print(f"Invalid response list length: {len(response_list)}")
            return raw_pairs

        new_pair = {
            'instruction': instruction,
            'models': [resp['model_name'] for resp in response_list],
            'responses': [resp['response'] for resp in response_list],
            'judges': {} 
        }

        raw_pairs.append(new_pair)
        return raw_pairs

def judge_with_gpu(args):
    """
    Helper function for multiprocessing
    Args:
        args: tuple of (judge_name, judge_path, pairs, gpu_id)
    Returns:
        tuple: (judge_name, judged_pairs)
    """
    judge_name, pairs, judge_path, gpu_id = args
    judge = Judge(judge_name, judge_path, gpu_id)
    judged_pairs = judge.judge(pairs)
    return judge_name, judged_pairs

def save_judged_pairs(judged_pairs, base_dir, iteration):
    """
    save the judged pairs to the specified directory
    
    Args:
        judged_pairs (List[Dict]): judged pairs
        base_dir (str): base directory
        iteration (int): current iteration
    """
    try:
        save_dir = os.path.join(base_dir, f"iteration_{iteration}", "judged_results")
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, "judged_pairs.json")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)
            
        print(f"\nJudged pairs saved to: {file_path}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(save_dir, f"judged_pairs_{timestamp}.json")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(judged_pairs, f, indent=2, ensure_ascii=False)
            
        print(f"Backup saved to: {backup_path}")
        
    except Exception as e:
        print(f"Error saving judged pairs: {e}")

def save_preference_pairs_to_json(preference_pairs, base_dir, filename):
    """
    Save preference pairs to a JSON file in the specified directory.
    
    Args:
        preference_pairs (list): The list of preference pairs to save
        base_dir (str): Base directory path
        filename (str): Name of the JSON file
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        file_path = os.path.join(base_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(preference_pairs, f, indent=2, ensure_ascii=False)
        print(f"Preference pairs saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving preference pairs to JSON: {e}")

def save_rating_history(rating_history, base_dir, iteration):
    """
    Save the detailed rating history to a JSON file in the specified directory.
    
    Args:
        rating_history (list): List of rating snapshots after each pair
        base_dir (str): Base directory path
        iteration (int): Current iteration number
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        
        file_path = os.path.join(base_dir, f"iteration_{iteration}_rating_history.json")
        
        history_data = {
            'iteration': iteration,
            'total_pairs': len(rating_history),
            'history': rating_history,
            'final_ratings': rating_history[-1]['ratings'] if rating_history else None,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"Detailed rating history saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving rating history to JSON: {e}")

def extract_from_knowledge_crossword(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the knowledge crossword dataset
    
    Args:
        file_path: Path to the JSONL file
        num: Number of questions to extract
        
    Returns:
        List of instruction strings
    """
    try:
        instructions = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'prompt' in item:
                        instructions.append(item['prompt'])
                    else:
                        print(f"Warning: Missing 'prompt' field in an item")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in file")
                    continue
        
        if not instructions:
            print(f"Warning: No valid instructions found in {file_path}")
            return []
        
        # Check if num is greater than available instructions
        if num > len(instructions):
            print(f"Warning: Requested {num} instructions but only {len(instructions)} are available")
            return instructions  # Return all available instructions
            
        # Shuffle and limit the number of instructions
        random.shuffle(instructions)
        return instructions[:num]
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []


def extract_from_alpaca(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the alpaca dataset
    
    Args:
        file_path: Path to the JSON file
        num: Number of questions to extract
        
    Returns:
        List of instruction strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract instructions from the data
        instructions = []
        for item in data:
            if isinstance(item, dict) and 'instruction' in item:
                instructions.append(item['instruction'])
        
        # Check if we found any instructions
        if not instructions:
            print(f"Warning: No instructions found in {file_path}")
            return []
            
        # Shuffle and limit the number of instructions
        random.shuffle(instructions)
        return instructions[:num]
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []
    

def extract_from_culture(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the culture dataset, only from the 'train' split.
    
    Args:
        file_path: Path to the JSON file
        num: Number of questions to extract
        
    Returns:
        List of instruction strings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Only extract from the 'train' split
        questions = []
        
        if 'train' in data:
            train_data = data['train']
            # For each item in the train split (indexed by IDs)
            for item_id in train_data:
                item = train_data[item_id]
                if 'instruction' in item:
                    questions.append(item['instruction'])
        else:
            print("Warning: No 'train' split found in the data")
        
        # Shuffle and limit the number of questions
        random.shuffle(questions)
        return questions[:num]
    
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return []

def extract_from_json_truthfulqa(file_path: str, num: int) -> List[str]:
    """
    From the JSONL file, extract the specified number of unique questions.
    
    Args:
        file_path (str): Path to the JSONL file
        num (int): Number of questions to extract
        
    Returns:
        List[str]: Extracted questions list (format: Q: ...\nA: ...)

    Raises:
        FileNotFoundError: When the file does not exist
        json.JSONDecodeError: When JSON parsing fails
    """
    questions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    item = json.loads(line.strip())
                    
                    # Extract question from prompt field
                    if 'prompt' not in item:
                        print(f"Warning: Missing 'prompt' field at line {line_num}")
                        continue
                        
                    questions.append(item['prompt'])
                    
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON at line {line_num}")
                    continue
        
        if not questions:
            print(f"Warning: No valid questions found in {file_path}")
            return []
            
        # Remove duplicates and shuffle
        unique_questions = list(set(questions))
        random.shuffle(unique_questions)
        
        # Handle requested number
        available_num = len(unique_questions)
        if num > available_num:
            print(f"Warning: Requested {num} questions but only {available_num} available")

            return unique_questions
        print(unique_questions[:num])
        return unique_questions[:num]
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error while processing file: {str(e)}")
        return []


def extract_from_json_gsm(file_path: str, num: int) -> List[str]:
    """
    Extract a specified number of unique inputs from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        num (int): Number of inputs to extract
        
    Returns:
        List[str]: List of extracted inputs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            inputs = [item['input'] for item in data]
            
        random.shuffle(inputs)
        
        return inputs[:num]
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting inputs: {e}")
        return []

def extract_from_math(file_path: str, num: int, difficulty: str) -> List[str]:
    """
    Extract questions from the math dataset
    
    Args:
        file_path (str): Path to the JSONL file
        num (int): Number of problems to extract
        
    Returns:
        List[str]: List of extracted problems with formatted prompts
    """
    try:
        problems = []
        if difficulty == 'easy':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'easy.jsonl')
        elif difficulty == 'medium':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'medium.jsonl')
        elif difficulty == 'hard':
            file_path = os.path.join(file_path, 'train_by_difficulty', 'hard.jsonl')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                prompt = data['problem'] + "Let's solve this problem step by step."
                problems.append(prompt)

        if num > len(problems):
            print(f"Warning: Requested {num} problems but only {len(problems)} available")
            random.shuffle(problems)
            return problems

        random.shuffle(problems)
        return problems[:num]
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error extracting problems: {e}")
        return []

def extract_from_com2(file_path: str, num: int) -> List[str]:
    """
    Extract questions from the com2 dataset
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            instructions = [item['prompt'] for item in data]

        if num > len(instructions):
            print(f"Warning: Requested {num} instructions but only {len(instructions)} available")
            random.shuffle(instructions)
            return instructions

        random.shuffle(instructions)
        return instructions[:num]
    except Exception as e:
        print(f"Error extracting instructions: {e}")
        return []
    
def cleanup_resources():
    """more thorough resource cleanup function"""
    try:

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass

        import gc
        gc.collect()

        import psutil
        process = psutil.Process()
        try:
            process.memory_full_info()
        except:
            pass

        sleep(5)
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

def save_model_info(model_info: Dict, base_dir: str, iteration: int):
    """save model_info to JSON file"""
    try:
        save_path = os.path.join(base_dir, f"iteration_{iteration}", "model_info.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        serializable_info = {}
        for model_name, info in model_info.items():
            serializable_info[model_name] = {
                'score': float(info['score']),
                'deviation': float(info['deviation'])
            }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, indent=2)
            
        logger.info(f"Model info saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving model info: {e}")


def save_judge_pairs(judge_pairs, output_dir, iteration):
    """
    save judge_pairs to the specified directory
    
    Args:
        judge_pairs (list): list of judge pairs
        output_dir (str): output directory
        iteration (int): iteration number
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, f"iteration_{iteration}_judge_pairs.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(judge_pairs, f, indent=2, ensure_ascii=False)
            
        print(f"Judge pairs saved to: {filepath}")
        
    except Exception as e:
        print(f"Error saving judge pairs: {e}")

def load_model_info(base_dir: str, iteration: int) -> Dict:
    """load model_info from JSON file"""
    try:
        load_path = os.path.join(base_dir, f"iteration_{iteration}", "model_info.json")
        
        if os.path.exists(load_path):
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                model_name: {
                    'score': 100,
                    'deviation': 0.5
                }
                for model_name in ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", 
                                 "lima", "oasst1", "open_orca", "science", 
                                 "sharegpt", "wizardlm"]
            }
            
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        return None

def filter_tie(preference_pairs):
    """
    filter the tie pairs
    """
    return [pair for pair in preference_pairs if pair['score_diff'] != 0]

def main():
    try:
        parser = argparse.ArgumentParser(description='Competition Phase')
        parser.add_argument('iteration', type=int, help='Current iteration number')
        parser.add_argument('--fair', action='store_true', help='Use fair initialization')
        parser.add_argument('--no-fair', action='store_false', dest='fair', help='Use unfair initialization')
        parser.set_defaults(fair=True)
        
        # 修改 random_select 参数类型为 store_true/store_false
        parser.add_argument('--random_select', type=lambda x: x.lower() == 'true', default=True,
                          help='Use random match (true/false)')
        
        parser.add_argument('--task', type=str, default='gsm8k', help='Task name, choose from [gsm8k, alpaca, truthfulqa, culture_country, culture_value, culture_rule_of_thumb, knowledge_crossword]')
        parser.add_argument('--base_model', type=str, default='gemma,gemma,gemma,gemma,gemma,gemma,gemma,gemma,gemma,gemma', 
                          help='Comma-separated list of base model types, each corresponding to model_names at same index. Choose from [gemma, qwen, deepseek]')
        parser.add_argument('--score_type', type=str, default='normal', 
                          help='Score type: normal or dynamic')
        parser.add_argument('--base_dir', type=str, required=True,
                          help='Base directory for model storage')
        parser.add_argument('--init_model_dir', type=str, required=True,
                          help='Directory containing initial models')
        parser.add_argument('--data_dir', type=str, required=True,
                          help='Directory containing training data')
        parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4',
                          help='Comma-separated list of GPU IDs to use')

        parser.add_argument('--num_instructions', type=int, default=500,
                          help='Number of instructions to process')
        parser.add_argument('--batch_size', type=int, default=16,
                          help='Batch size for model inference')
        parser.add_argument('--model_names', type=str, 
                          default='code_alpaca,cot,flan_v2,gemini_alpaca,lima,oasst1,open_orca,science,sharegpt,wizardlm',
                          help='Comma-separated list of model names')
        parser.add_argument('--difficulty', type=str, default='easy',
                          help='Difficulty level, choose from [easy, medium, hard]')

        parser.add_argument('--random_match_prob', type=float, default=0.2,
                          help='Probability of random opponent selection')
        parser.add_argument('--num_opponents', type=int, default=3,
                          help='Number of potential opponents to choose from')

        parser.add_argument('--random_seed', type=int, default=42,
                          help='Random seed for reproducibility')

        parser.add_argument('--initial_k', type=float, default=10.0,
                          help='Initial K factor for rating system')
        parser.add_argument('--min_k', type=float, default=5.0,
                          help='Minimum K factor for rating system')
        parser.add_argument('--window_size', type=int, default=10,
                          help='Window size for rating history')
        parser.add_argument('--min_deviation', type=float, default=0.1,
                          help='Minimum deviation threshold')
        parser.add_argument('--epsilon', type=float, default=0.01,
                          help='Epsilon for rating calculations')
        parser.add_argument('--decay_rate', type=float, default=0.9,
                          help='Decay rate for rating adjustments')
        parser.add_argument('--decay_steps', type=int, default=10,
                          help='Number of steps for decay')
        parser.add_argument('--scaling_factor', type=float, default=20.0,
                          help='Scaling factor for rating calculations')
        
        # 添加 freeze_ratings 参数
        parser.add_argument('--freeze_ratings', type=lambda x: x.lower() == 'true', default=False,
                          help='Freeze model ratings and deviations (true/false)')
        
        args = parser.parse_args()

        random.seed(args.random_seed)

        model_names = args.model_names.split(',')
        base_models = args.base_model.split(',')

        # 确保 base_models 和 model_names 长度一致
        if len(base_models) != len(model_names):
            if len(base_models) == 1:
                # 如果只提供一个base_model，复制到所有模型
                base_models = [base_models[0]] * len(model_names)
            else:
                raise ValueError(f"base_model list length ({len(base_models)}) must match model_names length ({len(model_names)})")

        # 创建 model_name 到 base_model 的映射
        model_base_mapping = dict(zip(model_names, base_models))

        iteration = args.iteration
        task = args.task
        base_dir = args.base_dir
        base_model_dir = base_dir
        score_type = args.score_type
        fair_or_not = args.fair
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
        

        # Update model registration path
        def model_register(model_name):
            model_base = model_base_mapping[model_name]
            if os.path.exists(os.path.join(args.init_model_dir, model_name)):
                if fair_or_not:
                    init_model_dir = os.path.join(args.init_model_dir, model_base)
                    model_init = ModelInitFair(save_dir=init_model_dir, task=task, base_model=model_base)
                else:
                    init_model_dir = os.path.join(args.init_model_dir, model_base)
                    model_init = ModelInit(save_dir=init_model_dir, task=task, base_model=model_base)
                return model_init.init_model(model_name)
            else:
                if fair_or_not:
                    init_model_dir = os.path.join(args.init_model_dir, model_base)
                    model_init = ModelInitFair(save_dir=init_model_dir, task=task, base_model=model_base)
                else:
                    init_model_dir = os.path.join(args.init_model_dir, model_base)
                    model_init = ModelInit(save_dir=init_model_dir, task=task, base_model=model_base)
                model_init.init_model_local(model_name)
                return model_init.init_model(model_name)

        for module_name in list(sys.modules.keys()):
            if 'unsloth' in module_name:
                del sys.modules[module_name]
        
        if iteration > 0:
            model_info = load_model_info(base_dir, iteration - 1)
            if model_info is None:
                raise ValueError(f"Failed to load model info from iteration {iteration-1}")
        else:
            model_info = {
                model_name: model_register(model_name)
                for model_name in model_names
            }

        print(f"Running iteration {iteration}")

        for model_name in model_names:
            if model_name not in model_info:
                model_info[model_name] = model_register(model_name)
        
        if iteration > 0:
            for model_name in model_names:
                model_path = os.path.join(base_model_dir, f"iteration_{iteration-1}", model_name)

                if model_name in model_info:
                    model_info[model_name]['path'] = model_path

        model_configs = []
        for model_name in model_names:
            model_type = model_base_mapping[model_name]
            
            if iteration == 0:
                # 初始模型路径格式: init_model_dir/model_type/model_name
                model_path = os.path.join(args.init_model_dir, model_type, model_name)
            else:
                # 非初始迭代：使用上一轮的结果
                model_path = os.path.join(base_model_dir, f"iteration_{iteration-1}", model_name)
            
            output_path = os.path.join(base_model_dir, f"iteration_{iteration}", model_name)
            
            model_configs.append({
                "name": model_name,
                "path": model_path,
                "output_path": output_path,
                "learning_rate": 1e-5,
                "model_type": model_type
            })
        
        if args.random_select:
            competition = Competition(
                model_configs, 
                model_info, 
                num_opponents=args.num_opponents,
                random_match_prob=args.random_match_prob,
                is_random_select=True
            )
        else:
            competition = Competition(
                model_configs, 
                model_info, 
                num_opponents=args.num_opponents,
                random_match_prob=args.random_match_prob,
                is_random_select=False
            )

        if task == 'gsm8k':
            total_instructions = extract_from_json_gsm(
                os.path.join(args.data_dir, 'gsm8k/questions.json'),
                args.num_instructions
            )
        elif task == 'truthfulqa_mc1,truthfulqa_mc2':
            total_instructions = extract_from_json_truthfulqa(
                os.path.join(args.data_dir, 'truthfulqa/data.jsonl'),
                args.num_instructions
            )
        elif task == 'culture_country':
            total_instructions = extract_from_culture(
                os.path.join(args.data_dir, 'culture/country_dataset.json'),
                args.num_instructions
            )
        elif task == 'culture_value':
            total_instructions = extract_from_culture(
                os.path.join(args.data_dir, 'culture/country_value_dataset.json'),
                args.num_instructions
            )
        elif task == 'culture_rule_of_thumb':
            total_instructions = extract_from_culture(
                os.path.join(args.data_dir, 'culture/rule_of_thumb_dataset.json'),
                args.num_instructions
            )
        elif task == 'alpaca':
            total_instructions = extract_from_alpaca(
                os.path.join(args.data_dir, 'alpaca/alpaca_data_processed.json'),
                args.num_instructions
            )
        elif task == 'kc_knowledge':
            total_instructions = extract_from_knowledge_crossword(
                os.path.join(args.data_dir, 'knowledge_crossword/KC_train_knowledge.jsonl'),
                args.num_instructions
            )
        elif task == 'kc_zeroshot':
            total_instructions = extract_from_knowledge_crossword(
                os.path.join(args.data_dir, 'knowledge_crossword/KC_train_zeroshot.jsonl'),
                args.num_instructions
            )
        elif task == 'math':
            total_instructions = extract_from_math(
                os.path.join(args.data_dir, 'MATH'),
                args.num_instructions,
                args.difficulty
            )
        elif task == 'com2':
            total_instructions = extract_from_com2(
                os.path.join(args.data_dir, 'com2/train.json'),
                args.num_instructions
            )
        else:
            raise ValueError(f"Invalid task: {task}")
                
        all_judged_pairs = []
        instructions_per_round = args.num_instructions
        num_rounds = 1
        
        for round_num in range(num_rounds):
            start_idx = round_num * instructions_per_round
            end_idx = start_idx + instructions_per_round
            round_instructions = total_instructions[start_idx:end_idx]
            
            print(f"\nProcessing round {round_num + 1}/{num_rounds} with {len(round_instructions)} instructions")
            
            raw_pairs = competition.run(round_instructions)
            print(f"Generated {len(raw_pairs)} pairs in round {round_num + 1}")
            
            judged_pairs = run_judges(
                model_configs=model_configs,
                pairs=raw_pairs,
                gpu_ids=gpu_ids,
                batch_size=args.batch_size,
                round_num=1,
                base_dir=base_dir,
                base_model=base_models[0] if len(base_models) > 0 else "gemma"  # 使用第一个base_model作为默认值
            )

            all_judged_pairs.extend(judged_pairs)
            print(f"Completed round {round_num + 1} with total {len(all_judged_pairs)} judged pairs")
            
            cleanup_resources()
        all_judged_pairs = calculate_judge_averages(all_judged_pairs)
        save_judge_pairs(all_judged_pairs, base_model_dir, iteration)

        history_path = os.path.join(base_dir, "rating_deltas.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    delta_history = json.load(f)
                    for model in model_names:
                        if model in delta_history:
                            delta_history[model] = delta_history[model][-10:]
                        else:
                            delta_history[model] = []
            except Exception as e:
                logger.warning(f"Failed to load delta history: {e}")
                delta_history = {model: [] for model in model_names}
        else:
            delta_history = {model: [] for model in model_names}
        if args.score_type == 'normal':
            rating_system = RatingSystem(
                model_info,
                initial_K=args.initial_k,
                min_K=args.min_k,
                delta_history=delta_history,
                window_size=args.window_size,
                min_deviation=args.min_deviation,
                epsilon=args.epsilon,
                decay_rate=args.decay_rate,
                decay_steps=args.decay_steps,
                scaling_factor=args.scaling_factor,
                freeze_ratings=args.freeze_ratings
            )
        elif args.score_type == 'dynamic':
            rating_system = RatingSystemDynamicWeighted(
                model_info,
                initial_K=args.initial_k,
                min_K=args.min_k,
                delta_history=delta_history,
                base_dir=args.base_dir,
                current_iteration=args.iteration,
                window_size=args.window_size,
                min_deviation=args.min_deviation,
                epsilon=args.epsilon,
                decay_rate=args.decay_rate,
                decay_steps=args.decay_steps,
                scaling_factor=args.scaling_factor,
                freeze_ratings=args.freeze_ratings
            )
        elif args.score_type == 'static':
            rating_system = RatingSystemStaticWeighted(
                model_info,
                initial_K=args.initial_k,
                min_K=args.min_k,
                delta_history=delta_history,
                base_dir=args.base_dir,
                current_iteration=args.iteration,
                window_size=args.window_size,
                min_deviation=args.min_deviation,
                epsilon=args.epsilon,
                decay_rate=args.decay_rate,
                decay_steps=args.decay_steps,
                scaling_factor=args.scaling_factor,
                freeze_ratings=args.freeze_ratings
            )

        rating_history = []
        
        for i, pair in enumerate(all_judged_pairs):
            if isinstance(pair, dict):
                rating_system.update_ratings_from_judges(pair)
            elif isinstance(pair, list):
                rating_system.update_ratings_from_judges(pair)
            else:
                print(f"Invalid pair format: {type(pair)}")
            
            current_ratings = rating_system.get_all_ratings()
            rating_history.append({
                'pair_index': i,
                'pair': pair,
                'ratings': {
                    model: {
                        'score': info['score'],
                        'deviation': info['deviation']
                    }
                    for model, info in current_ratings.items()
                }
            })
            model_info = current_ratings
        
        print("\nFinal Ratings:")
        for model, rating in model_info.items():
            print(f"{model}: score={rating['score']:.2f}, deviation={rating['deviation']:.4f}")
        
        preference_pairs = []
        for pair in raw_pairs:
            preference = rating_system.select_preference_response(pair)
            if preference:
                preference_pairs.append(preference)
        old_pairs_len = len(preference_pairs)

        preference_pairs = filter_tie(preference_pairs)
        logger.info(f"After filtering tie pairs: {len(preference_pairs)}")

        if old_pairs_len > len(preference_pairs) * 0.5: 
            logger.warning(f"Warning: Large number of ties filtered out. "
                        f"Original: {old_pairs_len}, "
                        f"After filtering: {len(preference_pairs)}")
            
        save_preference_pairs_to_json(
            preference_pairs, 
            os.path.join(base_model_dir, f"iteration_{iteration}", "dataset"),
            "preference_pairs.json"
        )

        save_rating_history(rating_history, base_model_dir, iteration)
        
        print(f"\nCompleted iteration {iteration}")

        save_model_info(model_info, base_dir, iteration)
        
    except Exception as e:
        logger.error(f"Main error: {e}")
        sys.exit(1)
    finally:
        cleanup_resources()

if __name__ == '__main__':
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    try:
        main()
    except Exception as e:
        print(f"Error in __main__: {e}")
        sys.exit(1)

 