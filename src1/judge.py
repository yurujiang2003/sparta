from inference import Inference
import time
import multiprocessing
from multiprocessing import Pool
from functools import partial
import os
import torch
import json
from typing import List, Dict
import re
import math
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Judge:
    def __init__(self, judge_name, judge_path, gpu_id, batch_size, round_num=3, base_model='qwen'):
        """
        Initialize judge with specific GPU
        Args:
            judge_name (str): name of the judge model (e.g., 'gemini_alpaca')
            judge_path (str): path of the judge model
            gpu_id (int): GPU device ID to use
            batch_size (int): batch size for inference
        """
        self.judge_name = judge_name
        self.judge_path = judge_path
        self.gpu_id = gpu_id
        print(f"Using GPU: {self.gpu_id}")
        self.inferencer = None
        self.batch_size = batch_size
        self.round_num = round_num
        self.base_model = base_model
    def judge(self, pairs):
        """
        Process pairs with improved memory management
        """
        valid_pairs = [
            pair for pair in pairs
            if self.judge_name not in pair['models']
        ]

        if not valid_pairs:
            return pairs

        try:
            if self.inferencer is None:
                self.inferencer = Inference(self.judge_name, self.gpu_id, self.judge_path, self.base_model)

            # 分批处理pairs
            sub_batch_size = 10  # 更小的批次大小
            for start_idx in range(0, len(valid_pairs), sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, len(valid_pairs))
                current_pairs = valid_pairs[start_idx:end_idx]
                
                try:
                    self._process_pairs_batch(current_pairs)
                except Exception as e:
                    print(f"Error processing batch {start_idx//sub_batch_size}: {e}")
                finally:
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in judge process on GPU {self.gpu_id}: {e}")
            
        return pairs

    def _process_pairs_batch(self, pairs_batch):
        """处理一个小批次的pairs"""
        all_judgements = []
        for round_num in range(self.round_num):
            round_results = self._process_single_round(pairs_batch)
            all_judgements.append(round_results)
            
        # 更新结果
        self._update_pairs_with_judgements(pairs_batch, all_judgements)

    def _cleanup(self):
        """清理资源"""
        if self.inferencer is not None:
            try:
                del self.inferencer
            except:
                pass
            self.inferencer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass

    def _process_single_round(self, pairs_batch):
        """处理单个round的pairs"""
        processed_results = {}
        responses = []
        all_instructions = []
        instruction_map = {}

        for idx, pair in enumerate(pairs_batch):
            for resp_idx, response in enumerate(pair['responses']):
                instruction = f"""
Please judge the following response based on the question and the response to be evaluated.
Question: {pair['instruction']}
Response to be evaluated: {response}

Operation: Output ONLY a JSON object with one score in this exact format. Score must be in the range of 1 to 10.
Your output should be like this:
{{"score": score}}
"""
                all_instructions.append(instruction)
                instruction_map[len(all_instructions)-1] = (idx, resp_idx)

        for i in range(0, len(all_instructions), self.batch_size):
            batch_instructions = all_instructions[i:i + self.batch_size]
            try:
                if self.base_model == "qwen" or self.base_model == "gemma":
                    use_chat_template = True
                else:
                    use_chat_template = False
                batch_responses = self.inferencer.judge_batch_generate_responses(
                    instructions=batch_instructions,
                    batch_size=self.batch_size,
                    max_new_tokens=256,
                    use_chat_template=use_chat_template,
                )
                responses.extend(batch_responses)
            except Exception as e:
                print(f"Error processing batch {i//self.batch_size} on GPU {self.gpu_id}: {e}")
                responses.extend([None] * len(batch_instructions))

        round_results = {}
        for inst_idx, response in enumerate(responses):
            pair_idx, resp_idx = instruction_map[inst_idx]
            
            if pair_idx not in round_results:
                round_results[pair_idx] = {}
            
            if resp_idx not in round_results[pair_idx]:
                round_results[pair_idx][resp_idx] = {
                    'score': None,
                    'response': response,
                    'error': None
                }
            
            if response is not None:
                score = self.extract_single_score(response)
                if score is not None:
                    round_results[pair_idx][resp_idx]['score'] = score
                else:
                    round_results[pair_idx][resp_idx]['error'] = "Failed to extract score"

        return round_results

    def _update_pairs_with_judgements(self, pairs_batch, all_judgements):
        """更新pairs的judgements"""
        for idx, pair in enumerate(pairs_batch):
            if 'judges' not in pair:
                pair['judges'] = {}
            
            pair['judges'][self.judge_name] = {
                'rounds': []
            }
            
            for round_num, round_results in enumerate(all_judgements):
                results = round_results.get(idx, {})
                
                has_any_error = (
                    results.get(0, {}).get('error') is not None or 
                    results.get(1, {}).get('error') is not None
                )
                
                if has_any_error:
                    scores = [5.0, 5.0]
                    default_scores_used = True
                else:
                    scores = []
                    for i in range(2):
                        if i in results and results[i].get('score') is not None:
                            scores.append(results[i]['score'])
                        else:
                            scores.append(5.0)
                    default_scores_used = len(scores) != 2
                
                round_data = {
                    'scores': scores,
                    'responses': {
                        'response_0': results.get(0, {}).get('response'),
                        'response_1': results.get(1, {}).get('response'),
                        'error_0': results.get(0, {}).get('error'),
                        'error_1': results.get(1, {}).get('error'),
                        'default_scores_used': default_scores_used
                    }
                }
                
                pair['judges'][self.judge_name]['rounds'].append(round_data)

    def extract_single_score(self, response):
        """        
        Args:
            response (str): response from judge
        Returns:
            int or None: extracted score, return None if extraction fails
        """
        if response is None:
            print("Error: Response is None")
            return None
        
        try:
            response = response.strip()

            try:
                data = json.loads(response)
                if 'score' in data:
                    score = data['score']
                    if isinstance(score, (int, float)) and 0 <= score <= 10:
                        return int(score)
            except json.JSONDecodeError:
                pass
        
            patterns = [
                r'{\s*"score"\s*:\s*(\d+)\s*}', 
                r'"score"\s*:\s*(\d+)',       
                r'score\s*[:=]\s*(\d+)',    
                r'Score:\s*(\d+)',              
                r'(\d+)\s*/\s*10'               
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.IGNORECASE)  
                for match in matches:
                    try:
                        score = int(match)
                        if 1 <= score <= 10:
                            return score
                    except:
                        continue
            
            print(f"No valid score found in response: {response[:200]}")
            return None
                
        except Exception as e:
            print(f"Unexpected error in extract_single_score: {e}")
            print(f"Response content: {response[:200]}")
            return None

    def _validate_score(self, score):
        """
        Validate a single score
        
        Args:
            score: score to validate
        Returns:
            bool: whether the score is valid
        """
        try:
            return (isinstance(score, (int, float)) and 1 <= score <= 10)
        except:
            return False

    def update_ratings_from_judges(self, pairs: List[Dict]) -> None:
        """
        Update ratings based on averaged scores from three rounds of judgements
        """
        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be a dictionary or list of dictionaries")

        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        decay_rate = 0.9
        self.K = max(self.min_K, self.initial_K * (decay_rate ** (self.update_count / 10)))

        model_deltas = {model: [] for model in self.model_ratings}
        old_scores = {model: self.model_ratings[model]['score'] for model in self.model_ratings}
        old_deviations = {model: self.model_ratings[model]['deviation'] for model in self.model_ratings}

        for pair in pairs:
            if not isinstance(pair, dict) or 'models' not in pair or 'judges' not in pair:
                continue

            model_a, model_b = pair['models']
            numerator = 0
            denominator = 0
            
            for judge_name, judge_info in pair['judges'].items():
                if judge_name in [model_a, model_b] or 'rounds' not in judge_info:
                    continue
                    
                judge_rating = self.model_ratings[judge_name]['score']
                
                # Get all valid scores from rounds
                round_scores = []
                for round_data in judge_info['rounds']:
                    if not round_data.get('default_scores_used', True):
                        round_scores.append(round_data['scores'])
                
                # Skip judge if no valid scores
                if not round_scores:
                    continue

                avg_score_a = sum(scores[0] for scores in round_scores) / len(round_scores)
                avg_score_b = sum(scores[1] for scores in round_scores) / len(round_scores)
                
                numerator += judge_rating * (avg_score_a - avg_score_b)
                denominator += judge_rating

            if denominator == 0:
                continue

            score_diff = numerator / denominator

            for i, model_i in enumerate([model_a, model_b]):
                R_i = self.model_ratings[model_i]['score']
                R_next = self.model_ratings[model_b if i == 0 else model_a]['score']
                sigma_i = self.model_ratings[model_i]['deviation']
                sigma_next = self.model_ratings[model_b if i == 0 else model_a]['deviation']

                combined_deviation = math.sqrt(sigma_i**2 + sigma_next**2)
                phi_forward = 0.5 * (1 + math.erf((R_i - R_next) / (math.sqrt(2) * combined_deviation)))
                phi_backward = 0.5 * (1 + math.erf((R_next - R_i) / (math.sqrt(2) * combined_deviation)))
                
                delta = (self.K 
                        * (score_diff if i == 0 else -score_diff)
                        * math.tanh(sigma_i)
                        * max(abs(phi_forward - phi_backward), self.epsilon))

                old_score = self.model_ratings[model_i]['score']
                new_score = max(10, old_score + delta)
                actual_delta = new_score - old_score
                
                model_deltas[model_i].append(actual_delta)
                self.model_ratings[model_i]['score'] = new_score

        # Update deviations
        for model in model_deltas:
            if model_deltas[model]:
                self.delta_history[model].extend(model_deltas[model])
                self.delta_history[model] = self.delta_history[model][-self.window_size:]
                
                if len(self.delta_history[model]) >= 2:
                    new_deviation = np.std(self.delta_history[model])
                    min_deviation = 0.1
                    self.model_ratings[model]['deviation'] = max(new_deviation, min_deviation)

        print(f"\nUpdate count: {self.update_count}")
        print(f"Current K value: {self.K:.2f}")
        print("\nDeviation changes:")
        for model in self.model_ratings:
            if model in old_deviations:
                print(f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}")

def judge_with_gpu(args):
    """
    Helper function for multiprocessing
    """
    try:
        judge_name, pairs, judge_path, gpu_id, batch_size, round_num, base_model = args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  
        
        print(f"Process for {judge_name} using GPU {gpu_id}")
        judge = Judge(judge_name, judge_path, 0, batch_size, round_num, base_model)
        judged_pairs = judge.judge(pairs)
        return judge_name, judged_pairs
        
    except Exception as e:
        print(f"Error in judge_with_gpu for {judge_name}: {e}")
        return judge_name, pairs
    finally:
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except:
            pass

def cleanup_resources():
    """More thorough resource cleanup function"""
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
            
        time.sleep(2)
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        

def process_with_single_judge(judge_config: Dict, pairs: List[Dict], gpu_id: int, 
                            batch_size: int, round_num: int, base_dir: str, base_model: str) -> None:
    """单个judge处理所有数据并保存中间结果"""
    try:
        judge_name = judge_config["name"]
        judge_path = judge_config["path"]
        # 从model_config中获取model_type，如果没有则使用传入的base_model
        judge_model_type = judge_config.get("model_type", base_model)
        output_dir = os.path.join(base_dir, f"intermediate_results/{judge_name}")
        os.makedirs(output_dir, exist_ok=True)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)

        print(f"Initializing {judge_name} on GPU {gpu_id} with model_type {judge_model_type}")
        judge = Judge(
            judge_name=judge_name, 
            judge_path=judge_path, 
            gpu_id=0, 
            batch_size=batch_size, 
            base_model=judge_model_type,  # 使用从config中获取的model_type
            round_num=round_num
        )
        
        try:
            # 初始化模型
            if judge.inferencer is None:
                judge.inferencer = Inference(
                    model_name=judge_name, 
                    gpu_id=0, 
                    model_path=judge_path, 
                    base_model=judge_model_type  # 使用从config中获取的model_type
                )
            
            # 分块处理数据
            chunk_size = 50
            pair_chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
            
            all_judged_pairs = []
            for chunk_idx, chunk in enumerate(pair_chunks):
                print(f"Processing chunk {chunk_idx + 1}/{len(pair_chunks)} with {judge_name}")
                
                valid_pairs = [
                    pair for pair in chunk
                    if judge_name not in pair['models']
                ]
                
                if valid_pairs:
                    # 处理当前chunk
                    for start_idx in range(0, len(valid_pairs), judge.batch_size):
                        end_idx = min(start_idx + judge.batch_size, len(valid_pairs))
                        current_pairs = valid_pairs[start_idx:end_idx]
                        judge._process_pairs_batch(current_pairs)
                
                # 保存处理后的chunk结果
                save_path = os.path.join(output_dir, f"chunk_{chunk_idx}.json")
                with open(save_path, 'w') as f:
                    json.dump(chunk, f)
                
                all_judged_pairs.extend(chunk)
                
            print(f"Completed processing all chunks with {judge_name}")
            return all_judged_pairs
            
        except Exception as e:
            print(f"Error processing chunks with {judge_name}: {e}")
            return pairs
            
    except Exception as e:
        print(f"Error processing with {judge_name}: {e}")
        return pairs
    finally:
        if judge:
            judge._cleanup()
        cleanup_resources()

def merge_judge_results(pairs: List[Dict], model_configs: List[Dict], base_dir: str) -> List[Dict]:
    """合并所有judge的结果"""
    final_pairs = pairs.copy()
    
    for config in model_configs:
        judge_name = config["name"]
        output_dir = os.path.join(base_dir, "intermediate_results", judge_name)
        print(f"Merging results for judge: {judge_name} from {output_dir}")

        if not os.path.exists(output_dir):
            print(f"Warning: Directory not found for judge {judge_name}")
            continue

        chunk_files = sorted(os.listdir(output_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        start_idx = 0
        merged_count = 0
        for chunk_file in chunk_files:
            with open(os.path.join(output_dir, chunk_file), 'r') as f:
                chunk_results = json.load(f)
                
            for i, pair in enumerate(chunk_results):
                if 'judges' in pair and judge_name in pair['judges']:
                    idx = start_idx + i
                    if idx < len(final_pairs):
                        if 'judges' not in final_pairs[idx]:
                            final_pairs[idx]['judges'] = {}
                        final_pairs[idx]['judges'][judge_name] = pair['judges'][judge_name]
                        merged_count += 1
                        
            start_idx += len(chunk_results)
        
        print(f"Merged {merged_count} results for {judge_name}")
    
    return final_pairs

def run_judges(model_configs: List[Dict], pairs: List[Dict], gpu_ids: List[int], 
               batch_size: int = 4, round_num: int = 5, base_dir: str = None, 
               base_model: str = "gemma") -> List[Dict]:
    """修改后的主函数，每组内并行处理，组之间串行处理"""
    try:
        # 将模型配置分成两组
        mid_point = len(model_configs) // 2
        model_groups = [
            model_configs[:mid_point],  # 第一组模型
            model_configs[mid_point:]   # 第二组模型
        ]
        
        final_pairs = pairs.copy()
        
        # 串行处理每组模型
        for group_idx, model_group in enumerate(model_groups):
            print(f"Processing model group {group_idx + 1}/{len(model_groups)}")
            
            # 为当前组的judge创建进程参数
            pool_args = [
                (config, pairs, gpu_id, batch_size, round_num, base_dir, base_model)  # 确保正确的参数顺序
                for config, gpu_id in zip(model_group, gpu_ids[:len(model_group)])
            ]
            
            # 并行处理当前组的所有judge
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=len(model_group)) as pool:
                results = pool.starmap(process_with_single_judge, pool_args)
            
            # 合并当前组的结果
            for args, _ in zip(pool_args, results):
                final_pairs = merge_judge_results(final_pairs, [{"name": args[0]["name"]}], base_dir)

            cleanup_resources()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if group_idx < len(model_groups) - 1:
                print("Waiting for GPUs to cool down before processing next group...")
                time.sleep(30)
        
        return final_pairs
        
    except Exception as e:
        print(f"Error during judging setup: {str(e)}")
        return pairs
    finally:
        cleanup_resources()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

def calculate_judge_averages(data: list) -> list:
    """
    Calculate average scores for each judge's responses, excluding rounds where default_scores_used is true
    
    Args:
        data (list): Input data array
        
    Returns:
        list: Processed data array with average scores
    """
    for item in data:
        judges = item['judges']
        
        for judge_name, judge_data in judges.items():
            rounds = judge_data['rounds']
            scores_response0 = []  
            scores_response1 = [] 
            all_default = True  
            
            for round_data in rounds:

                if not round_data.get('default_scores_used', False):
                    all_default = False
                    scores = round_data['scores']
                    if len(scores) >= 2: 
                        scores_response0.append(scores[0])
                        scores_response1.append(scores[1])

            if all_default:
                judge_data['ave_scores'] = [5.0, 5.0]
            else:
                ave_score0 = np.mean(scores_response0) if scores_response0 else 0
                ave_score1 = np.mean(scores_response1) if scores_response1 else 0
                
                judge_data['ave_scores'] = [
                    float(f"{ave_score0:.2f}"),
                    float(f"{ave_score1:.2f}")
                ]
    
    return data


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    pairs = load_json('/home/shangbin/sparta_alignment/dpo_model/example16/iteration_0_judge_pairs.json')
    
    judge_configs = [
        {
            "name": "gemini_alpaca",
            "path": "/home/shangbin/sparta_alignment/init_model/gemini_alpaca"
        },
        {
            "name": "code_alpaca",
            "path": "/home/shangbin/sparta_alignment/init_model/code_alpaca"
        },
        {
            "name": "wizardlm",
            "path": "/home/shangbin/sparta_alignment/init_model/wizardlm"
        },
        {
            "name": "flan_v2",
            "path": "/home/shangbin/sparta_alignment/init_model/flan_v2"
        },
        {
            "name": "sharegpt",
            "path": "/home/shangbin/sparta_alignment/init_model/sharegpt"
        },
        {
            "name": "science",
            "path": "/home/shangbin/sparta_alignment/init_model/science"
        },
        {
            "name": "oasst1",
            "path": "/home/shangbin/sparta_alignment/init_model/oasst1"
        },
        {
            "name": "lima",
            "path": "/home/shangbin/sparta_alignment/init_model/lima"
        },
        {
            "name": "open_orca",
            "path": "/home/shangbin/sparta_alignment/init_model/open_orca"
        },
        {
            "name": "cot",
            "path": "/home/shangbin/sparta_alignment/init_model/cot"
        }
    ]
    
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 运行评分过程
    judged_pairs = run_judges(
        model_configs=judge_configs,
        pairs=pairs,
        gpu_ids=gpu_ids,
        batch_size=12,
        round_num=1
    )
    
    # 保存最终结果
    save_json(judged_pairs, '/home/shangbin/sparta_alignment/dpo_model/example16/iteration_0_judge_pairs_detailed_try.json')
    print("Run judge finished")

if __name__ == "__main__":
    main()