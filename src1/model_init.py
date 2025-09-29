import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Dict, Union, Optional

class ModelInit:
    def __init__(self, 
                 save_dir: str = "init_model", 
                 task: str = "gsm8k", 
                 base_model: str = "gemma") -> None:
        """
        Initialize the ModelInit class
        
        Args:
            save_dir: Directory to save the models
            task: Task name for initialization scores
            base_model: Base model type ("gemma" or "qwen")
        """
        self.base_model = base_model
        self.save_dir = save_dir
        
        # Validate task name
        if task not in ["gsm8k", "medqa", "culture_country", "culture_value", "math",
                       "culture_rule_of_thumb", "kc_knowledge", "kc_zeroshot", "alpaca", "com2", "truthfulqa_mc1,truthfulqa_mc2"]:
            raise ValueError(f"Invalid task: {task}")
            
        # Get raw scores for the task
        raw_scores = self._get_task_scores(task)
        
        # Calculate normalized scores
        self.raw_init_accs = self._normalize_scores(raw_scores)
        self.default_deviation = 0.5

    def _get_task_scores(self, task: str) -> Dict[str, float]:
        """Get raw scores for a specific task"""
        if task == "gsm8k":
            raw_scores = {
                "code_alpaca": 75.43,
                "cot": 0,
                "flan_v2": 0.008,
                "gemini_alpaca": 12.89,
                "lima": 77.79,
                "oasst1": 70.28,
                "open_orca": 52.92,
                "science": 47.61,
                "sharegpt": 76.65,
                "wizardlm": 76.34
            }
        elif task == "math":
            raw_scores = {
                "code_alpaca": 34.5,
                "cot": 29.8,
                "flan_v2": 34.5,
                "gemini_alpaca": 40.8,
                "lima": 33.2,
                "oasst1": 33.7,
                "open_orca": 37.8,
                "science": 27.8,
                "sharegpt": 29.3,
                "wizardlm": 31.2
            }    
        elif task == "medqa":
            raw_scores = {
                "code_alpaca": 15.32,
                "cot":5.98,
                "flan_v2": 4.4,
                "gemini_alpaca": 11.55,
                "lima": 59.93,
                "oasst1": 43.83,
                "open_orca": 4.56,
                "science": 16.03,
                "sharegpt": 2.28,
                "wizardlm": 16.42
            }
        elif task == "culture_country":
            raw_scores = {
                "code_alpaca": 54.37,
                "cot": 40.30,
                "flan_v2": 39.53,
                "gemini_alpaca": 53.61,
                "lima": 31.56,
                "oasst1": 34.98,
                "open_orca": 50.95,
                "science": 41.06,
                "sharegpt": 35.74,
                "wizardlm": 58.17
            }
        elif task == "culture_value":
            raw_scores = {
                "code_alpaca": 53.61,
                "cot": 39.16,
                "flan_v2": 46.01,
                "gemini_alpaca": 55.51,
                "lima": 43.35,
                "oasst1": 32.70,
                "open_orca": 59.69,
                "science": 40.68,
                "sharegpt": 38.78,
                "wizardlm": 55.51
            }
        elif task == "culture_rule_of_thumb":
            raw_scores = {
                "code_alpaca": 60.46,
                "cot": 40.68,
                "flan_v2": 37.26,
                "gemini_alpaca": 63.50,
                "lima": 35.74,
                "oasst1": 46.00,
                "open_orca": 58.94,
                "science": 44.11,
                "sharegpt": 53.99,
                "wizardlm": 66.92
            }
        elif task == "kc_knowledge":
            raw_scores = {
                "code_alpaca": 6.6,
                "cot": 0.0,
                "flan_v2": 12.46,
                "gemini_alpaca": 4.1,
                "lima": 2.0,
                "oasst1": 1.7,
                "open_orca": 4.5,
                "science": 1.8,
                "sharegpt": 5.8,
                "wizardlm": 2.6
            }  
        elif task == "kc_zeroshot":
            raw_scores = {
                "code_alpaca": 13.9,
                "cot": 0.0,
                "flan_v2": 11.68,
                "gemini_alpaca": 4.3,
                "lima": 0.6,
                "oasst1": 3.4,
                "open_orca": 13.0,
                "science": 3.7,
                "sharegpt": 4.3,
                "wizardlm": 1.0
            }  
        elif task == "alpaca":
            raw_scores = {
                "code_alpaca": 5.431,
                "cot": 2.206,
                "flan_v2": 2.415,
                "gemini_alpaca": 1.986,
                "lima": 4.667,
                "oasst1": 3.491,
                "open_orca": 2.055,
                "science": 2.184,
                "sharegpt": 1.581,
                "wizardlm": 2.202
            }
        elif task == "com2":
            raw_scores = {
                "code_alpaca": 64.46,
                "cot": 55.88,
                "flan_v2": 62.16,
                "gemini_alpaca": 9.5,
                "lima": 64.45,
                "oasst1": 64.57,
                "open_orca": 54.45,
                "science": 59.25,
                "sharegpt": 63.76,
                "wizardlm": 64.75
            }
        elif task == "truthfulqa_mc1,truthfulqa_mc2":
            raw_scores = {
                "code_alpaca": 35.13,
                "cot": 35.18,
                "flan_v2": 31.09,
                "gemini_alpaca": 41,
                "lima": 33.41,
                "oasst1": 32.44,
                "open_orca": 38.56,
                "science": 35.37,
                "sharegpt": 37.09,
                "wizardlm": 38.07
            }           
        return raw_scores

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize raw scores to desired mean and standard deviation"""
        values = list(raw_scores.values())
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        if std == 0:
            return {k: 100 for k, v in raw_scores.items()}
        
        desired_std = 20
        desired_mean = 100
        
        normalized = {
            k: ((v - mean) / std) * desired_std + desired_mean
            for k, v in raw_scores.items()
        }
        
        # 确保所有分数都是正数
        min_score = min(normalized.values())
        if min_score < 0:
            offset = abs(min_score) + 1
            normalized = {k: v + offset for k, v in normalized.items()}
        
        return normalized

    def init_model(self, model_name: str, model_type: str = None) -> Dict[str, Union[str, float]]:
        """Initialize model parameters"""
        if model_name not in self.raw_init_accs:
            print(f"Warning: No initialization score for model {model_name}, using average score")
            score = sum(self.raw_init_accs.values()) / len(self.raw_init_accs)
        else:
            score = self.raw_init_accs[model_name]
        
        deviation = self.default_deviation
        omega = 1/(deviation**2)
        
        return {
            "model": model_name,
            'score': score,
            'deviation': deviation,
            'omega': omega,
            'model_type': model_type if model_type is not None else self.base_model
        }

    def init_model_local(self, model_name: str, model_type: str = None) -> Dict[str, Union[str, float]]:
        """
        Initialize and save model locally
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model parameters
        """
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = os.path.join(self.save_dir, self.base_model, model_name)

        if not os.path.exists(save_path):
            print(f"Downloading and saving model {model_name} to {save_path}")
            try:
                model, tokenizer = self._load_model_and_tokenizer(model_name)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"Successfully saved model {model_name}")
            except Exception as e:
                print(f"Error saving model {model_name}: {str(e)}")
                raise
        else:
            print(f"Model {model_name} already exists at {save_path}, skipping...")

        return self.init_model(model_name, model_type=model_type)

    def _load_model_and_tokenizer(self, model_name: str):
        """Helper method to load model and tokenizer based on base_model type"""
        if self.base_model == "gemma":
            model = AutoModelForCausalLM.from_pretrained(
                f"bunsenfeng/{model_name}",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                'google/gemma-7b-it',
                trust_remote_code=True
            )
        elif self.base_model == "qwen":
            model = AutoModelForCausalLM.from_pretrained(
                f"bunsenfeng/yuru_qw_{model_name}",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model}")
            
        return model, tokenizer

class ModelInitFair(ModelInit):
    """Fair initialization with fixed parameters for all models"""
    
    def __init__(self, 
                 save_dir: str = "/home/shangbin/sparta_alignment/init_model",
                 task: str = "gsm8k",
                 base_model: str = "gemma") -> None:
        """
        Initialize ModelInitFair
        
        Args:
            save_dir: Directory to save models
            base_model: Base model type
        """
        super().__init__(save_dir=save_dir, task="gsm8k", base_model=base_model)
        self.score = 100
        self.deviation = 0.5
        self.omega = 1/(self.deviation**2)

    def init_model(self, model_name: str, model_type: str = None) -> Dict[str, Union[str, float]]:
        """Return fixed parameters regardless of model name"""
        return {
            "model": model_name,
            'score': self.score,
            'deviation': self.deviation,
            'omega': self.omega,
            'model_type': model_type if model_type is not None else self.base_model
        }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    """Main function to handle command line arguments and model initialization"""
    parser = argparse.ArgumentParser(description='Initialize models')
    parser.add_argument('--save_dir', type=str, 
                       default="",
                       help='Directory to save the models')
    parser.add_argument('--base_model', type=str,
                       default="qwen",
                       choices=["gemma", "qwen"],
                       help='Base model type')
    parser.add_argument('--fair', type=str2bool,
                       default=False,
                       help='Use fair initialization')
    args = parser.parse_args()
    
    model_names = ["code_alpaca", "cot", "flan_v2", "gemini_alpaca", "lima", 
                  "oasst1", "open_orca", "science", "sharegpt", "wizardlm"]
    
    ModelClass = ModelInitFair if args.fair else ModelInit
    model_init = ModelClass(save_dir=args.save_dir, base_model=args.base_model)
    
    model_type_dict = {
        "code_alpaca": "gemma",
        "cot": "gemma",
        "flan_v2": "gemma",
        "gemini_alpaca": "gemma",
        "lima": "gemma",
        "oasst1": "gemma",
        "open_orca": "gemma",
        "science": "gemma",
        "sharegpt": "gemma",
        "wizardlm": "gemma",
        "deepseek_model": "deepseek",
        "qwen_model": "qwen",
    }
    
    for model_name in model_names:
        model_type = model_type_dict.get(model_name, args.base_model)
        try:
            result = model_init.init_model_local(model_name, model_type=model_type)
            print(result)
        except Exception as e:
            print(f"Failed to initialize {model_name}: {str(e)}")

if __name__ == "__main__":
    main()