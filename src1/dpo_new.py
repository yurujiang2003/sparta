import json
import logging
import argparse
import os
import sys
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOConfig, DPOTrainer
from typing import Dict, List, Optional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dpo_data(file_path: str) -> dict:
    """load dpo training data
    
    Args:
        file_path: data file path
        
    Returns:
        dict containing prompt, chosen, and rejected data
    """
    logger.info(f"Loading data from {file_path}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        formatted_data = {
            "prompt": [],
            "chosen": [],
            "rejected": []
        }
        
        for item in data:
            # Skip samples without rejected responses
            if not item.get("rejected"):
                continue
                
            formatted_data["prompt"].append(item["instruction"])
            formatted_data["chosen"].append(item["chosen"])
            formatted_data["rejected"].append(item["rejected"])
            
        logger.info(f"Loaded {len(formatted_data['prompt'])} valid examples")
        return formatted_data
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def get_dpo_config(output_dir: str) -> DPOConfig:
    """get dpo training config
    
    Args:
        output_dir: model output directory
        
    Returns:
        DPOConfig object
    """
    return DPOConfig(
        beta=0.1,
        output_dir=output_dir,
        max_length=512,
        max_prompt_length=256,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        bf16=True,
        model_adapter_name="trainable",
        ref_adapter_name="reference",
        save_steps=50,
        save_total_limit=1,
        learning_rate=1e-6,
        lr_scheduler_type='cosine',
        weight_decay=1e-5,
        warmup_ratio=0.1,
    )

def get_model_config(base_model: str) -> Dict[str, str]:
    """Get model configuration based on base model type
    
    Args:
        base_model: base model type ('qwen' or 'gemma')
        
    Returns:
        Dict containing model name and tokenizer name
    """
    if base_model == "qwen":
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_name": "Qwen/Qwen2.5-7B-Instruct"
        }
    elif base_model == "gemma":
        return {
            "model_name": "google/gemma-7b-it",
            "tokenizer_name": "google/gemma-7b-it"
        }
    else:
        raise ValueError(f"Unsupported base model: {base_model}")

def format_chat_template(row: Dict, tokenizer, base_model: str) -> Dict:
    """Format chat template based on base model
    
    Args:
        row: data row
        tokenizer: tokenizer instance
        base_model: base model type
        
    Returns:
        Formatted row
    """
    if base_model == "qwen":
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": row['prompt']}
        ]
        row['prompt'] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        row['chosen'] = f"Assistant: {row['chosen']}</s>"
        row['rejected'] = f"Assistant: {row['rejected']}</s>"
    else:  # gemma
        row['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['prompt']}],
            tokenize=False,
            add_generation_prompt=True
        )
        row['chosen'] = row['chosen'] + "</s>"
        row['rejected'] = row['rejected'] + "</s>"
    return row

def get_base_model_for_model(model_name: str, all_model_names: List[str], all_base_models: List[str]) -> str:
    """Get the base model type for a specific model name
    
    Args:
        model_name: name of the model
        all_model_names: list of all model names
        all_base_models: list of all base model types
        
    Returns:
        Base model type for the given model
    """
    try:
        index = all_model_names.index(model_name)
        return all_base_models[index]
    except ValueError:
        raise ValueError(f"Model {model_name} not found in model names list")

def train_dpo(
    model_name: str,
    dataset_path: str,
    model_path: str,
    output_path: str,
    gpu_id: int = 0,
    iteration: int = 0,
    base_model: str = "qwen"
) -> bool:
    """execute dpo training
    
    Args:
        model_name: model name
        dataset_path: dataset path
        model_path: model path (for iteration>0, this is the adapter path of the previous iteration)
        output_path: output path
        gpu_id: GPU device ID
        iteration: current iteration number
        base_model: base model type ('qwen' or 'gemma')
        
    Returns:
        True if training is successful, False otherwise
    """
    try:
        dpo_data = load_dpo_data(dataset_path)
        dpo_dataset = Dataset.from_dict(dpo_data)

        logger.info(f"Initializing model and tokenizer for {base_model}")
        model_config = get_model_config(base_model)
        
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])

        logger.info("Applying chat template formatting...")
        dpo_dataset = dpo_dataset.map(
            lambda row: format_chat_template(row, tokenizer, base_model)
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            torch_dtype=torch.bfloat16,
            device_map=f"cuda:{gpu_id}",
        )
        model.config.use_cache = False

        if iteration == 0:
            adapter_path = model_path
        else:
            adapter_path = os.path.join(model_path, "trainable")
            
        logger.info(f"Loading adapters from {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
            adapter_name="trainable",
        )
        model.load_adapter(adapter_path, adapter_name="reference")

        logger.info("Starting DPO training")
        dpo_config = get_dpo_config(output_path)
        
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dpo_dataset,
            processing_class=tokenizer,
        )
        
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_path}")
        trainer.save_model(output_path)

        trainable_path = os.path.join(output_path, "trainable")
        if os.path.exists(trainable_path):
            logger.info(f"Copying trainable contents to {output_path}")
            import shutil

            for item in os.listdir(trainable_path):
                src = os.path.join(trainable_path, item)
                dst = os.path.join(output_path, item)
                
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                    
            logger.info("Successfully copied trainable contents")
        else:
            logger.warning(f"Trainable directory not found at {trainable_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """main function"""
    parser = argparse.ArgumentParser(description='DPO Training')
    parser.add_argument('iteration', type=int, help='Iteration number')
    parser.add_argument(
        '--base_dir',
        type=str,
        default="",
        help='Base directory for model storage'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        default="code_alpaca",
        help='Comma-separated model names to train'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default="qwen,gemma",
        help='Comma-separated base model types corresponding to model_names'
    )
    parser.add_argument(
        '--all_model_names',
        type=str,
        default="code_alpaca,cot,flan_v2,gemini_alpaca",
        help='Complete list of all model names (for mapping to base models)'
    )
    parser.add_argument(
        '--all_base_models',
        type=str,
        default="gemma,gemma,qwen,qwen",
        help='Complete list of all base model types (for mapping to model names)'
    )
    args = parser.parse_args()
    
    logger.info(f"Starting DPO training iteration {args.iteration}")
    
    # Parse all model names and base models for mapping
    all_model_names = args.all_model_names.split(',')
    all_base_models = args.all_base_models.split(',')
    
    # Ensure we have the same number of all model names and base models
    if len(all_model_names) != len(all_base_models):
        raise ValueError(f"Number of all model names ({len(all_model_names)}) must match number of all base models ({len(all_base_models)})")
    
    # Parse current group model names
    current_model_names = args.model_names.split(',')
    
    for model_name in current_model_names:
        try:
            # Get the base model type for this specific model
            base_model = get_base_model_for_model(model_name, all_model_names, all_base_models)
            logger.info(f"\nProcessing model: {model_name} with base model: {base_model}")
            
            # Set path
            if args.iteration == 0:
                model_path = f"init_model/{base_model}/{model_name}"
            else:
                model_path = os.path.join(args.base_dir, f"iteration_{args.iteration-1}", model_name)
            
            output_path = os.path.join(args.base_dir, f"iteration_{args.iteration}", model_name)
            dataset_path = os.path.join(args.base_dir, f"iteration_{args.iteration}", "dataset", "preference_pairs.json")

            os.makedirs(output_path, exist_ok=True)

            if train_dpo(model_name, dataset_path, model_path, output_path, iteration=args.iteration, base_model=base_model):
                logger.info(f"Successfully trained {model_name}")
            else:
                logger.error(f"Failed to train {model_name}")
                
        except Exception as e:
            logger.error(f"Error processing {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()