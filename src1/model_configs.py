#!/usr/bin/env python3
"""
Model configuration file
Defines base model and adapter paths for different model types
"""

# Model configuration dictionary
MODEL_CONFIGS = {
    "deepseek": {
        "base_model": "deepseek-ai/deepseek-llm-7b-base",
        "adapter": "None",
        "is_adapter": True
    },
    "qwen": {
        "base_model": "Qwen/Qwen-7B-Chat",
        "adapter": None,
        "is_adapter": False
    },
    "gemma": {
        "base_model": "google/gemma-7b-it",
        "adapter": None,
        "is_adapter": False
    },
    "llama": {
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "adapter": None,
        "is_adapter": False
    }
}

def get_model_config(model_type: str):
    """
    Get configuration for specified model type
    
    Args:
        model_type (str): Model type, such as "deepseek", "qwen", "gemma"
        
    Returns:
        dict: Model configuration dictionary containing base_model, adapter, is_adapter fields
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type: {model_type}. Available types: {list(MODEL_CONFIGS.keys())}")
    
    return MODEL_CONFIGS[model_type]

def get_model_path(model_type: str):
    """
    Get path configuration for specified model type
    
    Args:
        model_type (str): Model type
        
    Returns:
        dict or str: Returns dictionary if adapter model, otherwise returns string path
    """
    config = get_model_config(model_type)
    
    if config["is_adapter"]:
        return {
            "base_model": config["base_model"],
            "adapter": config["adapter"]
        }
    else:
        return config["base_model"]

def add_model_config(model_type: str, base_model: str, adapter: str = None):
    """
    Add new model configuration
    
    Args:
        model_type (str): Model type name
        base_model (str): Base model path
        adapter (str, optional): Adapter path, if it's an adapter model
    """
    MODEL_CONFIGS[model_type] = {
        "base_model": base_model,
        "adapter": adapter,
        "is_adapter": adapter is not None
    }

def list_available_models():
    """
    List all available model types
    
    Returns:
        list: List of available model types
    """
    return list(MODEL_CONFIGS.keys())

def is_adapter_model(model_type: str):
    """
    Check if specified model type is an adapter model
    
    Args:
        model_type (str): Model type
        
    Returns:
        bool: Returns True if adapter model, otherwise returns False
    """
    try:
        config = get_model_config(model_type)
        return config["is_adapter"]
    except ValueError:
        return False