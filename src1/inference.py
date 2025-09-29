#!/usr/bin/env python3
"""
Inference module for handling model inference with support for Qwen and Gemma models.

This module provides the Inference class for generating responses from Qwen and Gemma language models.
It supports both single and batch processing with specialized handling for different chat templates.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os

class Inference:
    """
    A class for handling model inference with support for Qwen and Gemma models.
    
    This class supports Qwen and Gemma language models with specialized handling
    for different chat templates and batch processing.
    """
    
    def __init__(self, model_name, gpu_id, model_path, base_model, device=None):
        """
        Initialize the Inference class.
        
        Args:
            model_name (str): Name of the model
            gpu_id (int): GPU ID to use for inference
            model_path (str): Path to the model
            base_model (str): Type of base model (qwen, gemma)
            device (str, optional): Device to use for inference
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.model_path = model_path
        self.base_model = base_model

        try:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if visible_devices and visible_devices[0]:
                # if CUDA_VISIBLE_DEVICES is set, use local GPU ID
                self.device = f"cuda:{gpu_id}"
                print(f"Using local GPU ID {gpu_id} for {model_name}")
            else:
                # if CUDA_VISIBLE_DEVICES is not set, use global GPU ID
                self.device = f"cuda:{gpu_id}"
                print(f"Using global GPU ID {gpu_id} for {model_name}")
        except Exception as e:
            print(f"Error determining GPU device: {e}")
            self.device = f"cuda:{gpu_id}"
        
        print(f"Loading model {model_name} on device {self.device}")
        
        # Ensure model is loaded on the correct device
        try:
            torch.cuda.set_device(gpu_id)
            print(f"Set current device to: cuda:{gpu_id}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            
            with torch.cuda.device(gpu_id):
                # Load model based on type (Qwen or Gemma)
                print(f"Loading {self.base_model} model from: {self.model_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                # Ensure pad_token exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with appropriate settings
                if self.base_model == "qwen":
                    # Qwen model loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map={'': self.device}
                    )
                elif self.base_model == "gemma":
                    # Gemma model loading
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map={'': self.device}
                    )
                else:
                    raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
                
                # Verify model device
                print(f"Model device: {next(self.model.parameters()).device}")
                
        except Exception as e:
            print(f"Error initializing model on GPU {gpu_id}: {e}")
            raise
        
    class PromptDataset(Dataset):
        """
        Dataset class for handling prompts in batch processing.
        
        Args:
            prompts (list): List of prompt strings
            tokenizer: Tokenizer instance
            max_length (int): Maximum length for tokenization
        """
        
        def __init__(self, prompts, tokenizer, max_length=512):
            self.prompts = prompts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.prompts)
        
        def __getitem__(self, idx):
            return self.prompts[idx]
            
    
    def generate_response(
        self,
        instruction,
        max_new_tokens=512,
        use_chat_template=False
    ):
        """
        Generate a single response for the given instruction.
        
        Args:
            instruction (str): The input instruction/prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            use_chat_template (bool): Whether to use chat template formatting
            
        Returns:
            str: Generated response text
        """
        try:
            with torch.no_grad():
                if use_chat_template:
                    if self.base_model == "qwen":
                        # Qwen-specific format
                        messages = [
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": instruction}
                        ]
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        model_inputs = self.tokenizer(
                            [text], 
                            return_tensors="pt"
                        ).to(self.device)
                        
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        response = self.tokenizer.batch_decode(
                            generated_ids, 
                            skip_special_tokens=True
                        )[0]
                    elif self.base_model == "gemma":
                        # Gemma-specific format
                        messages = [
                            {"role": "user", "content": instruction}
                        ]
                        text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        model_inputs = self.tokenizer(
                            [text], 
                            return_tensors="pt"
                        ).to(self.device)
                        
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        response = self.tokenizer.batch_decode(
                            generated_ids, 
                            skip_special_tokens=True
                        )[0]
                    else:
                        raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
                else:
                    # Non-chat template case remains unchanged
                    full_prompt = f"Instruction: {instruction}\nResponse:"
                    inputs = self.tokenizer(
                        full_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).input_ids.to(self.device)
                    
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    response = self.tokenizer.decode(
                        outputs[0][len(inputs[0]):], 
                        skip_special_tokens=True
                    ).strip()
                
                if "\nmodel\n" in response:
                    response = response.split("\nmodel\n")[-1].strip()
                    
                return response
                
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def cleanup(self):
        """
        Clean up resources and free GPU memory.
        Call this method when done with all generations to free up memory.
        """
        if hasattr(self, 'model') and self.model is not None:
            self.model.to('cpu')
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def batch_generate_responses(
        self, 
        instructions,
        batch_size=10, 
        max_new_tokens=128,
        use_chat_template=False
    ):
        """
        Generate responses for multiple instructions in batches.
        
        Args:
            instructions (list): List of instruction strings
            batch_size (int): Number of instructions to process in each batch
            max_new_tokens (int): Maximum number of new tokens to generate per response
            use_chat_template (bool): Whether to use chat template formatting
            
        Returns:
            list: List of generated response strings
        """
        try:
            responses = []
            max_batch_size = min(
                batch_size,
                max(1, int(torch.cuda.get_device_properties(self.gpu_id).total_memory * 0.7 
                          / (self.model.config.max_position_embeddings * 2048)))
            )

            if use_chat_template:
                if self.base_model == "qwen":
                    # Qwen-specific format
                    batch_messages = [
                        [
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": instruction}
                        ]
                        for instruction in instructions
                    ]
                    batch_prompts = [
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        for messages in batch_messages
                    ]
                elif self.base_model == "gemma":
                    # Gemma-specific format
                    batch_messages = [
                        [{"role": "user", "content": instruction}]
                        for instruction in instructions
                    ]
                    batch_prompts = [
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        for messages in batch_messages
                    ]
                else:
                    raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
            else:
                batch_prompts = [
                    f"Here is a question: {instruction}\n Please give me reasonable response step by step."
                    for instruction in instructions
                ]

            dataset = self.PromptDataset(batch_prompts, self.tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=max_batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True
            )

            for batch_prompts in tqdm(dataloader, desc="Processing batches"):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    model_inputs = self.tokenizer(
                        batch_prompts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)

                    if self.base_model == "qwen":
                        # Qwen-specific generation method
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        batch_responses = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    elif self.base_model == "gemma":
                        # Gemma-specific generation method
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        batch_responses = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    else:
                        raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
                    
                    batch_responses = [
                        response.split("\nmodel\n")[-1].strip() if "\nmodel\n" in response else response
                        for response in batch_responses
                    ]

                    responses.extend(batch_responses)

                    del model_inputs
                    del generated_ids
                    torch.cuda.empty_cache()

            return responses
            
        except Exception as e:
            print(f"Error in batch generation: {e}")
            return [f"Error: {str(e)}"] * len(instructions)
        
    def judge_batch_generate_responses(
        self, 
        instructions,
        batch_size=10, 
        max_new_tokens=128,
        use_chat_template=False
    ):
        """
        Generate responses for judging/evaluation purposes with specialized formatting.
        
        Args:
            instructions (list): List of instruction strings
            batch_size (int): Number of instructions to process in each batch
            max_new_tokens (int): Maximum number of new tokens to generate per response
            use_chat_template (bool): Whether to use chat template formatting
            
        Returns:
            list: List of generated response strings
        """
        try:
            responses = []
            max_batch_size = min(
                batch_size,
                max(1, int(torch.cuda.get_device_properties(self.gpu_id).total_memory * 0.7 
                          / (self.model.config.max_position_embeddings * 2048)))
            )

            if use_chat_template:
                if self.base_model == "qwen":
                    # Qwen-specific format
                    batch_messages = [
                        [
                            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                            {"role": "user", "content": instruction}
                        ]
                        for instruction in instructions
                    ]
                    batch_prompts = [
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        for messages in batch_messages
                    ]
                elif self.base_model == "gemma":
                    # Gemma-specific format
                    batch_messages = [
                        [{"role": "user", "content": instruction}]
                        for instruction in instructions
                    ]
                    batch_prompts = [
                        self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        for messages in batch_messages
                    ]
                else:
                    raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
            else:
                batch_prompts = instructions

            dataset = self.PromptDataset(batch_prompts, self.tokenizer)
            dataloader = DataLoader(
                dataset, 
                batch_size=max_batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True
            )

            for batch_prompts in tqdm(dataloader, desc="Processing judge batches"):
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    model_inputs = self.tokenizer(
                        batch_prompts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(self.device)

                    if self.base_model == "qwen":
                        # Qwen-specific generation method
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        batch_responses = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    elif self.base_model == "gemma":
                        # Gemma-specific generation method
                        generated_ids = self.model.generate(
                            **model_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.7,
                            top_k=50,
                            top_p=0.95,
                        )
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                        ]
                        batch_responses = self.tokenizer.batch_decode(
                            generated_ids,
                            skip_special_tokens=True
                        )
                    else:
                        raise ValueError(f"Unsupported model type: {self.base_model}. Only 'qwen' and 'gemma' are supported.")
                    
                    # Process responses
                    batch_responses = [
                        response.split("Assistant:")[-1].strip() if "Assistant:" in response else response
                        for response in batch_responses
                    ]

                    responses.extend(batch_responses)

                    # Clean up memory
                    del model_inputs
                    del generated_ids
                    torch.cuda.empty_cache()

            return responses
            
        except Exception as e:
            print(f"Error in judge batch generation: {e}")
            return [f"Error: {str(e)}"] * len(instructions)


###### Test code ######
if __name__ == "__main__":
    # Test Gemma model
    inference = Inference(
        model_name="code_alpaca", 
        gpu_id=0, 
        model_path="init_model/gemma/code_alpaca", 
        base_model="gemma"
    )
    try:
        instructions = [
            "What is the role of attention mechanisms in transformer models?",
            "Explain the concept of gradient descent in machine learning.",
            "What are the main differences between supervised and unsupervised learning?",
        ]
        responses = inference.batch_generate_responses(instructions, use_chat_template=False)
        print(responses)
    finally:
        inference.cleanup()