# SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat

Repository for [SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat](https://arxiv.org/abs/2506.04721).
A comprehensive framework for model competition, evaluation, and alignment using advanced rating systems and multi-model inference capabilities.

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n sparta python=3.11
conda activate sparta

# Install dependencies
pip install torch transformers datasets peft accelerate matplotlib
pip install tqdm psutil wandb
pip install trl
```

### One-Command Execution

**Simply run the pipeline script to execute the complete workflow:**

```bash
cd src1
bash run_pipeline.sh
```

This single command will automatically:
1. **Initialize models** with fair or unfair competition settings
2. **Run competitive evaluations** for multiple iterations
3. **Calculate dynamic ratings** using advanced rating systems
4. **Generate preference data** from model competitions
5. **Train models with DPO** using the preference data


## Supported Tasks

- **GSM8K**: Mathematical reasoning problems
- **Alpaca**: Instruction following tasks
- **TruthfulQA**: Truthfulness evaluation
- **Culture Datasets**: Country, value, and rule-of-thumb cultural understanding
- **Knowledge Crossword**: Knowledge-based question answering
- **MATH**: Mathematical problem solving (easy/medium/hard)
- **COM2**: Communication tasks

## Model Types

### Supported Base Models
- **Qwen**: Alibaba's Qwen models with specialized chat templates
- **Gemma**: Google's Gemma models with optimized generation

### Model Configuration

Models are configured in `model_configs.py`:

```python
MODEL_CONFIGS = {
    "qwen": {
        "base_model": "Qwen/Qwen-7B-Chat",
        "adapter": None,
        "is_adapter": False
    },
    "gemma": {
        "base_model": "google/gemma-7b-it", 
        "adapter": None,
        "is_adapter": False
    }
}
```

## Rating Systems

### 1. Normal Rating System
Standard Elo-based rating with configurable parameters:
- Initial K factor: 10.0
- Minimum K factor: 5.0
- Window size: 10
- Decay rate: 0.9

### 2. Dynamic Weighted System
Adaptive rating system that adjusts based on performance history:
- Dynamic K factor adjustment
- Performance-based weighting
- Historical trend analysis

### 3. Static Weighted System
Fixed weighting system for consistent evaluation:
- Predefined model weights
- Stable rating calculations
- Consistent evaluation metrics

## Configuration

All configuration is done by editing variables in `run_pipeline.sh`. No need to modify Python scripts or command-line arguments.

## File Structure

```
sparta/
├── src1/                    # Main source code
│   ├── competition.py       # Competition orchestration
│   ├── inference.py         # Model inference engine
│   ├── model_configs.py     # Model configurations
│   ├── model_init.py        # Model initialization
│   ├── rating_system.py     # Rating algorithms
│   ├── rating_tracker.py    # Rating tracking
│   ├── dpo_new.py          # DPO training
│   └── run_pipeline.sh     # Pipeline execution script
├── data/                    # Dataset directory
├── init_models/            # Initial model checkpoints
├── results/                # Competition results
└── aligned_models/         # Trained model outputs
```

## Configuration

All configuration is done by editing variables in `run_pipeline.sh`. Key settings include:

### Main Configuration Variables

```bash
# Task and Models
TASK="alpaca"                                    # Task: gsm8k, alpaca, truthfulqa, etc.
MODEL_NAMES="code_alpaca,cot,flan_v2,gemini_alpaca,lima,oasst1"  # Model names
BASE_MODEL="gemma,gemma,gemma,qwen,qwen,qwen"   # Base model types

# Hardware
GPU_IDS="6,7,8"                                  # GPU allocation
MAX_ITERATIONS=8                                 # Number of iterations

# Competition Settings
SCORE_TYPE="static"                              # Rating system: normal/dynamic/static
RANDOM_MATCH_PROB=0.6                           # Random opponent probability
NUM_OPPONENTS=5                                  # Number of potential opponents
NUM_INSTRUCTIONS=1000                            # Instructions per iteration

# Training Settings
NUM_EPOCHS=1                                     # DPO training epochs
LR=1e-7                                          # Learning rate
BS=8                                             # Batch size
```

### How to Customize

Simply edit the variables in `run_pipeline.sh`:

1. **Change task**: Modify `TASK="your_task"`
2. **Change models**: Update `MODEL_NAMES` and `BASE_MODEL`
3. **Change GPU**: Update `GPU_IDS`
4. **Change iterations**: Modify `MAX_ITERATIONS`
5. **Change training**: Adjust `NUM_EPOCHS`, `LR`, `BS`

## Advanced Customization

For advanced users who want to modify the core functionality:

### Adding New Models
1. Update `model_configs.py` with new model configuration
2. Implement model-specific loading in `inference.py`
3. Add the model to `MODEL_NAMES` in `run_pipeline.sh`

### Adding New Tasks
1. Create data extraction function in `competition.py`
2. Add task handling in the main evaluation loop
3. Use the new task name in `TASK` variable in `run_pipeline.sh`

### Adding New Rating Systems
1. Extend `RatingSystem` base class in `rating_system.py`
2. Implement custom rating calculation logic
3. Use the new system in `SCORE_TYPE` variable in `run_pipeline.sh`

## Performance Optimization

### GPU Memory Management
- Automatic memory cleanup between batches
- Efficient model loading and unloading
- Multi-GPU support with load balancing

### Batch Processing
- Configurable batch sizes based on GPU memory
- Dynamic batch size adjustment
- Parallel processing support

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use fewer models
2. **Model Loading Errors**: Check model paths and configurations
3. **Rating System Errors**: Verify rating parameters and data format

### Debug Mode

For debugging, you can run individual components:

```bash
# Run competition phase only
python competition.py 0 --task alpaca --base_model qwen,qwen,gemma,gemma

# Run DPO training only  
python dpo_new.py 0 --base_dir ./results --model_names model1,model2
```

## Citation

If Sparta is helpful you:

```bibtex
@misc{jiang2025spartaalignmentcollectivelyaligning,
      title={SPARTA ALIGNMENT: Collectively Aligning Multiple Language Models through Combat}, 
      author={Yuru Jiang and Wenxuan Ding and Shangbin Feng and Greg Durrett and Yulia Tsvetkov},
      year={2025},
      eprint={2506.04721},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.04721}, 
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
