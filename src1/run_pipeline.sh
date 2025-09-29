#!/bin/bash
# Source conda
source ~/miniconda3/etc/profile.d/conda.sh

echo "Starting 3 complete runs of the pipeline"

for run in {1..1}; do
    echo "==============================================="
    echo "Starting Run $run of 3"
    echo "==============================================="

    # Get the directory of the script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

    # Generate unique identifier using timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    EXPERIMENT_NAME="static_${TIMESTAMP}_run${run}"  

    # Configuration variables
    BASE_MODEL="gemma,gemma,gemma,qwen,qwen,qwen"
    BASE_DIR="$PROJECT_ROOT/dpo_model/${EXPERIMENT_NAME}"
    INIT_MODEL_DIR="$PROJECT_ROOT/init_model"
    BASE_MODEL_DIR="$PROJECT_ROOT/base_model/$BASE_MODEL"
    DATA_DIR="$PROJECT_ROOT/data"
    MAX_ITERATIONS=8
    GPU_IDS="6,7,8"
    WANDB_PROJECT="model_evaluation"
    USE_FAIR=true
    FIRST_GPU=$(echo $GPU_IDS | cut -d',' -f1)
    NUM_INSTRUCTIONS=1000
    BATCH_SIZE=24
    CONDA_ENV="rebuttal"
    MODEL_NAMES="code_alpaca,cot,flan_v2,gemini_alpaca,lima,oasst1"
    ACCELERATE_CONFIG="$SCRIPT_DIR/accelerate_config/fsdp.yaml"

    # Competition phase configuration
    TASK="alpaca"                      # Task name, choose from [gsm8k, alpaca, truthfulqa, medqa, culture_country, culture_value, culture_rule_of_thumb, knowledge_crossword]
    SCORE_TYPE="static"                # Rating system type: 'dynamic' or 'static'
    RANDOM_MATCH_PROB=0.6           # Probability of random opponent selection
    NUM_OPPONENTS=5                   # Number of potential opponents to choose from
    RANDOM_SEED=42                    # Random seed for reproducibility
    RANDOM_SELECT=true
    DIFFICULTY="hard"
    FREEZE_RATINGS=false

    # Rating system configuration
    INITIAL_K=10.0                    # Initial K factor for rating system
    MIN_K=5.0                         # Minimum K factor for rating system
    WINDOW_SIZE=10                    # Window size for rating history
    MIN_DEVIATION=0.1                 # Minimum deviation threshold
    EPSILON=0.01                      # Epsilon for rating calculations
    DECAY_RATE=0.9                    # Decay rate for rating adjustments
    DECAY_STEPS=10                    # Number of steps for decay
    SCALING_FACTOR=20.0               # Scaling factor for rating calculations

    # DPO phase configuration
    NUM_EPOCHS=1
    BS=8 ### keep unchanged
    GAS=16
    MSEQLEN=512
    LR=1e-7
    QLORA=true
    COMPUTE_DTYPE="bfloat16"
    ATTN_IMPLEMENTATION="sdpa"

    clean_gpu() {
        echo "Cleaning GPU memory..."
        nvidia-smi --gpu-reset
        sleep 5
    }

    # Function to check if a command was successful
    check_status() {
        if [ $? -ne 0 ]; then
            echo "Error: $1 failed in iteration $2"
            exit 1
        fi
    }


    echo "Starting pipeline with $MAX_ITERATIONS iterations"

    for ((i=0; i<MAX_ITERATIONS; i++)); do
        echo "Starting iteration $i"

        echo "Running competition phase..."
        conda activate $CONDA_ENV

        if [ "$USE_FAIR" = true ]; then
            echo "Using fair competition"
            python "${SCRIPT_DIR}/competition.py" $i \
                --task "$TASK" \
                --fair \
                --score_type "$SCORE_TYPE" \
                --base_model "$BASE_MODEL" \
                --base_dir "$BASE_DIR" \
                --init_model_dir "$INIT_MODEL_DIR" \
                --data_dir "$DATA_DIR" \
                --gpu_ids "$GPU_IDS" \
                --num_instructions $NUM_INSTRUCTIONS \
                --batch_size $BATCH_SIZE \
                --model_names "$MODEL_NAMES" \
                --random_select $RANDOM_SELECT \
                --random_match_prob $RANDOM_MATCH_PROB \
                --difficulty $DIFFICULTY \
                --freeze_ratings $FREEZE_RATINGS \
                --num_opponents $NUM_OPPONENTS \
                --random_seed $RANDOM_SEED \
                --initial_k $INITIAL_K \
                --min_k $MIN_K \
                --window_size $WINDOW_SIZE \
                --min_deviation $MIN_DEVIATION \
                --epsilon $EPSILON \
                --decay_rate $DECAY_RATE \
                --decay_steps $DECAY_STEPS \
                --scaling_factor $SCALING_FACTOR
        else
            echo "Using non-fair competition"
            python "${SCRIPT_DIR}/competition.py" $i \
                --task "$TASK" \
                --no-fair \
                --score_type "$SCORE_TYPE" \
                --base_model "$BASE_MODEL" \
                --base_dir "$BASE_DIR" \
                --init_model_dir "$INIT_MODEL_DIR" \
                --data_dir "$DATA_DIR" \
                --gpu_ids "$GPU_IDS" \
                --num_instructions $NUM_INSTRUCTIONS \
                --batch_size $BATCH_SIZE \
                --model_names "$MODEL_NAMES" \
                --random_select $RANDOM_SELECT \
                --random_match_prob $RANDOM_MATCH_PROB \
                --freeze_ratings $FREEZE_RATINGS \
                --difficulty $DIFFICULTY \
                --num_opponents $NUM_OPPONENTS \
                --random_seed $RANDOM_SEED \
                --initial_k $INITIAL_K \
                --min_k $MIN_K \
                --window_size $WINDOW_SIZE \
                --min_deviation $MIN_DEVIATION \
                --epsilon $EPSILON \
                --decay_rate $DECAY_RATE \
                --decay_steps $DECAY_STEPS \
                --scaling_factor $SCALING_FACTOR
        fi

        check_status "Competition" $i

        if [ ! -f "${BASE_DIR}/iteration_${i}/model_info.json" ]; then
            echo "Error: model_info.json not found for iteration $i"
            exit 1
        fi
        
        clean_gpu

        # Split models into groups and run them in parallel
        IFS=',' read -ra MODEL_ARRAY <<< "$MODEL_NAMES"
        IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
        NUM_MODELS=${#MODEL_ARRAY[@]}
        MODELS_PER_GROUP=2
        # Use ceiling division to ensure all models are included
        NUM_GROUPS=$(( (NUM_MODELS + MODELS_PER_GROUP - 1) / MODELS_PER_GROUP ))

        echo "Processing $NUM_MODELS models in $NUM_GROUPS groups"

        for ((group=0; group<NUM_GROUPS; group++)); do
            # Calculate start and end indices for this group
            START_IDX=$((group * MODELS_PER_GROUP))
            END_IDX=$(((group + 1) * MODELS_PER_GROUP))
            if [ $END_IDX -gt $NUM_MODELS ]; then
                END_IDX=$NUM_MODELS
            fi
            
            # Extract models for this group
            GROUP_MODELS=""
            for ((idx=START_IDX; idx<END_IDX; idx++)); do
                if [ $idx -gt $START_IDX ]; then
                    GROUP_MODELS+=","
                fi
                GROUP_MODELS+="${MODEL_ARRAY[idx]}"
            done

            # Get GPU for this group from GPU_IDS
            GPU_ID="${GPU_ARRAY[group]}"
            
            echo "Starting DPO training for group $((group + 1)) (Models: $GROUP_MODELS) on GPU $GPU_ID"
            
            # Run DPO training for this group in background
            CUDA_VISIBLE_DEVICES=$GPU_ID python "${SCRIPT_DIR}/dpo_new.py" \
                $i \
                --base_dir "$BASE_DIR" \
                --model_names "$GROUP_MODELS" \
                --all_model_names "$MODEL_NAMES" \
                --all_base_models "$BASE_MODEL" &
        done

        # Wait for all background processes to complete
        wait
        
        # Check if any of the DPO processes failed
        for job in $(jobs -p); do
            wait $job || { 
                echo "Error: DPO training failed for one or more groups"
                exit 1
            }
        done

        unset CUDA_VISIBLE_DEVICES
        clean_gpu
        
        echo "Completed iteration $i"
    done  # End of iteration loop


    save_parameters() {
        local params_file="$BASE_DIR/parameters.json"
        cat > "$params_file" << EOF
{
    "paths": {
        "base_dir": "$BASE_DIR",
        "init_model_dir": "$INIT_MODEL_DIR",
        "base_model_dir": "$BASE_MODEL_DIR",
        "data_dir": "$DATA_DIR",
        "accelerate_config": "$ACCELERATE_CONFIG"
    },
    "general_config": {
        "max_iterations": $MAX_ITERATIONS,
        "gpu_ids": "$GPU_IDS",
        "first_gpu": "$FIRST_GPU",
        "wandb_project": "$WANDB_PROJECT",
        "num_instructions": $NUM_INSTRUCTIONS,
        "batch_size": $BATCH_SIZE,
        "conda_env": "$CONDA_ENV",
        "model_names": "$MODEL_NAMES"
    },
    "competition_config": {
        "task": "$TASK",
        "score_type": "$SCORE_TYPE",
        "fair_or_not": $USE_FAIR,
        "random_match_prob": $RANDOM_MATCH_PROB,
        "num_opponents": $NUM_OPPONENTS,
        "random_seed": $RANDOM_SEED,
        "random_select": $RANDOM_SELECT,
        "freeze_ratings": $FREEZE_RATINGS,
        "difficulty": "$DIFFICULTY"
    },
    "rating_system": {
        "initial_k": $INITIAL_K,
        "min_k": $MIN_K,
        "window_size": $WINDOW_SIZE,
        "min_deviation": $MIN_DEVIATION,
        "epsilon": $EPSILON,
        "decay_rate": $DECAY_RATE,
        "decay_steps": $DECAY_STEPS,
        "scaling_factor": $SCALING_FACTOR
    }
}
EOF

        echo "Parameters saved to $params_file"
    }

    save_parameters

    touch "$BASE_DIR/experiment_completed"

    echo "Pipeline completed successfully"
    echo "Run $run of 3 completed successfully"
    echo "==============================================="

    if [ $run -lt 3 ]; then
        echo "Waiting 30 seconds before starting next run..."
        sleep 30
    fi
done  # End of run loop

echo "All 3 runs completed successfully"