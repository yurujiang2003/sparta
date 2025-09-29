import numpy as np
from typing import List, Dict, Tuple, Optional
import math
import json
import random
import matplotlib.pyplot as plt
import os

class RatingSystem:
    def __init__(self, model_scores: Dict[str, Dict[str, float]], 
                 initial_K: float, 
                 min_K: float,
                 delta_history: Dict[str, List[float]] = None,
                 window_size: int = 10,
                 min_deviation: float = 0.1,
                 epsilon: float = 0.01,
                 decay_rate: float = 0.9,
                 decay_steps: int = 10,
                 scaling_factor: float = 20.0,
                 freeze_ratings: bool = False):
        """
        Initialize the rating system with pre-defined model scores
        Args:
            model_scores: Dictionary of model scores and deviations
            initial_K: Initial scaling factor for rating updates
            min_K: Minimum value for K
            delta_history: Optional historical delta values for each model
            window_size: Size of window for calculating deviation
            min_deviation: Minimum allowed deviation value
            epsilon: Small constant to prevent division by zero
            decay_rate: Rate at which K decays
            decay_steps: Number of steps for K decay
            scaling_factor: Factor to scale delta updates
            freeze_ratings: Boolean to freeze ratings and deviations
        """
        self.initial_K = initial_K
        self.min_K = min_K
        self.K = initial_K
        self.model_ratings = model_scores.copy()
        self.window_size = window_size
        self.min_deviation = min_deviation
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.scaling_factor = scaling_factor
        self.freeze_ratings = freeze_ratings

        if delta_history is None:
            self.delta_history = {model: [] for model in model_scores}
        else:
            self.delta_history = delta_history

            for model in model_scores:
                if model in delta_history and len(delta_history[model]) >= 2:
                    new_deviation = np.std(delta_history[model])
                    self.model_ratings[model]['deviation'] = max(new_deviation, self.min_deviation)

    def select_preference_response(self, pair: Dict) -> Dict:
        """
        Select preference response based on weighted judge scores
        Args:
            pair: {
                'instruction': str,
                'models': [model1, model2],
                'responses': [response1, response2],
                'judges': {
                    'judge_name1': {
                        'scores': [score1, score2],
                        'responses': {
                            'normal': str,
                            'inverse': str,
                            'default_scores_used': bool
                        }
                    },
                    'judge_name2': {...},
                }
            }
        Returns:
            Dict: {
                'instruction': str,
                'chosen': str,      # better response
                'rejected': str,    # worse response
                'chosen_model': str,
                'rejected_model': str,
                'score_diff': float,  # weighted score difference
                'weighted_scores': [float, float]  # weighted scores for both responses
            }
        """
        model_a, model_b = pair['models']
        response_a, response_b = pair['responses']

        total_weight = 0
        weighted_score_a = 0
        weighted_score_b = 0
        
        for judge_name, judge_info in pair['judges'].items():
            if judge_name in [model_a, model_b]:  # Skip if judge is one of the models
                continue

            judge_weight = self.model_ratings[judge_name]['score']
            
            score_a, score_b = judge_info['ave_scores']
            
            weighted_score_a += judge_weight * score_a
            weighted_score_b += judge_weight * score_b
            total_weight += judge_weight

        if total_weight == 0:
            return None

        # Normalize weighted scores
        weighted_score_a /= total_weight
        weighted_score_b /= total_weight
        score_diff = weighted_score_a - weighted_score_b

        # Select preference based on weighted scores
        if weighted_score_a > weighted_score_b:
            preference = {
                'instruction': pair['instruction'],
                'chosen': response_a,
                'rejected': response_b,
                'chosen_model': model_a,
                'rejected_model': model_b,
                'score_diff': score_diff,
                'weighted_scores': [weighted_score_a, weighted_score_b]
            }
        else:
            preference = {
                'instruction': pair['instruction'],
                'chosen': response_b,
                'rejected': response_a,
                'chosen_model': model_b,
                'rejected_model': model_a,
                'score_diff': -score_diff,
                'weighted_scores': [weighted_score_b, weighted_score_a]
            }

        return preference

    def update_ratings_from_judges(self, pairs: List[Dict]) -> None:
        """
        Update ratings based on the exact formula:
        R_i' = R_i 
               + K * ( sum_{j not in {a,b}} [ R_j * (s_{j,i} - s_{j,i+1}) ] / sum_{j not in {a,b}} R_j )
               * tanh(σ_i)
               * max(|Φ((R_i - R_{i+1})/sqrt(σ_i²+σ_{i+1}²)) - Φ((R_{i+1} - R_i)/sqrt(σ_i²+σ_{i+1}²))|, ε)
        """
        if self.freeze_ratings:
            return

        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be a dictionary or list of dictionaries")

        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        self.K = max(self.min_K, 
                    self.initial_K * (self.decay_rate ** (self.update_count / self.decay_steps)))

        model_deltas = {model: [] for model in self.model_ratings}
        old_scores = {model: self.model_ratings[model]['score'] for model in self.model_ratings}
        old_deviations = {model: self.model_ratings[model]['deviation'] for model in self.model_ratings}

        for pair in pairs:
            if not isinstance(pair, dict) or 'models' not in pair:
                continue

            model_a, model_b = pair['models']

            numerator = 0
            denominator = 0
            
            for judge_name, judge_info in pair['judges'].items():
                if judge_name in [model_a, model_b]: 
                    continue
                    
                judge_rating = self.model_ratings[judge_name]['score']
                score_a, score_b = judge_info['ave_scores']
                
                numerator += judge_rating * (score_a - score_b)
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

                delta = delta / self.scaling_factor  # Use scaling_factor parameter

                old_score = self.model_ratings[model_i]['score']
                new_score = max(10, old_score + delta)
                actual_delta = new_score - old_score
                
                model_deltas[model_i].append(actual_delta)
                self.model_ratings[model_i]['score'] = new_score

        for model in model_deltas:
            if model_deltas[model]:

                self.delta_history[model].extend(model_deltas[model])
                self.delta_history[model] = self.delta_history[model][-self.window_size:]

                if len(self.delta_history[model]) >= 2:  
                    new_deviation = np.std(self.delta_history[model])
                    self.model_ratings[model]['deviation'] = max(new_deviation, self.min_deviation)

        print(f"\nUpdate count: {self.update_count}")
        print(f"Current K value: {self.K:.2f}")
        print("\nDeviation changes:")
        for model in self.model_ratings:
            if model in old_deviations:
                print(f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}")

    def compute_expected_score(self, model_a: str, model_b: str) -> float:
        """
        Compute expected score using the Phi function
        """
        R_a = self.model_ratings[model_a]['score']
        R_b = self.model_ratings[model_b]['score']
        sigma_a = self.model_ratings[model_a]['deviation']
        sigma_b = self.model_ratings[model_b]['deviation']
        
        combined_deviation = math.sqrt(sigma_a**2 + sigma_b**2)
        return 0.5 * (1 + math.erf((R_a - R_b) / (math.sqrt(2) * combined_deviation)))

    def get_model_rating(self, model_name: str) -> Dict:
        """Get current rating and deviation for a model"""
        return self.model_ratings.get(model_name, None)

    def get_all_ratings(self) -> Dict[str, Dict]:
        """Get current ratings for all models"""
        return self.model_ratings

    def _update_deviation(self, model: str, rating_change: float) -> None:
        """
        Update the deviation (uncertainty) of a model's rating based on rating changes
        
        Args:
            model (str): The model name
            rating_change (float): The change in rating from the last update
        """
        if model not in self.delta_history:
            self.delta_history[model] = []
            
        self.delta_history[model].append(abs(rating_change))

        self.delta_history[model] = self.delta_history[model][-self.window_size:]

        if len(self.delta_history[model]) >= 2:
            new_deviation = np.std(self.delta_history[model])
            min_deviation = 0.1  
            self.model_ratings[model]['deviation'] = max(new_deviation, min_deviation)

    def _record_delta(self, model: str, delta: float) -> None:
        """
        Record a rating change for a model
        
        Args:
            model (str): The model name
            delta (float): The rating change to record
        """
        if model not in self.delta_history:
            self.delta_history[model] = []
        
        self.delta_history[model].append(delta)
        self.delta_history[model] = self.delta_history[model][-self.window_size:]

class RatingSystemDynamicWeighted(RatingSystem):
    def __init__(self, model_scores: Dict[str, Dict[str, float]], 
                 initial_K: float, 
                 min_K: float,
                 delta_history: Dict[str, List[float]] = None,
                 base_dir: str = None,
                 current_iteration: int = None,
                 window_size: int = 10,
                 min_deviation: float = 0.1,
                 epsilon: float = 0.01,
                 decay_rate: float = 0.9,
                 decay_steps: int = 10,
                 scaling_factor: float = 10.0,
                 freeze_ratings: bool = False):  # Note: different default scaling
        """
        Initialize the weighted rating system with all configurable parameters
        """
        super().__init__(
            model_scores=model_scores,
            initial_K=initial_K,
            min_K=min_K,
            delta_history=delta_history,
            window_size=window_size,
            min_deviation=min_deviation,
            epsilon=epsilon,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            scaling_factor=scaling_factor,
            freeze_ratings=freeze_ratings
        )
        self.base_dir = base_dir
        self.current_iteration = current_iteration
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> dict:
        """
        Calculate the weights for each model based on previous iteration scores
        
        Returns:
            dict: model name to weight mapping
        """
        # 默认权重为1.0
        weights = {model: 1.0 for model in self.model_ratings.keys()}
        
        if not self.base_dir or self.current_iteration is None:
            return weights
            
        try:
            # 如果是第8次迭代或之后，加载第7次迭代的权重并固定
            if self.current_iteration >= 8:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                if os.path.exists(weights_path):
                    with open(weights_path, 'r') as f:
                        return json.load(f)
                return weights
            
            # 第7次迭代及之前的动态权重计算逻辑
            if self.current_iteration >= 2:
                prev_iter = self.current_iteration - 1
                prev_model_info_path = os.path.join(
                    self.base_dir, 
                    f"iteration_{prev_iter}", 
                    "model_info.json"
                )
                
                with open(prev_model_info_path, 'r') as f:
                    prev_model_info = json.load(f)

                sorted_models = sorted(
                    prev_model_info.keys(),
                    key=lambda x: prev_model_info[x]['score']
                )
                
                num_weighted = self.current_iteration - 1
                for i in range(min(num_weighted, len(sorted_models))):
                    model = sorted_models[i]
                    if i == 0:
                        weights[model] = 0.0
                    else:
                        weights[model] = 0.1 * i
                
                # 在第7次迭代时保存权重
                if self.current_iteration == 7:
                    weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                    with open(weights_path, 'w') as f:
                        json.dump(weights, f, indent=2)
                        
            return weights
            
        except Exception as e:
            print(f"Error calculating weights: {e}")
            return weights

    def update_ratings_from_judges(self, pairs: List[Dict]) -> None:
        """
        Update ratings based on the exact formula:
        R_i' = R_i 
               + K * ( sum_{j not in {a,b}} [ R_j * (s_{j,i} - s_{j,i+1}) ] / sum_{j not in {a,b}} R_j )
               * tanh(σ_i)
               * max(|Φ((R_i - R_{i+1})/sqrt(σ_i²+σ_{i+1}²)) - Φ((R_{i+1} - R_i)/sqrt(σ_i²+σ_{i+1}²))|, ε)
        """
        if self.freeze_ratings:
            return

        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be a dictionary or list of dictionaries")

        # Initialize update counter and adjust K
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        decay_rate = 0.9
        self.K = max(self.min_K, self.initial_K * (decay_rate ** (self.update_count / 10)))

        # Track rating changes
        model_deltas = {model: [] for model in self.model_ratings}
        old_scores = {model: self.model_ratings[model]['score'] for model in self.model_ratings}
        old_deviations = {model: self.model_ratings[model]['deviation'] for model in self.model_ratings}

        for pair in pairs:
            if not isinstance(pair, dict) or 'models' not in pair:
                continue

            model_a, model_b = pair['models']
            
            # Calculate weighted score difference exactly as per formula
            numerator = 0
            denominator = 0
            
            for judge_name, judge_info in pair['judges'].items():
                if judge_name in [model_a, model_b]:  # Skip if judge is one of the models
                    continue
                    
                judge_rating = self.model_ratings[judge_name]['score']
                score_a, score_b = judge_info['ave_scores']
                
                # Apply weights to the scores
                score_a *= self.weights[model_a]
                score_b *= self.weights[model_b]
                
                numerator += judge_rating * (score_a - score_b)
                denominator += judge_rating

            if denominator == 0:
                continue

            # Normalized score difference term
            score_diff = numerator / denominator

            # Update both models
            for i, model_i in enumerate([model_a, model_b]):
                R_i = self.model_ratings[model_i]['score']
                R_next = self.model_ratings[model_b if i == 0 else model_a]['score']
                sigma_i = self.model_ratings[model_i]['deviation']
                sigma_next = self.model_ratings[model_b if i == 0 else model_a]['deviation']

                # Calculate combined deviation term
                combined_deviation = math.sqrt(sigma_i**2 + sigma_next**2)

                # Calculate Phi terms exactly as in formula
                phi_forward = 0.5 * (1 + math.erf((R_i - R_next) / (math.sqrt(2) * combined_deviation)))
                phi_backward = 0.5 * (1 + math.erf((R_next - R_i) / (math.sqrt(2) * combined_deviation)))
                
                # Calculate delta exactly matching the formula
                delta = (self.K 
                        * (score_diff if i == 0 else -score_diff)  # Score difference term
                        * math.tanh(sigma_i)  # tanh(σ_i) term
                        * max(abs(phi_forward - phi_backward), self.epsilon))  # Phi difference term

                delta = delta / 10.0 ### scaling trial

                # Update score and track changes
                old_score = self.model_ratings[model_i]['score']
                new_score = max(10, old_score + delta)
                actual_delta = new_score - old_score
                
                model_deltas[model_i].append(actual_delta)
                self.model_ratings[model_i]['score'] = new_score

        # Update deviations based on recent deltas
        for model in model_deltas:
            if model_deltas[model]:

                self.delta_history[model].extend(model_deltas[model])
                self.delta_history[model] = self.delta_history[model][-self.window_size:]

                if len(self.delta_history[model]) >= 2:  
                    new_deviation = np.std(self.delta_history[model])
                    min_deviation = 0.1  
                    
                    self.model_ratings[model]['deviation'] = max(new_deviation, min_deviation)

        # Print updates
        print(f"\nUpdate count: {self.update_count}")
        print(f"Current K value: {self.K:.2f}")
        print("\nDeviation changes:")
        for model in self.model_ratings:
            if model in old_deviations:
                print(f"{model}: {old_deviations[model]:.4f} -> {self.model_ratings[model]['deviation']:.4f}")

    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights for all models
        
        Returns:
            Dict[str, float]: model name to weight mapping
        """
        return self.weights
    
class RatingSystemStaticWeighted(RatingSystem):
    def __init__(self, model_scores: Dict[str, Dict[str, float]], 
                 initial_K: float, 
                 min_K: float,
                 delta_history: Dict[str, List[float]] = None,
                 base_dir: str = None,
                 current_iteration: int = None,
                 window_size: int = 10,
                 min_deviation: float = 0.1,
                 epsilon: float = 0.01,
                 decay_rate: float = 0.9,
                 decay_steps: int = 10,
                 scaling_factor: float = 20.0,
                 freeze_ratings: bool = False):
        """
        Initialize the static weighted rating system with all configurable parameters
        """
        super().__init__(
            model_scores=model_scores,
            initial_K=initial_K,
            min_K=min_K,
            delta_history=delta_history,
            window_size=window_size,
            min_deviation=min_deviation,
            epsilon=epsilon,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            scaling_factor=scaling_factor,
            freeze_ratings=freeze_ratings
        )
        self.base_dir = base_dir
        self.current_iteration = current_iteration
        self.weights = self._calculate_static_weights()
        
    def _calculate_static_weights(self) -> dict:
        """
        Calculate static weights based on iteration order.
        Track the complete history of weighted models across all iterations.
        
        Returns:
            dict: model name to weight mapping
        """
        weights = {model: 1.0 for model in self.model_ratings.keys()}
        
        if not self.base_dir or self.current_iteration is None:
            return weights
            
        try:
            # 如果是第8次迭代或之后，加载第7次迭代的权重并固定
            if self.current_iteration >= 8:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                if os.path.exists(weights_path):
                    with open(weights_path, 'r') as f:
                        return json.load(f)
                return weights
            
            # 第7次迭代及之前的静态权重计算逻辑
            weighted_models = []
            
            for iter_num in range(2, self.current_iteration + 1):
                prev_iter = iter_num - 1
                prev_model_info_path = os.path.join(
                    self.base_dir, 
                    f"iteration_{prev_iter}", 
                    "model_info.json"
                )
                
                try:
                    with open(prev_model_info_path, 'r') as f:
                        prev_model_info = json.load(f)
                    
                    remaining_models = [
                        model for model in prev_model_info.keys()
                        if model not in weighted_models
                    ]
                    
                    if remaining_models:
                        sorted_models = sorted(
                            remaining_models,
                            key=lambda x: prev_model_info[x]['score']
                        )
                        
                        model = sorted_models[0]
                        weighted_models.append(model)
                        
                        weight_index = len(weighted_models) - 1
                        if weight_index == 0:
                            weights[model] = 0.0
                        else:
                            weights[model] = 0.1 * weight_index
                    
                except Exception as e:
                    print(f"Error reading iteration {prev_iter}: {e}")
                    continue
            
            # save weights at the 7th iteration
            if self.current_iteration == 7:
                weights_path = os.path.join(self.base_dir, "iteration_7", "weights.json")
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                with open(weights_path, 'w') as f:
                    json.dump(weights, f, indent=2)
                    
            return weights
            
        except Exception as e:
            print(f"Error calculating static weights: {e}")
            return weights
            
    def update_ratings_from_judges(self, pairs: List[Dict]) -> None:
        """
        Update ratings based on the exact formula:
        R_i' = R_i 
               + K * ( sum_{j not in {a,b}} [ R_j * (s_{j,i} - s_{j,i+1}) ] / sum_{j not in {a,b}} R_j )
               * tanh(σ_i)
               * max(|Φ((R_i - R_{i+1})/sqrt(σ_i²+σ_{i+1}²)) - Φ((R_{i+1} - R_i)/sqrt(σ_i²+σ_{i+1}²))|, ε)
        """
        if self.freeze_ratings:
            return

        if isinstance(pairs, dict):
            pairs = [pairs]
        elif not isinstance(pairs, list):
            raise ValueError("Input must be a dictionary or list of dictionaries")

        # Initialize update counter and adjust K
        if not hasattr(self, 'update_count'):
            self.update_count = 0
        self.update_count += 1
        
        decay_rate = 0.9
        self.K = max(self.min_K, self.initial_K * (decay_rate ** (self.update_count / 10)))

        # Track rating changes
        model_deltas = {model: [] for model in self.model_ratings}
        old_scores = {model: self.model_ratings[model]['score'] for model in self.model_ratings}
        old_deviations = {model: self.model_ratings[model]['deviation'] for model in self.model_ratings}

        for pair in pairs:
            if not isinstance(pair, dict) or 'models' not in pair:
                continue

            model_a, model_b = pair['models']
            
            # Calculate weighted score difference exactly as per formula
            numerator = 0
            denominator = 0
            
            for judge_name, judge_info in pair['judges'].items():
                if judge_name in [model_a, model_b]:  # Skip if judge is one of the models
                    continue
                    
                judge_rating = self.model_ratings[judge_name]['score']
                score_a, score_b = judge_info['ave_scores']
                
                # Apply weights to the scores
                score_a *= self.weights[model_a]
                score_b *= self.weights[model_b]
                
                numerator += judge_rating * (score_a - score_b)
                denominator += judge_rating

            if denominator == 0:
                continue

            # Normalized score difference term
            score_diff = numerator / denominator

            # Update both models
            for i, model_i in enumerate([model_a, model_b]):
                R_i = self.model_ratings[model_i]['score']
                R_next = self.model_ratings[model_b if i == 0 else model_a]['score']
                sigma_i = self.model_ratings[model_i]['deviation']
                sigma_next = self.model_ratings[model_b if i == 0 else model_a]['deviation']

                # Calculate combined deviation term
                combined_deviation = math.sqrt(sigma_i**2 + sigma_next**2)

                # Calculate Phi terms exactly as in formula
                phi_forward = 0.5 * (1 + math.erf((R_i - R_next) / (math.sqrt(2) * combined_deviation)))
                phi_backward = 0.5 * (1 + math.erf((R_next - R_i) / (math.sqrt(2) * combined_deviation)))
                
                # Calculate delta exactly matching the formula
                delta = (self.K 
                        * (score_diff if i == 0 else -score_diff)  # Score difference term
                        * math.tanh(sigma_i)  # tanh(σ_i) term
                        * max(abs(phi_forward - phi_backward), self.epsilon))  # Phi difference term

                delta = delta / 20.0 ### scaling trial

                # Update score and track changes
                old_score = self.model_ratings[model_i]['score']
                new_score = max(10, old_score + delta)
                actual_delta = new_score - old_score
                
                model_deltas[model_i].append(actual_delta)
                self.model_ratings[model_i]['score'] = new_score

        # Update deviations based on recent deltas
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
    def get_weights(self) -> Dict[str, float]:
        """
        Get current weights for all models
        
        Returns:
            Dict[str, float]: model name to weight mapping
        """
        return self.weights
    
#######################Below is the test code#######################

def test_generate_preferences(json_path: str) -> List[Dict]:
    """
    generate preferences from judged pairs
    
    Args:
        json_path (str): the path of the judged pairs json file
    
    Returns:
        List[Dict]: the list of generated preference pairs
    """
    try:

        with open(json_path, 'r', encoding='utf-8') as f:
            judged_pairs = json.load(f)

        model_names = set()
        for pair in judged_pairs:
            model_names.update(pair['models'])
            if 'judges' in pair:
                model_names.update(pair['judges'].keys())
                
        model_info_path = ''
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        initial_scores = {
            model: {
                'score': model_info[model]['score'], 
                'deviation': model_info[model]['deviation']
            }
            for model in model_names
        }

        rating_system = RatingSystemStaticWeighted(
            model_scores=initial_scores,
            initial_K=20,
            min_K= 10
        )
        
        preferences = []
        for pair in judged_pairs:
            preference = rating_system.select_preference_response(pair)
            if preference is not None:
                preferences.append(preference)

        print(f"\nGenerated {len(preferences)} preferences from {len(judged_pairs)} pairs")

        if preferences:
            print("\nExample preference:")
            example = preferences[0]
            print(f"Instruction: {example['instruction']}")
            print(f"Chosen model: {example['chosen_model']}")
            print(f"Rejected model: {example['rejected_model']}")
            print(f"Score difference: {example['score_diff']:.4f}")
            print(f"Weighted scores: {example['weighted_scores']}")

        output_path = json_path.replace('.json', '_preferences.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2, ensure_ascii=False)
        print(f"\nPreferences saved to: {output_path}")
        
        return preferences
        
    except Exception as e:
        print(f"Error generating preferences: {e}")
        return []

def main():
    test_generate_preferences('') ## add the path of the judged pairs json file

if __name__ == "__main__":
    main()