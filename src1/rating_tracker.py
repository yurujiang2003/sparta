from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
import json
from pathlib import Path

@dataclass
class StopCriteria:
    min_improvement: float = 0.1  # the minimum improvement threshold
    window_size: int = 5  # the size of the window for checking improvement
    patience: int = 3  # the number of iterations without improvement
class RatingTracker:
    def __init__(self, 
                 save_dir: str = "sparta_alignment/ratings",
                 stop_criteria: StopCriteria = StopCriteria()):
        """
        Initialize rating tracker
        Args:
            save_dir: directory to save rating history
            stop_criteria: criteria for early stopping
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stop_criteria = stop_criteria
        
        self.rating_history: Dict[str, List[float]] = {}
        self.iteration_count = 0
        self.no_improvement_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Save ratings after each iteration
        self.ratings_file = self.save_dir / "rating_history.json"
        
    def update(self, current_ratings: Dict[str, Dict[str, float]]) -> bool:
        """
        Update rating history and check stop condition
        Args:
            current_ratings: {model_name: {'score': float, 'deviation': float}}
        Returns:
            bool: True if should stop, False otherwise
        """
        self.iteration_count += 1
        
        # Update history
        for model_name, rating_info in current_ratings.items():
            if model_name not in self.rating_history:
                self.rating_history[model_name] = []
            self.rating_history[model_name].append(rating_info['score'])
        
        # Save current state
        self._save_history()
        
        # Plot current state
        if self.iteration_count % 10 == 0:  # 每10轮画一次图
            self.plot_history()
        
        # Check stop condition
        return self._should_stop()
    
    def _should_stop(self) -> bool:
        """Check if training should stop based on top model's improvement"""
        if self.iteration_count < self.stop_criteria.window_size:
            return False
            
        # Find top model
        latest_scores = {
            model: scores[-1] 
            for model, scores in self.rating_history.items()
        }
        top_model = max(latest_scores.items(), key=lambda x: x[1])[0]
        
        # Get history of top model
        top_history = self.rating_history[top_model]
        if len(top_history) < self.stop_criteria.window_size:
            return False
            
        # Check improvement over window
        window = top_history[-self.stop_criteria.window_size:]
        improvement = window[-1] - window[0]
        
        if improvement < self.stop_criteria.min_improvement:
            self.no_improvement_count += 1
            self.logger.info(f"No significant improvement for {self.no_improvement_count} iterations")
        else:
            self.no_improvement_count = 0
            
        if self.no_improvement_count >= self.stop_criteria.patience:
            self.logger.info(f"Stopping: No significant improvement for {self.stop_criteria.patience} iterations")
            return True
            
        return False
    
    def _save_history(self):
        """Save rating history to file"""
        with open(self.ratings_file, 'w') as f:
            json.dump({
                'iteration_count': self.iteration_count,
                'rating_history': self.rating_history,
                'no_improvement_count': self.no_improvement_count
            }, f, indent=2)
    
    def load_history(self) -> bool:
        """Load rating history from file"""
        if not self.ratings_file.exists():
            return False
            
        try:
            with open(self.ratings_file, 'r') as f:
                data = json.load(f)
                self.iteration_count = data['iteration_count']
                self.rating_history = data['rating_history']
                self.no_improvement_count = data.get('no_improvement_count', 0)
            return True
        except Exception as e:
            self.logger.error(f"Error loading history: {e}")
            return False
    
    def plot_history(self, save_path: str = None):
        """Plot rating history"""
        plt.figure(figsize=(12, 8))
        for model_name, ratings in self.rating_history.items():
            plt.plot(ratings, label=model_name)
            
        plt.title("Model Rating History")
        plt.xlabel("Iteration")
        plt.ylabel("Rating")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig(self.save_dir / f"rating_history_{self.iteration_count}.png")
        plt.close()
    
    def get_summary(self) -> Dict:
        """Get summary of current ratings"""
        latest_ratings = {}
        improvements = {}
        
        for model_name, ratings in self.rating_history.items():
            if len(ratings) > 0:
                latest_ratings[model_name] = ratings[-1]
                if len(ratings) > 1:
                    improvements[model_name] = ratings[-1] - ratings[0]
                    
        return {
            'iteration_count': self.iteration_count,
            'latest_ratings': latest_ratings,
            'total_improvements': improvements,
            'no_improvement_count': self.no_improvement_count
        }

# def main():
#     # Example usage
#     tracker = RatingTracker(
#         save_dir="sparta_alignment/ratings",
#         stop_criteria=StopCriteria(
#             min_improvement=0.1,
#             window_size=5,
#             patience=3
#         )
#     )
    
#     # Load existing history if available
#     tracker.load_history()
    
#     # Simulate some rating updates
#     for i in range(20):
#         current_ratings = {
#             'model1': {'score': 1500 + i * 10 + np.random.randn(), 'deviation': 1.0},
#             'model2': {'score': 1600 + i * 5 + np.random.randn(), 'deviation': 1.2},
#             'model3': {'score': 1700 + i * 2 + np.random.randn(), 'deviation': 0.8}
#         }
        
#         should_stop = tracker.update(current_ratings)
#         if should_stop:
#             print("Early stopping triggered!")
#             break
    
#     # Print final summary
#     summary = tracker.get_summary()
#     print("\nTraining Summary:")
#     print(f"Total iterations: {summary['iteration_count']}")
#     print("\nFinal Ratings:")
#     for model, rating in summary['latest_ratings'].items():
#         print(f"{model}: {rating:.2f}")
#     print("\nTotal Improvements:")
#     for model, improvement in summary['total_improvements'].items():
#         print(f"{model}: {improvement:.2f}")

# if __name__ == "__main__":
#     main()