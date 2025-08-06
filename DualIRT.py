from utils import (
    load_valid_csv,
    load_public_test_csv,
)
import pickle
import numpy as np

# Import the model classes so pickle can deserialize them
from question_irt import QuestionIRT
from subject_irt import SubjectIRT

def load_question_model(model_save_path="question_irt_model.pkl"):
    """Load a trained QuestionIRT model from file."""
    with open(model_save_path, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Question model loaded from {model_save_path}")
    return loaded_model

def load_subject_model(model_save_path="subject_irt_model.pkl"):
    """Load a trained SubjectIRT model from file."""
    with open(model_save_path, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Subject model loaded from {model_save_path}")
    return loaded_model

class DualIRT:
    """Dual IRT model that combines QuestionIRT and SubjectIRT with weighted ensemble."""
    
    def __init__(self, question_model=None, subject_model=None, 
                 question_model_path="question_irt_model.pkl", 
                 subject_model_path="subject_irt_model.pkl"):
        """Initialize DualIRT model.
        
        Args:
            question_model: Pre-loaded QuestionIRT model (optional)
            subject_model: Pre-loaded SubjectIRT model (optional)
            question_model_path: Path to saved QuestionIRT model
            subject_model_path: Path to saved SubjectIRT model
        """
        if question_model is None:
            self.question_model = load_question_model(question_model_path)
        else:
            self.question_model = question_model
            
        if subject_model is None:
            self.subject_model = load_subject_model(subject_model_path)
        else:
            self.subject_model = subject_model
    
    def predict_probabilities(self, data, weight=0.5):
        """Generate probability predictions using weighted ensemble.
        
        Args:
            data: Dictionary with user_id, question_id lists
            weight: Weight for question model (0-1). Subject model gets (1-weight)
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Get predictions from both models
        pred_1, prob_1 = self.question_model.predict_probabilities(data)
        pred_2, prob_2 = self.subject_model.predict_probabilities(data)
        
        # Convert to numpy arrays for easier computation
        prob_1 = np.array(prob_1)
        prob_2 = np.array(prob_2)
        
        # Weighted combination
        total_prob = weight * prob_1 + (1 - weight) * prob_2
        
        # Convert back to lists and generate predictions
        combined_probs = total_prob.tolist()
        combined_preds = [p >= 0.5 for p in combined_probs]
        
        return combined_preds, combined_probs
    
    def evaluate(self, data, weight=0.5):
        """Evaluate the dual model on given data.
        
        Args:
            data: Dictionary with user_id, question_id, is_correct lists
            weight: Weight for question model (0-1)
            
        Returns:
            float: Accuracy
        """
        predictions, _ = self.predict_probabilities(data, weight)
        
        correct = sum(1 for pred, actual in zip(predictions, data["is_correct"]) 
                     if pred == actual)
        total = len(data["is_correct"])
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_weight_range(self, data, weights=None):
        """Evaluate the model across a range of weights.
        
        Args:
            data: Dictionary with user_id, question_id, is_correct lists
            weights: List of weights to test. Defaults to [0, 0.1, 0.2, ..., 1.0]
            
        Returns:
            tuple: (weights, accuracies)
        """
        if weights is None:
            weights = [i/10 for i in range(11)]  # 0.0 to 1.0 in 0.1 increments
        
        accuracies = []
        
        for weight in weights:
            accuracy = self.evaluate(data, weight)
            accuracies.append(accuracy)
            print(f"Weight {weight:.1f}: Accuracy = {accuracy:.4f}")
        
        return weights, accuracies
    
    def find_best_weight(self, val_data, weights=None):
        """Find the best weight using validation data.
        
        Args:
            val_data: Validation data dictionary
            weights: List of weights to test
            
        Returns:
            float: Best weight value
        """
        weights, accuracies = self.evaluate_weight_range(val_data, weights)
        
        best_idx = np.argmax(accuracies)
        best_weight = weights[best_idx]
        best_accuracy = accuracies[best_idx]
        
        print(f"\nBest weight: {best_weight:.1f} (Accuracy: {best_accuracy:.4f})")
        
        return best_weight

def main():
    """Main function to evaluate DualIRT model."""
    # Initialize dual model
    dual_model = DualIRT()
    
    # Load data
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    print("Evaluating DualIRT model on validation data:")
    print("=" * 50)

    # Evaluate across weight range on validation data
    best_weight = dual_model.find_best_weight(val_data)
    print(f"\nEvaluating on test data with best weight: {best_weight:.1f}:")
    
    test_accuracy = dual_model.evaluate(test_data, best_weight)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print(f"Best Dual Model weight={best_weight:.1f}): {test_accuracy:.4f}")

if __name__ == "__main__":
    main()