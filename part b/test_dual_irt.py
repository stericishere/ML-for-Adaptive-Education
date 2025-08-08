import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from question_irt import QuestionIRT
from subject_irt import SubjectIRT
from DualIRT import DualIRT

def test_dual_irt_initialization():
    """Test DualIRT initialization with mock models."""
    # Create mock models
    question_model = QuestionIRT(num_students=10, num_questions=5)
    
    # Mock subject data for SubjectIRT
    question_subjects = {0: [0, 1], 1: [1, 2], 2: [0, 2], 3: [1], 4: [0, 1, 2]}
    subject_model = SubjectIRT(num_students=10, question_subjects=question_subjects)
    
    # Initialize DualIRT with mock models
    dual_model = DualIRT(question_model=question_model, subject_model=subject_model)
    
    # Check that models are properly assigned
    assert dual_model.question_model is question_model, "Question model not properly assigned"
    assert dual_model.subject_model is subject_model, "Subject model not properly assigned"

def test_dual_irt_predict_probabilities():
    """Test probability prediction with weighted ensemble."""
    # Create mock models with known parameters
    question_model = QuestionIRT(num_students=3, num_questions=2)
    question_model.theta = np.array([1.0, 0.0, -1.0])  # High, medium, low ability
    question_model.beta = np.array([0.0, 0.0])         # Neutral difficulty
    question_model.a_list = np.array([1.0, 1.0])       # Standard discrimination
    question_model.b_list = np.array([0.1, 0.1])       # Low guessing
    
    question_subjects = {0: [0], 1: [0]}  # Both questions from subject 0
    subject_model = SubjectIRT(num_students=3, question_subjects=question_subjects)
    subject_model.theta = np.array([[0.5], [0.0], [-0.5]])  # Subject abilities
    subject_model.beta = np.array([0.0, 0.0])               # Question difficulties
    subject_model.a_list = np.array([1.0, 1.0])             # Discrimination
    subject_model.b_list = np.array([0.1, 0.1])             # Guessing
    
    dual_model = DualIRT(question_model=question_model, subject_model=subject_model)
    
    # Test data
    test_data = {
        'user_id': [0, 1, 2],
        'question_id': [0, 0, 1]
    }
    
    # Test with equal weighting
    preds, probs = dual_model.predict_probabilities(test_data, weight=0.5)
    
    # Check that predictions and probabilities are lists of correct length
    assert len(preds) == 3, f"Expected 3 predictions, got {len(preds)}"
    assert len(probs) == 3, f"Expected 3 probabilities, got {len(probs)}"
    
    # Check that probabilities are in valid range
    for prob in probs:
        assert 0 <= prob <= 1, f"Probability {prob} not in range [0, 1]"
    
    # Test with extreme weights
    preds_q_only, probs_q_only = dual_model.predict_probabilities(test_data, weight=1.0)
    preds_s_only, probs_s_only = dual_model.predict_probabilities(test_data, weight=0.0)
    
    # With weight=1.0, should match question model only
    q_preds, q_probs = question_model.predict_probabilities(test_data)
    assert probs_q_only == q_probs, "Weight=1.0 should match question model exactly"
    
    # With weight=0.0, should match subject model only
    s_preds, s_probs = subject_model.predict_probabilities(test_data)
    assert probs_s_only == s_probs, "Weight=0.0 should match subject model exactly"

def test_dual_irt_evaluate():
    """Test evaluation function."""
    # Simple mock models
    question_model = QuestionIRT(num_students=2, num_questions=2)
    question_subjects = {0: [0], 1: [0]}
    subject_model = SubjectIRT(num_students=2, question_subjects=question_subjects)
    
    dual_model = DualIRT(question_model=question_model, subject_model=subject_model)
    
    # Test data with known labels
    test_data = {
        'user_id': [0, 1],
        'question_id': [0, 1],
        'is_correct': [1, 0]
    }
    
    # Should return a valid accuracy
    accuracy = dual_model.evaluate(test_data, weight=0.5)
    
    assert 0 <= accuracy <= 1, f"Accuracy {accuracy} not in range [0, 1]"
    assert isinstance(accuracy, float), f"Accuracy should be float, got {type(accuracy)}"

def test_dual_irt_weight_range():
    """Test weight range evaluation."""
    # Simple mock models
    question_model = QuestionIRT(num_students=2, num_questions=2)
    question_subjects = {0: [0], 1: [0]}
    subject_model = SubjectIRT(num_students=2, question_subjects=question_subjects)
    
    dual_model = DualIRT(question_model=question_model, subject_model=subject_model)
    
    # Test data
    test_data = {
        'user_id': [0, 1],
        'question_id': [0, 1],
        'is_correct': [1, 0]
    }
    
    # Test with custom weight range
    weights = [0.0, 0.5, 1.0]
    returned_weights, accuracies = dual_model.evaluate_weight_range(test_data, weights)
    
    assert returned_weights == weights, "Returned weights should match input weights"
    assert len(accuracies) == len(weights), "Should have accuracy for each weight"
    
    for acc in accuracies:
        assert 0 <= acc <= 1, f"Accuracy {acc} not in range [0, 1]"

if __name__ == "__main__":
    test_dual_irt_initialization()
    test_dual_irt_predict_probabilities()
    test_dual_irt_evaluate()
    test_dual_irt_weight_range()
    print("All tests passed!")