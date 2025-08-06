import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from question_irt import QuestionIRT

def test_question_irt_initialization():
    """Test QuestionIRT initialization"""
    num_students = 100
    num_questions = 50
    
    model = QuestionIRT(num_students, num_questions)
    
    # Check parameter shapes
    assert model.theta.shape == (num_students,), f"Expected theta shape {(num_students,)}, got {model.theta.shape}"
    assert model.beta.shape == (num_questions,), f"Expected beta shape {(num_questions,)}, got {model.beta.shape}"
    assert model.a_list.shape == (num_questions,), f"Expected a_list shape {(num_questions,)}, got {model.a_list.shape}"
    assert model.b_list.shape == (num_questions,), f"Expected b_list shape {(num_questions,)}, got {model.b_list.shape}"
    
    # Check initial values
    assert np.all(model.theta == 0), "Initial theta should be zeros"
    assert np.all(model.beta == 0), "Initial beta should be zeros"
    assert np.all(model.a_list == 1), "Initial a_list should be ones"
    assert np.all(model.b_list == 0.0005), "Initial b_list should be 0.0005"

def test_probability_calculation():
    """Test probability calculation"""
    model = QuestionIRT(num_students=5, num_questions=10)
    
    # Set known values for testing
    model.theta[0] = 1.0      # Student 0 ability
    model.beta[0] = 0.0       # Question 0 difficulty  
    model.a_list[0] = 1.0     # Question 0 discrimination
    model.b_list[0] = 0.1     # Question 0 guessing
    
    prob = model.predict_probability(student_id=0, question_id=0)
    
    # Probability should be between 0 and 1
    assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
    
    # Should be greater than guessing probability
    assert prob > 0.1, "Probability should be greater than guessing parameter"
    
    # With positive ability and zero difficulty, should be quite high
    assert prob > 0.5, "With positive ability and zero difficulty, probability should be > 0.5"

def test_neg_log_likelihood():
    """Test negative log likelihood computation"""
    model = QuestionIRT(num_students=3, num_questions=3)
    
    # Simple test data
    test_data = {
        'user_id': [0, 1, 2, 0, 1],
        'question_id': [0, 0, 1, 1, 2],
        'is_correct': [1, 0, 1, 1, 0]
    }
    
    nll = model.neg_log_likelihood(test_data)
    
    # NLL should be positive
    assert nll > 0, f"Negative log likelihood should be positive, got {nll}"
    
    # Should be finite
    assert np.isfinite(nll), "NLL should be finite"

def test_evaluate():
    """Test evaluation function"""
    model = QuestionIRT(num_students=3, num_questions=3)
    
    # Simple test data
    test_data = {
        'user_id': [0, 1, 2],
        'question_id': [0, 0, 1],
        'is_correct': [1, 0, 1]
    }
    
    accuracy = model.evaluate(test_data)
    
    # Accuracy should be between 0 and 1
    assert 0 <= accuracy <= 1, f"Accuracy should be between 0 and 1, got {accuracy}"

def test_parameter_update():
    """Test parameter update"""
    model = QuestionIRT(num_students=3, num_questions=3)
    
    # Store initial parameters
    initial_theta = model.theta.copy()
    initial_beta = model.beta.copy()
    initial_a = model.a_list.copy()
    initial_b = model.b_list.copy()
    
    # Simple training data
    train_data = {
        'user_id': [0, 1, 2, 0, 1],
        'question_id': [0, 0, 1, 1, 2],
        'is_correct': [1, 0, 1, 1, 0]
    }
    
    # Update parameters
    model.update_parameters(train_data, lr=0.01)
    
    # Parameters should have changed (at least some of them)
    assert not np.array_equal(model.theta, initial_theta) or \
           not np.array_equal(model.beta, initial_beta) or \
           not np.array_equal(model.a_list, initial_a) or \
           not np.array_equal(model.b_list, initial_b), \
           "At least some parameters should have changed after update"
    
    # b_list should still be clipped to reasonable range
    assert np.all(model.b_list >= 0.0), "b_list should be >= 0"
    assert np.all(model.b_list <= 0.35), "b_list should be <= 0.35"

if __name__ == "__main__":
    test_question_irt_initialization()
    test_probability_calculation()
    test_neg_log_likelihood()
    test_evaluate()
    test_parameter_update()
    print("All tests passed!")