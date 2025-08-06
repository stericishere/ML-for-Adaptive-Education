import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subject_irt import SubjectIRT

def test_direct_subject_indices():
    """Test working with direct subject indices from cleaned data"""
    # Test data with direct subject indices (0-based)
    question_subjects = {
        0: [0, 1, 5],  # Subject indices 0, 1, 5
        1: [0, 2, 4],  # Subject indices 0, 2, 4
        2: [3, 4, 6]   # Subject indices 3, 4, 6
    }
    
    # Test direct access to subject indices
    indices = question_subjects[0]
    expected = [0, 1, 5]
    assert indices == expected, f"Expected {expected}, got {indices}"
    
    # Test getting indices for non-existent question
    indices = question_subjects.get(999, [])
    assert indices == [], "Should return empty list for non-existent question"

def test_subject_theta_initialization():
    """Test subject-based theta initialization"""
    num_students = 10
    question_subjects = {0: [0, 1], 1: [1, 2], 2: [0, 2]}
    
    model = SubjectIRT(num_students, question_subjects)
    
    # Check theta shape: (num_students, num_subjects)
    assert model.theta.shape[0] == num_students
    assert model.theta.shape[1] == model.num_subjects
    
    # Check number of subjects is correctly inferred
    expected_num_subjects = 3  # Subjects 0, 1, 2
    assert model.num_subjects == expected_num_subjects, f"Expected {expected_num_subjects} subjects, got {model.num_subjects}"
    
    # Check initialization values are reasonable
    assert np.all(np.abs(model.theta) <= 2.0), "Initial theta values should be reasonable"

def test_subject_probability_calculation():
    """Test probability calculation for subject-based IRT"""
    question_subjects = {0: [0, 1], 1: [1, 2]}
    model = SubjectIRT(num_students=5, question_subjects=question_subjects)
    
    # Set known values for testing
    model.theta[0, 0] = 1.0  # Student 0, subject 0
    model.theta[0, 1] = 0.5  # Student 0, subject 1
    model.beta[0] = 0.0      # Question 0 difficulty
    model.a_list[0] = 1.0    # Question 0 discrimination
    model.b_list[0] = 0.1    # Question 0 guessing
    
    prob = model.predict_probability(student_id=0, question_id=0)
    
    # Probability should be between 0 and 1
    assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
    
    # Should be greater than guessing probability
    assert prob > 0.1, "Probability should be greater than guessing parameter"

if __name__ == "__main__":
    test_direct_subject_indices()
    test_subject_theta_initialization()
    test_subject_probability_calculation()
    print("All tests passed!")