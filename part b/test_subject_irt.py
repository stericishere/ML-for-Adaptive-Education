import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from subject_irt import SubjectIRT, create_multi_hot_encoder, load_question_subjects

def test_multi_hot_encoder():
    """Test multi-hot encoding functionality"""
    # Test data
    question_subjects = {
        0: [1, 2, 98],
        1: [1, 3, 96],
        2: [5, 8, 168]
    }
    
    encoder = create_multi_hot_encoder(question_subjects)
    
    # Test encoding for question 0
    encoded = encoder.encode_question(0)
    expected_subjects = {1, 2, 98}
    # Get the actual subject IDs from the encoder mapping
    actual_subject_indices = np.where(encoded == 1)[0]
    actual_subjects = {encoder.all_subjects[idx] for idx in actual_subject_indices}
    
    assert expected_subjects == actual_subjects, f"Expected {expected_subjects}, got {actual_subjects}"
    
    # Test that all questions have consistent encoding size
    sizes = [len(encoder.encode_question(q)) for q in question_subjects.keys()]
    assert len(set(sizes)) == 1, "All encoded vectors should have same size"

def test_subject_theta_initialization():
    """Test subject-based theta initialization"""
    num_students = 10
    question_subjects = {0: [1, 2], 1: [2, 3], 2: [1, 3]}
    
    model = SubjectIRT(num_students, question_subjects)
    
    # Check theta shape: (num_students, num_subjects)
    assert model.theta.shape[0] == num_students
    assert model.theta.shape[1] == model.num_subjects
    
    # Check initialization values are reasonable
    assert np.all(np.abs(model.theta) <= 2.0), "Initial theta values should be reasonable"

def test_subject_probability_calculation():
    """Test probability calculation for subject-based IRT"""
    question_subjects = {0: [1, 2], 1: [2, 3]}
    model = SubjectIRT(num_students=5, question_subjects=question_subjects)
    
    # Set known values for testing
    model.theta[0, 0] = 1.0  # Student 0, subject 1
    model.theta[0, 1] = 0.5  # Student 0, subject 2
    model.beta[0] = 0.0      # Question 0 difficulty
    model.a_list[0] = 1.0    # Question 0 discrimination
    model.b_list[0] = 0.1    # Question 0 guessing
    
    prob = model.predict_probability(student_id=0, question_id=0)
    
    # Probability should be between 0 and 1
    assert 0 <= prob <= 1, f"Probability should be between 0 and 1, got {prob}"
    
    # Should be greater than guessing probability
    assert prob > 0.1, "Probability should be greater than guessing parameter"

if __name__ == "__main__":
    test_multi_hot_encoder()
    test_subject_theta_initialization()
    test_subject_probability_calculation()
    print("All tests passed!")