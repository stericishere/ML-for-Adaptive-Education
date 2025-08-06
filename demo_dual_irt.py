"""
Demo script showing how to use the DualIRT class to combine 
QuestionIRT and SubjectIRT models with different weights.
"""

from question_irt import QuestionIRT
from subject_irt import SubjectIRT, load_question_subjects
from DualIRT import DualIRT
from utils import load_train_csv, load_valid_csv, load_public_test_csv
import numpy as np
import matplotlib.pyplot as plt

def create_sample_models():
    """Create and train sample models for demonstration."""
    print("Creating sample models...")
    
    # Load some data for training
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    
    num_students = max(train_data['user_id']) + 1
    num_questions = max(train_data['question_id']) + 1
    
    # Create and train question model (small training for demo)
    print("Training QuestionIRT model...")
    question_model = QuestionIRT(num_students, num_questions)
    question_model.fit(train_data, val_data, lr=0.001, iterations=50)
    
    # Create and train subject model
    print("Training SubjectIRT model...")
    question_subjects = load_question_subjects("./data/clean_question_meta.csv")
    subject_model = SubjectIRT(num_students, question_subjects)
    subject_model.fit(train_data, val_data, lr=0.001, iterations=50)
    
    return question_model, subject_model

def demo_dual_irt():
    """Demonstrate the DualIRT functionality."""
    
    # Option 1: Use pre-trained models if they exist
    try:
        print("Loading pre-trained models...")
        dual_model = DualIRT()
        print("Successfully loaded pre-trained models!")
    except FileNotFoundError:
        print("Pre-trained models not found. Creating sample models...")
        question_model, subject_model = create_sample_models()
        dual_model = DualIRT(question_model=question_model, subject_model=subject_model)
    
    # Load test data
    val_data = load_valid_csv("./data")
    
    print("\n" + "="*60)
    print("DUAL IRT MODEL DEMONSTRATION")
    print("="*60)
    
    # Test different weights
    weights_to_test = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    print(f"Testing weights: {weights_to_test}")
    print("-" * 60)
    
    accuracies = []
    for weight in weights_to_test:
        accuracy = dual_model.evaluate(val_data, weight=weight)
        accuracies.append(accuracy)
        
        if weight == 0.0:
            print(f"Weight {weight:.1f} (Subject only):  Accuracy = {accuracy:.4f}")
        elif weight == 1.0:
            print(f"Weight {weight:.1f} (Question only): Accuracy = {accuracy:.4f}")
        else:
            print(f"Weight {weight:.1f} (Mixed):        Accuracy = {accuracy:.4f}")
    
    # Find best weight
    best_idx = np.argmax(accuracies)
    best_weight = weights_to_test[best_idx]
    best_accuracy = accuracies[best_idx]
    
    print("-" * 60)
    print(f"BEST PERFORMANCE:")
    print(f"Weight: {best_weight:.1f}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    # Show individual vs combined performance
    print(f"\nCOMPARISON:")
    print(f"Question Model Alone:  {accuracies[-1]:.4f}")
    print(f"Subject Model Alone:   {accuracies[0]:.4f}")
    print(f"Best Combined Model:   {best_accuracy:.4f}")
    
    improvement_over_question = best_accuracy - accuracies[-1]
    improvement_over_subject = best_accuracy - accuracies[0]
    
    print(f"\nIMPROVEMENT:")
    print(f"vs Question Model: {improvement_over_question:+.4f}")
    print(f"vs Subject Model:  {improvement_over_subject:+.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(weights_to_test, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7, 
                label=f'Best accuracy: {best_accuracy:.4f} at weight {best_weight:.1f}')
    plt.xlabel('Weight (1.0 = Question only, 0.0 = Subject only)')
    plt.ylabel('Accuracy')
    plt.title('DualIRT Performance vs Weight')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-0.05, 1.05)
    
    # Add annotations for extremes
    plt.annotate('Subject Only', xy=(0, accuracies[0]), xytext=(0.1, accuracies[0]),
                arrowprops=dict(arrowstyle='->', alpha=0.7))
    plt.annotate('Question Only', xy=(1, accuracies[-1]), xytext=(0.9, accuracies[-1]),
                arrowprops=dict(arrowstyle='->', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return dual_model, best_weight

if __name__ == "__main__":
    dual_model, best_weight = demo_dual_irt()
    
    print(f"\nDemonstration complete!")
    print(f"The DualIRT model combines predictions from both QuestionIRT and SubjectIRT")
    print(f"using a weighted average. The optimal weight found was {best_weight:.1f}")
    
    if best_weight == 0.0:
        print("This suggests the Subject model performs better on this dataset.")
    elif best_weight == 1.0:
        print("This suggests the Question model performs better on this dataset.")
    else:
        print("This suggests that combining both models improves performance!")