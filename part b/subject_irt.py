from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import pickle

def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def load_question_subjects(data_path="./data/clean_question_meta.csv"):
    """Load question subject mappings from CSV file.
    Data is already cleaned to contain subject indices directly.
    
    Returns:
        dict: {question_id: [subject_indices]}
    """
    df = pd.read_csv(data_path)
    question_subjects = {}
    
    for _, row in df.iterrows():
        question_id = row.iloc[0]  # First column is question ID
        subjects_str = row.iloc[1]  # Second column is subject indices string
        
        # Parse the string representation of list of indices
        try:
            subject_indices = ast.literal_eval(subjects_str)
            # Ensure it's a list
            if not isinstance(subject_indices, list):
                subject_indices = [subject_indices] if subject_indices is not None else []
            question_subjects[question_id] = subject_indices
        except (ValueError, SyntaxError):
            # Handle malformed entries
            question_subjects[question_id] = []
    
    return question_subjects

class SubjectIRT:
    """Subject-based 3-parameter IRT model."""
    
    def __init__(self, num_students, question_subjects=None, data_path="./data/clean_question_meta.csv"):
        """Initialize Subject IRT model.
        
        Args:
            num_students: Number of students
            question_subjects: dict {question_id: [subject_indices]} or None to load from file
            data_path: Path to question metadata CSV file
        """
        if question_subjects is None:
            question_subjects = load_question_subjects(data_path)
        
        self.question_subjects = question_subjects
        
        # Find number of subjects from the data
        all_subject_indices = set()
        for subjects in question_subjects.values():
            all_subject_indices.update(subjects)
        
        self.num_subjects = max(all_subject_indices) + 1 if all_subject_indices else 0
        self.num_students = num_students
        self.num_questions = max(question_subjects.keys()) + 1 if question_subjects else 0
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Subject-specific abilities: (num_students, num_subjects)
        self.theta = np.random.normal(0, 0.5, (self.num_students, self.num_subjects))
        
        # Question difficulties
        self.beta = np.random.normal(0, 1, self.num_questions)
        
        # Question discrimination parameters
        self.a_list = np.ones(self.num_questions)
        
        # Question guessing parameters
        self.b_list = np.full(self.num_questions, 0.1)
        
        # Track which subjects each student has encountered
        self.student_subject_counts = np.zeros((self.num_students, self.num_subjects))
    
    def predict_probability(self, student_id, question_id):
        """Predict probability of correct answer for student-question pair.
        
        Args:
            student_id: ID of the student
            question_id: ID of the question
            
        Returns:
            float: Probability of correct answer
        """
        if question_id not in self.question_subjects:
            return 0.5  # Default probability for unknown questions
        
        subject_indices = self.question_subjects[question_id]
        
        if not subject_indices:
            return 0.5  # Default probability if no subjects
        
        # Aggregate ability across relevant subjects (mean)
        subject_abilities = self.theta[student_id, subject_indices]
        mean_ability = np.mean(subject_abilities)
        
        # 3PL IRT model
        a = self.a_list[question_id]
        b = self.b_list[question_id]
        beta = self.beta[question_id]
        
        diff = mean_ability - beta
        sig = sigmoid(a * diff)
        
        probability = b + (1 - b) * sig
        return probability
    
    def predict_probabilities(self, data):
        """Generate probability predictions for data.
        
        Args:
            data: Dictionary with user_id, question_id lists
            
        Returns:
            tuple: (predictions, probabilities)
        """
        preds = []
        probs = []
        
        for i in range(len(data["user_id"])):
            u = data["user_id"][i]
            q = data["question_id"][i]

            # Use the predict_probability method for consistency
            p = self.predict_probability(u, q)

            probs.append(p)
            preds.append(p >= 0.5)

        return preds, probs

    def neg_log_likelihood(self, data):
        """Compute negative log-likelihood for the data.
        
        Args:
            data: Dictionary with user_id, question_id, is_correct lists
            
        Returns:
            float: Negative log-likelihood
        """
        log_likelihood = 0.0
        
        for i in range(len(data["user_id"])):
            user_id = data["user_id"][i]
            question_id = data["question_id"][i]
            is_correct = data["is_correct"][i]

            prob = self.predict_probability(user_id, question_id)
            prob = np.clip(prob, 1e-9, 1 - 1e-9)  # Avoid log(0)

            if is_correct:
                log_likelihood += np.log(prob)
            else:
                log_likelihood += np.log(1 - prob)

        return -log_likelihood

    def update_parameters(self, data, lr):
        """Update model parameters using gradient descent.
        
        Args:
            data: Training data dictionary
            lr: Learning rate
        """
        # Initialize gradients
        grad_theta = np.zeros_like(self.theta)
        grad_beta = np.zeros_like(self.beta)
        grad_a = np.zeros_like(self.a_list)
        grad_b = np.zeros_like(self.b_list)
        
        # Count subject encounters for better initialization
        subject_encounter_counts = np.zeros((self.num_students, self.num_subjects))
        
        for i in range(len(data["user_id"])):
            user_id = data["user_id"][i]
            question_id = data["question_id"][i]
            is_correct = data["is_correct"][i]
            
            if question_id not in self.question_subjects:
                continue
                
            subject_indices = self.question_subjects[question_id]
            if not subject_indices:
                continue
            
            # Track subject encounters
            for subj_idx in subject_indices:
                subject_encounter_counts[user_id, subj_idx] += 1
            
            # Current prediction
            subject_abilities = self.theta[user_id, subject_indices]
            mean_ability = np.mean(subject_abilities)

            a = self.a_list[question_id]
            b = self.b_list[question_id]
            beta = self.beta[question_id]
            
            diff = mean_ability - beta
            sig = sigmoid(a * diff)
            prob = b + (1 - b) * sig
            prob = np.clip(prob, 1e-9, 1 - 1e-9)
            
            # Compute gradients
            if is_correct:
                grad_factor = 1 / prob
            else:
                grad_factor = -1 / (1 - prob)
            
            # Gradient components
            dsig_ddiff = sig * (1 - sig) * a
            dprob_dsig = (1 - b)
            dprob_da = (1 - b) * sig * (1 - sig) * diff
            dprob_db = 1 - sig
            
            # Update theta gradients (distributed across relevant subjects)
            for subj_idx in subject_indices:
                grad_theta[user_id, subj_idx] += grad_factor * dprob_dsig * dsig_ddiff / len(subject_indices)
            
            # Update other parameter gradients
            grad_beta[question_id] += grad_factor * dprob_dsig * (-dsig_ddiff)
            grad_a[question_id] += grad_factor * dprob_da
            grad_b[question_id] += grad_factor * dprob_db
        
        # Apply updates
        self.theta += lr * grad_theta
        self.beta += lr * grad_beta
        self.a_list += lr * grad_a / 10  # Smaller learning rate for discrimination
        self.b_list += lr * grad_b / 25  # Even smaller for guessing
        
        # Clip parameters to reasonable ranges
        self.b_list = np.clip(self.b_list, 0.0, 0.35)
        self.a_list = np.clip(self.a_list, 0.1, 3.0)
        
        # Handle unencountered subjects with regularization towards global mean
        self.student_subject_counts += subject_encounter_counts
        unencountered_mask = self.student_subject_counts == 0
        if np.any(unencountered_mask):
            global_mean = np.mean(self.theta[self.student_subject_counts > 0], axis=0)
            regularization_penalty = 0.01
            self.theta[unencountered_mask] += regularization_penalty * (global_mean - self.theta[unencountered_mask])
    
    def evaluate(self, data):
        """Evaluate model accuracy on data.
        
        Args:
            data: Dictionary with user_id, question_id, is_correct lists
            
        Returns:
            float: Accuracy
        """
        correct = 0
        total = 0

        for i in range(len(data["user_id"])):
            user_id = data["user_id"][i]
            question_id = data["question_id"][i]
            is_correct = data["is_correct"][i]

            prob = self.predict_probability(user_id, question_id)
            prediction = prob >= 0.5

            if prediction == is_correct:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0
    
    def fit(self, train_data, val_data, lr=0.001, iterations=1000):
        """Train the model.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            lr: Learning rate
            iterations: Number of training iterations
            
        Returns:
            tuple: (train_nll_history, val_nll_history, val_acc_history)
        """
        train_nll_history = []
        val_nll_history = []
        val_acc_history = []
        
        for iteration in range(iterations):
            # Compute metrics
            train_nll = self.neg_log_likelihood(train_data)
            val_nll = self.neg_log_likelihood(val_data)
            val_acc = self.evaluate(val_data)
            
            train_nll_history.append(train_nll)
            val_nll_history.append(val_nll)
            val_acc_history.append(val_acc)
            
            # Update parameters
            self.update_parameters(train_data, lr)
            
            # Print progress
            print(f"Iteration {iteration + 1}: Train NLL={train_nll:.4f}, "
                    f"Val NLL={val_nll:.4f}, Val Acc={val_acc:.4f}")
        
        return train_nll_history, val_nll_history, val_acc_history

def main():
    """Main function to train and evaluate subject-based IRT model."""
    # Load data
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    
    # Load question subjects
    question_subjects = load_question_subjects("./data/clean_question_meta.csv")
    
    # Initialize model
    num_students = max(train_data['user_id']) + 1
    model = SubjectIRT(num_students, question_subjects)
    
    print(f"Model initialized with {model.num_subjects} subjects and {num_students} students")
    print(f"Number of questions: {model.num_questions}")
    
    # Debug: check some question mappings
    sample_questions = list(question_subjects.keys())[:5]
    print("Sample question-subject mappings:")
    for q_id in sample_questions:
        subject_indices = question_subjects[q_id]
        print(f"  Question {q_id}: subject indices {subject_indices}")
    
    # Train model
    lr = 0.001
    iterations = 850
    
    train_nll, val_nll, val_acc = model.fit(train_data, val_data, lr, iterations)
    
    # Final evaluation
    final_val_acc = model.evaluate(val_data)
    test_acc = model.evaluate(test_data)
    
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    model_save_path = "subject_irt_model.pkl"
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved directly to {model_save_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_nll, label="Train NLL", color="blue")
    plt.plot(val_nll, label="Validation NLL", color="red")
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Training Curve for subject IRT")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_acc, label="Validation Accuracy", color="green")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()