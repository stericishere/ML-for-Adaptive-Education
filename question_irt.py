from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_cleaned_subject_meta_csv
)
import numpy as np
import matplotlib.pyplot as plt
import pickle

def sigmoid(x):
    """Apply sigmoid function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class QuestionIRT:
    """3-parameter IRT model for question-based learning."""
    
    def __init__(self, num_students, num_questions):
        """Initialize Question IRT model.

        Args:
            num_students: Number of students
            num_questions: Number of questions
        """
        self.num_students = num_students
        self.num_questions = num_questions

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Student abilities
        self.theta = np.zeros(self.num_students)

        # Question difficulties
        self.beta = np.zeros(self.num_questions)

        # Question discrimination parameters
        self.a_list = np.ones(self.num_questions)

        # Question guessing parameters
        self.b_list = np.full(self.num_questions, 0.0005)

    def predict_probability(self, student_id, question_id):
        """Predict probability of correct answer for student-question pair.

        Args:
            student_id: ID of the student
            question_id: ID of the question

        Returns:
            float: Probability of correct answer
        """
        a = self.a_list[question_id]
        b = self.b_list[question_id]
        beta = self.beta[question_id]
        theta = self.theta[student_id]

        diff = theta - beta
        sig = sigmoid(a * diff)

        probability = b + (1 - b) * sig
        return probability

    def neg_log_likelihood(self, data):
        """Compute the negative log-likelihood.

        Args:
            data: A dictionary {user_id: list, question_id: list, is_correct: list}
            
        Returns:
            float: Negative log-likelihood
        """
        log_likelihood = 0.0
        
        for i in range(len(data["user_id"])):
            user_id = data["user_id"][i]
            quest_id = data["question_id"][i]
            c = data["is_correct"][i]
            
            # Get parameters for this question
            a = self.a_list[quest_id]
            b = self.b_list[quest_id]

            diff = self.theta[user_id] - self.beta[quest_id]
            sig = sigmoid(a * diff)

            # Compute log likelihood using 3PL formula
            prob_correct = b + (1-b) * sig
            prob_correct = np.clip(prob_correct, 1e-9, 1 - 1e-9)  # Avoid log(0)

            if c:
                log_likelihood += np.log(prob_correct)
            else:
                log_likelihood += np.log(1 - prob_correct)

        return -log_likelihood

    def update_parameters(self, data, lr):
        """Update theta and beta using gradient descent.

        Args:
            data: A dictionary {user_id: list, question_id: list, is_correct: list}
            lr: float learning rate
            
        Returns:
            None (updates parameters in place)
        """
        user_ids = data["user_id"]
        question_ids = data["question_id"]
        is_correct = data["is_correct"]

        grad_theta = np.zeros_like(self.theta)

        # First pass: update theta
        for i in range(len(user_ids)):
            user_id = user_ids[i]
            quest_id = question_ids[i]
            c = is_correct[i]
            a = self.a_list[quest_id]
            b = self.b_list[quest_id]
            
            diff = self.theta[user_id] - self.beta[quest_id]
            sig = sigmoid(a * diff)
            prob_correct = b + (1-b) * sig
            prob_correct = np.clip(prob_correct, 1e-9, 1-1e-9) # Avoid division by zero

            common_factor = (c - prob_correct) / (prob_correct * (1-prob_correct))
            grad_theta[user_id] += common_factor * (1-b) * sig * (1-sig) * a

        new_theta = self.theta + lr * grad_theta

        grad_beta = np.zeros_like(self.beta)

        # Second pass: update beta using updated theta
        for i in range(len(question_ids)):
            user_id = user_ids[i]
            quest_id = question_ids[i]
            c = is_correct[i]
            a = self.a_list[quest_id]
            b = self.b_list[quest_id]
            
            diff = new_theta[user_id] - self.beta[quest_id]
            sig = sigmoid(a * diff)
            prob_correct = b + (1-b) * sig
            prob_correct = np.clip(prob_correct, 1e-9, 1-1e-9)

            # CORRECTED GRADIENT CALCULATION
            common_factor = (c - prob_correct) / (prob_correct * (1-prob_correct))
            grad_beta[quest_id] += common_factor * (1-b) * sig * (1-sig) * (-a)


        new_beta = self.beta + lr * grad_beta

        grad_a_list = np.zeros_like(self.a_list)

        # Third pass: update discrimination parameters
        for i in range(len(question_ids)):
            user_id = user_ids[i]
            quest_id = question_ids[i]
            c = is_correct[i]
            a = self.a_list[quest_id]
            b = self.b_list[quest_id]
            
            diff = new_theta[user_id] - new_beta[quest_id]
            sig = sigmoid(a * diff)
            prob_correct = b + (1-b) * sig
            prob_correct = np.clip(prob_correct, 1e-9, 1-1e-9)

            # CORRECTED GRADIENT CALCULATION
            common_factor = (c - prob_correct) / (prob_correct * (1-prob_correct))
            grad_a_list[quest_id] += common_factor * (1-b) * sig * (1-sig) * diff

        new_a_list = self.a_list + (lr / 10) * grad_a_list

        grad_b_list = np.zeros_like(self.b_list)

        # Fourth pass: update guessing parameters
        for i in range(len(question_ids)):
            user_id = user_ids[i]
            quest_id = question_ids[i]
            c = is_correct[i]
            a = new_a_list[quest_id]
            b = self.b_list[quest_id]
            
            diff = new_theta[user_id] - new_beta[quest_id]
            sig = sigmoid(a * diff)
            prob_correct = b + (1-b) * sig
            prob_correct = np.clip(prob_correct, 1e-9, 1-1e-9)

            # CORRECTED GRADIENT CALCULATION
            common_factor = (c - prob_correct) / (prob_correct * (1-prob_correct))
            grad_b_list[quest_id] += common_factor * (1 - sig)
            
        new_b_list = self.b_list + (lr / 25) * grad_b_list

        # Update parameters
        self.theta = new_theta
        self.beta = new_beta
        self.a_list = new_a_list
        self.b_list = new_b_list

        # Clip parameters to reasonable ranges
        self.b_list = np.clip(self.b_list, 0.0, 0.35)

    def evaluate(self, data):
        """Evaluate the model given data and return the accuracy.
        
        Args:
            data: A dictionary {user_id: list, question_id: list, is_correct: list}

        Returns:
            float: Accuracy
        """
        pred = []
        for i, q in enumerate(data["question_id"]):
            u = data["user_id"][i]

            theta_i = self.theta[u]
            beta_j = self.beta[q]
            a_j = self.a_list[q]
            b_j = self.b_list[q]

            diff = theta_i - beta_j
            sig = sigmoid(a_j * diff)

            p_correct = b_j + (1 - b_j) * sig
            pred.append(p_correct >= 0.5)

        return np.mean(np.array(pred) == np.array(data["is_correct"]))

    def fit(self, train_data, val_data, lr=0.0001, iterations=5000):
        """Train the IRT model.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary  
            lr: Learning rate
            iterations: Number of training iterations
            
        Returns:
            tuple: (train_nll_history, val_nll_history, val_acc_history)
        """
        val_acc_lst = []
        train_nll = []
        val_nll = []

        for i in range(iterations):
            # Compute metrics
            neg_lld = self.neg_log_likelihood(train_data)
            train_nll.append(neg_lld)

            neg_lld_val = self.neg_log_likelihood(val_data)
            val_nll.append(neg_lld_val)

            score = self.evaluate(val_data)
            val_acc_lst.append(score)

            # Update parameters
            self.update_parameters(train_data, lr)

            # Print progress

            print(f"Iteration {i + 1}: Train NLL={neg_lld:.4f}, "
                    f"Val NLL={neg_lld_val:.4f}, Val Acc={score:.4f}")

        return train_nll, val_nll, val_acc_lst

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

            a = self.a_list[q]
            b = self.b_list[q]
            x = self.theta[u] - self.beta[q]
            sig = sigmoid(a * x)
            p = b + (1 - b) * sig

            probs.append(p)
            preds.append(p >= 0.5)

        return preds, probs

def main():
    """Main function to train and evaluate question-based IRT model."""
    # Load data
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Initialize model
    num_students = max(train_data['user_id']) + 1
    num_questions = max(train_data['question_id']) + 1
    
    model = QuestionIRT(num_students, num_questions)
    
    print(f"Model initialized with {num_students} students and {num_questions} questions")

    # Train model
    lr = 0.001
    iterations = 300

    train_nll, val_nll, val_acc = model.fit(train_data, val_data, lr, iterations)

    # Final evaluation
    final_val_acc = model.evaluate(val_data)
    test_acc = model.evaluate(test_data)
    
    print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    model_save_path = "question_irt_model.pkl"
    with open(model_save_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved directly to {model_save_path}")
    
    # Plot training curves
    x_vals = list(range(1, iterations + 1))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)

    plt.scatter(x_vals, train_nll, color="blue", label="Train NLL", s=25)
    plt.scatter(x_vals, val_nll, color="red", label="Validation NLL", s=25)
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Training Curve of question IRT")
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