import numpy as np
import matplotlib.pyplot as plt
from item_response import irt, sigmoid
from utils import load_train_csv, load_valid_csv

def plot_item_response_curves(beta, questions=None):
    """Plot P(correct) vs theta for selected questions using learned beta."""
    theta_range = np.linspace(-4, 4, 100)
    
    if questions is None:
        questions = [
            np.argmin(beta),           # Easiest question (lowest β)
            np.argsort(beta)[len(beta)//2],  # Medium difficulty
            np.argmax(beta)            # Hardest question (highest β)
        ]

    for q in questions:
        p_correct = sigmoid(theta_range - beta[q])
        plt.plot(theta_range, p_correct, label=f"Question {q} (β={beta[q]:.2f})")

    plt.xlabel("θ (Student Ability)")
    plt.ylabel("P (Correct)")
    plt.title("Item Response Curves (1PL Base Model)")
    plt.legend()
    plt.grid(True)
    plt.savefig("irt_curve_1pl.png", dpi=300)
    plt.show()

def main():
    # Load data
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")

    # Train IRT model
    theta, beta, _, _, _ = irt(train_data, val_data, lr=0.001, iterations=500)

    # Plot IRCs for selected questions
    plot_item_response_curves(beta)

if __name__ == "__main__":
    main()
