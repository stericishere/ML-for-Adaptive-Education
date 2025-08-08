import numpy as np
import matplotlib.pyplot as plt
import pickle
from question_irt import QuestionIRT

# === Sigmoid function ===
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# === Plotting function ===
def plot_3pl_curves(a_list, b_list, beta, question_ids=[1165,34,1410]):     #fixing question ids for comparison
    """Plot 3PL item response curves for given question IDs."""
    theta_range = np.linspace(-4, 4, 200)

    for q_id in question_ids:
        a = a_list[q_id]
        b = b_list[q_id]
        beta_q = beta[q_id]

        prob = b + (1 - b) * sigmoid(a * (theta_range - beta_q))
        label = f"Q{q_id} | β={beta_q:.2f}, a={a:.2f}, b={b:.2f}"
        plt.plot(theta_range, prob, label=label)

    plt.xlabel("θ (Student Ability)")
    plt.ylabel("P (Correct)")
    plt.title("Item Response Curves (3PL Question Model)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("irt_curve_3pl_question.png", dpi=300)
    plt.show()

def main():
    # Load saved QuestionIRT model
    with open("question_irt_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Extract parameters
    a_list = model.a_list
    b_list = model.b_list
    beta = model.beta

    # Plot curves
    plot_3pl_curves(a_list, b_list, beta)

if __name__ == "__main__":
    main()
