from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        quest_id = data["question_id"][i]
        c = data["is_correct"][i]
        diff = theta[user_id] - beta[quest_id]
        sig = sigmoid(diff)
        log_lklihood += c * np.log(sig + 1e-9) + (1 - c) * np.log(1 - sig + 1e-9)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # first we update theta
    user_ids = data["user_id"]
    question_ids = data["question_id"]
    is_correct = data["is_correct"]

    grad_theta = np.zeros_like(theta)

    # First pass: update theta
    for i in range(len(user_ids)):
        #getting id's and c values
        user_id = user_ids[i]
        quest_id = question_ids[i]
        c = is_correct[i]

        #computing grad based on formula
        diff = theta[user_id] - beta[quest_id]
        sig = sigmoid(diff)
        grad_theta[user_id] += c - sig

    new_theta = theta + lr * grad_theta

    grad_beta = np.zeros_like(beta)

    # now we update beta using the updated theta
    for i in range(len(question_ids)):
        #Getting the id's and c values
        user_id = user_ids[i]
        quest_id = question_ids[i]
        c = is_correct[i]

        #computing grad based on formula
        diff = new_theta[user_id] - beta[quest_id]
        sig = sigmoid(diff)
        grad_beta[quest_id] += sig - c

    new_beta = beta + lr * grad_beta


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(max(data['user_id']) + 1)  # plus 1 because id's start from 0
    beta = np.zeros(max(data['question_id']) + 1)

    val_acc_lst = []
    train_nll = []
    val_nll = []


    for i in range(iterations):
        #add training neg log likelihoods to list
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_nll.append(neg_lld)
        #add validation neg log liklihoods to list
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_nll.append(neg_lld_val)

        #add validation accuracies to list
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)



        theta, beta = update_theta_beta(data, lr, theta, beta) #update params


    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nll, val_nll


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def prob_label_outputs(training_data, test_data):
    #training theta and beta on the training data, with best lr and iterations found
    theta, beta, _, _, _ = irt(training_data, test_data, 0.001, 500)



    # getting the validation predictions and probabilities
    train_preds = []
    train_probs = []
    for i in range(len(training_data["user_id"])):
        u = training_data["user_id"][i]
        q = training_data["question_id"][i]
        x = theta[u] - beta[q]
        p = sigmoid(x)
        train_probs.append(p)
        train_preds.append(p >= 0.5)

    # getting the test predictions and probabilities
    test_preds = []
    test_probs = []
    for i in range(len(test_data["user_id"])):
        u = test_data["user_id"][i]
        q = test_data["question_id"][i]
        x = theta[u] - beta[q]
        p = sigmoid(x)
        test_probs.append(p)
        test_preds.append(p >= 0.5)

    return train_preds, train_probs, test_preds, test_probs



def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")


    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.001
    iterations = 500

    theta, beta, val_acc_lst, train_nll, val_nll = irt(train_data, val_data, lr, iterations)

    print("Final Validation Accuracy: " + str(val_acc_lst[-1]) + "\n")
    test_acc = evaluate(test_data, theta, beta)
    print("Test Accuracy: " + str(test_acc))

    x_vals = list(range(1, iterations + 1))
    plt.scatter(x_vals, train_nll, color="blue", label="Train NLL", s=25)
    plt.scatter(x_vals, val_nll, color="red", label="Validation NLL", s=25)
    plt.xlabel("Iteration #")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Training Curve")
    plt.legend()
    plt.show()
    # #####################################################################
    # #                       END OF YOUR CODE                            #
    # #####################################################################
    #
    # #####################################################################
    # # TODO:                                                             #
    # # Implement part (d)                                                #
    # #####################################################################
    # theta_range = np.linspace(-4, 4, 100)
    # questions = [5, 10, 20]
    #
    # for q in questions:
    #     p = sigmoid(theta_range - beta[q])
    #     plt.plot(theta_range, p, label=f"Question {q}")
    #
    # plt.xlabel("Theta (Ability)")
    # plt.ylabel("P(Correct)")
    # plt.title("Item Response Curves for Selected Questions")
    # plt.legend()
    # plt.show()
    # #####################################################################
    # #                       END OF YOUR CODE                            #
    # #####################################################################

    # train_preds, train_probs, test_preds, test_probs = prob_label_outputs(train_data, val_data)
    # print(train_preds)
    # print(train_probs)
    # print(test_preds)
    # print(test_probs)


if __name__ == "__main__":
    main()
