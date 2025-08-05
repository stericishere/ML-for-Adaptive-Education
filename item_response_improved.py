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
    return 1 / (1 + np.exp(-x))


def neg_log_likelihood(data, theta, beta, a_list, b_list):
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
        #a's and b's depend on j index
        a = a_list[quest_id]
        b = b_list[quest_id]

        diff = theta[user_id] - beta[quest_id]
        sig = sigmoid(a * diff) #modify sigma based on new formula

        #modify log_liklihood based on new formula
        log_lklihood += c * np.log(b + (1-b) * sig + 1e-9) + (1 - c) * np.log(1 - b - (1-b) * sig + 1e-9)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, a_list, b_list):
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
        a = a_list[quest_id]
        b = b_list[quest_id]

        #computing grad based on formula
        diff = theta[user_id] - beta[quest_id]
        sig = sigmoid(a * diff)

        first_term = (c / (b + (1-b) * sig)) * sig * (1 - sig) * a
        second_term = ( (1-c) / (1- b - (1-b) * sig) ) * sig * (1-sig) * a
        grad_theta[user_id] += first_term + second_term

    new_theta = theta + lr * grad_theta

    grad_beta = np.zeros_like(beta)

    # now we update beta using the updated theta
    for i in range(len(question_ids)):
        #Getting the id's and c values
        user_id = user_ids[i]
        quest_id = question_ids[i]
        c = is_correct[i]
        a = a_list[quest_id]
        b = b_list[quest_id]

        #computing grad based on formula
        diff = new_theta[user_id] - beta[quest_id]
        sig = sigmoid(a * diff)
        first_term = (c / (b + (1-b) * sig) ) * sig * (1-sig) * (-a)
        second_term = (  (1-c)  /  ( 1 - b - (1-b) * sig ) ) * sig * (1-sig) * (-a)
        grad_beta[quest_id] += first_term + second_term

    new_beta = beta + lr * grad_beta

    grad_a_list = np.zeros_like(a_list)

    for i in range(len(question_ids)):
        user_id = user_ids[i]
        quest_id = question_ids[i]
        c = is_correct[i]
        a = a_list[quest_id]
        b = b_list[quest_id]

        diff = new_theta[user_id] - new_beta[quest_id]
        sig = sigmoid(a * diff)
        first_term = ( c / (b + (1-b) * sig) ) * sig * (1-sig) * (1-b) * diff
        second_term = ( (1-c) / ( 1 - b - (1-b) * sig ) ) * sig * (1-sig) * (b-1) * diff
        grad_a_list[quest_id] += first_term + second_term

    new_a_list = a_list + (lr / 10) * grad_a_list #smaller learning rate for a_i's

    grad_b_list = np.zeros_like(b_list)

    for i in range(len(question_ids)):
        user_id = user_ids[i]
        quest_id = question_ids[i]
        c = is_correct[i]
        a = new_a_list[quest_id]
        b = b_list[quest_id]
        diff = new_theta[user_id] - new_beta[quest_id]
        sig = sigmoid(a * diff)

        first_term = ( c / (b + (1-b) * sig) ) * (1-sig)
        second_term = ( (1-c) / (1 - b - (1-b) * sig) ) * sig

        grad_b_list[quest_id] += first_term + second_term
    new_b_list = b_list + (lr / 25) * grad_b_list # even smaller learning rate for b_i's





    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return new_theta, new_beta, new_a_list, new_b_list


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
    a_list = np.ones(max(data['question_id']) + 1)  # starting a_i factors set to 1
    b_list = np.ones(max(data['question_id']) + 1) * 0.0005  # non zero reasonable starting guessing probability

    val_acc_lst = []
    train_nll = []
    val_nll = []


    for i in range(iterations):
        #add training neg log likelihoods to list
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, a_list=a_list, b_list=b_list)
        train_nll.append(neg_lld)
        #add validation neg log liklihoods to list
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta, a_list=a_list, b_list=b_list)
        val_nll.append(neg_lld_val)

        #add validation accuracies to list
        score = evaluate(data=val_data, theta=theta, beta=beta, a_list=a_list, b_list=b_list)
        val_acc_lst.append(score)



        theta, beta, a_list, b_list = update_theta_beta(data, lr, theta, beta, a_list, b_list) #update params
        b_list = np.clip(b_list, 0.0, 0.35) #makes sure the guessing rate stays reasonable


    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nll, val_nll, a_list, b_list


def evaluate(data, theta, beta, a_list, b_list):
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

        theta_i = theta[u]
        beta_j = beta[q]
        a_j = a_list[q]
        b_j = b_list[q]

        diff = theta_i - beta_j
        sig = sigmoid(a_j * diff)

        p_correct = b_j + (1 - b_j) * sig
        pred.append(p_correct >= 0.5)

    return np.mean(np.array(pred) == np.array(data["is_correct"]))

def prob_label_outputs(training_data, test_data):
    theta, beta, _, _, _, a_list, b_list = irt(training_data, test_data, 0.00001, 50000)

    # predictions and probs for the training set
    train_preds = []
    train_probs = []
    for i in range(len(training_data["user_id"])):
        u = training_data["user_id"][i]
        q = training_data["question_id"][i]

        a = a_list[q]
        b = b_list[q]
        x = theta[u] - beta[q]
        sig = sigmoid(a * x)
        p = b + (1 - b) * sig

        train_probs.append(p)
        train_preds.append(p >= 0.5)

    # predictions and probs for test set
    test_preds = []
    test_probs = []
    for i in range(len(test_data["user_id"])):
        u = test_data["user_id"][i]
        q = test_data["question_id"][i]

        a = a_list[q]
        b = b_list[q]
        x = theta[u] - beta[q]
        sig = sigmoid(a * x)
        p = b + (1 - b) * sig

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
    lr = 0.0001
    iterations = 5000

    theta, beta, val_acc_lst, train_nll, val_nll, a_list, b_list = irt(train_data, val_data, lr, iterations)

    print("Final Validation Accuracy: " + str(val_acc_lst[-1]) + "\n")
    test_acc = evaluate(test_data, theta, beta, a_list, b_list)
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