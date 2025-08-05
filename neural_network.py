import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # Apply first linear layer with sigmoid activation
        hidden = torch.sigmoid(self.g(inputs))
        # Apply second linear layer with sigmoid activation
        out = torch.sigmoid(self.h(hidden))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out
    
    # Output the predicted probabilities for each question
    def predict(self, input_tensor):
        """Return predicted probabilities for each question."""
        self.eval()
        with torch.no_grad():
            output = self.forward(input_tensor)
        return output


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    # Define the regularizer
    regularizer = 0.0

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # Lists to store training losses and validation accuracies
    train_losses = []
    valid_accs = []
    
    best_model = None
    best_valid_acc = 0.0

    # Training loop
    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            
            # Add L2 regularization if lamb > 0
            if lamb > 0:
                regularizer = model.get_weight_norm()
                loss += (lamb / 2) * regularizer
                
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        
        # Update best model if current validation accuracy is higher
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return train_losses, valid_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k = [10, 50, 100, 200, ]#500]
    model = AutoEncoder(num_question=train_matrix.shape[1], k=50)
    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 45
    # Set regularization for initial model
    lamb = 0.0
    # lamb = 0.1

    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    
    # Part (a): Evaluate the model on validation/test data
    print("Part (a): Evaluating the model with lamb=0.0 on validation/test data")
    # Next, evaluate your network on validation/test data
    valid_acc = evaluate(model, zero_train_matrix, valid_data)
    print(f"Validation accuracy: {valid_acc:.4f}")

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Part (c): Try different k values
    results = {}
    print("Part (c): Testing different k values...")
    for k in k:
        print(f"\nTraining with k={k}")
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
        train_losses, valid_accs = train(
            model, lr, lamb, train_matrix, zero_train_matrix, 
            valid_data, num_epoch
        )
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        test_acc = evaluate(model, zero_train_matrix, test_data)
        results[k] = {
            'valid_accuracy': valid_acc,
            'test_accuracy': test_acc,
            'train_losses': train_losses,
            'valid_accuracies': valid_accs
        }
    for i in results:
        print(f"k: {i}")
        print(f"Valid accuracy: {results[i]['valid_accuracy']:.4f}")
        print(f"Test accuracy: {results[i]['test_accuracy']:.4f}")
    
    k_star = max(result['test_accuracy'] for result in results.values())
    print(f"Best k: {k_star}")
    print(f"Best test accuracy: {results[k_star]['test_accuracy']:.4f}")
    
    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(results[k_star]['train_losses'], label='Training Loss')
    plt.plot(results[k_star]['valid_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title(f'Training and Validation Losses for k={k_star}')
    plt.legend()
    plt.show()
    
    # Part (e): Try different learning rates
    print("Part (e): Trying different λ...")
    lamb_list  = [0.001, 0.01, 0.1, 1.0]
    for lamb in lamb_list :
        print(f"\nTraining with λ={lamb}")
        model = AutoEncoder(num_question=train_matrix.shape[1], k=k_star)
        train_losses, valid_accs = train(
            model, lr, lamb, train_matrix, zero_train_matrix, 
            valid_data, num_epoch
        )
        valid_acc = evaluate(model, zero_train_matrix, valid_data)
        test_acc = evaluate(model, zero_train_matrix, test_data)
        results[lamb] = {
            'valid_accuracy': valid_acc,
            'test_accuracy': test_acc,
            'train_losses': train_losses,
            'valid_accuracies': valid_accs
        }
    print(results)
    lamb_star = max(result['test_accuracy'] for result in results.values())

    print(f"Best λ: {lamb_star}")
    print(f"Best test accuracy: {results[lamb_star]['test_accuracy']:.4f}")
    
    # Plot the training and validation losses
    plt.figure(figsize=(10, 5))

    # Evaluate the model on the test data
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
