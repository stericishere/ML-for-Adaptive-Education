import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    '''(Question 1a)
    Perform user-based KNN and evaluate on validation data 
    '''
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy, k = {}: {}".format(k, acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    '''(Question 1c)
    Perform item-based KNN and evaluate on validation data 
    '''
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    # Transpose the sparse matrix to get question-based similarity
    mat_transposed = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat_transposed)
    print("Validation Accuracy, k = {}: {}".format(k, acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    # Load data
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Given
    ks = [1, 6, 11, 16, 21, 26]

    # (Question 1a) Try user-based KNN for different k
    print('User-based KNN:')
    user_acc = []
    for k in ks:
      acc = knn_impute_by_user(sparse_matrix, val_data, k)
      user_acc.append(acc)
    
    # Plot accuracy vs. k
    plt.plot(ks, user_acc)
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('[User-based] KNN')
    plt.show()

    # (Question 1b) Select optimal k* and evaluate on test set
    k_star_user = ks[np.argmax(user_acc)]   # optimal k*
    print("[User-based] Optimal k*: {}".format(k_star_user))
    knn_user_model = KNNImputer(n_neighbors=k_star_user)    # compute imputation values based on user similarity
    mat_user = knn_user_model.fit_transform(sparse_matrix)  # fits the KNN model on the sparse matrix and fills in all NaN values
    user_test_acc = sparse_matrix_evaluate(test_data, mat_user) # compares predicted (from mat_user) to true labels in test_data.csv
    print("[User-based] Test Accuracy: {}".format(user_test_acc))   # result: test accuracy as a float between 0 and 1

    # (Question 1c-a) Try item-based KNN for different k
    print('Item-based KNN:')
    item_acc = []
    for k in ks:
      acc = knn_impute_by_item(sparse_matrix, val_data, k)
      item_acc.append(acc)
    
    # Plot accuracy vs. k
    plt.plot(ks, item_acc)
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('[Item-based] KNN')
    plt.show()

    # (Question 1c-b) Select optimal k* and evaluate on test set
    k_star_item = ks[np.argmax(item_acc)]
    print("[Item-based] Optimal k*: {}".format(k_star_item))
    knn_item_model = KNNImputer(n_neighbors=k_star_item)
    mat_item = knn_item_model.fit_transform(sparse_matrix.T).T  # transpose before fitting and transpose back
    item_test_acc = sparse_matrix_evaluate(test_data, mat_item)
    print("[Item-based] Test Accuracy: {}".format(item_test_acc))

    # (Question 1d) Compare user-based vs item-based
    if user_test_acc > item_test_acc:
      print(f"[User-based] performs better than [Item-based] with (Acc: {user_test_acc:.4f} vs {item_test_acc:.4f})")
    elif user_test_acc < item_test_acc:
      print(f"[Item-based] performs better than [User-based] with (Acc: {item_test_acc:.4f} vs {user_test_acc:.4f})")
    else:
      print(f"[User-based] and [Item-based] perform equally with (Acc: {user_test_acc:.4f})")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
