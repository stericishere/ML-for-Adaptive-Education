import numpy as np
import torch
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
from knn import knn_impute_by_user
from item_response import irt as irt
from neural_network import AutoEncoder, load_data as load_nn_data

# === Bootstrap data ===
def bootstrap_data(data):
    # sample data with replacement to create bootstrap sample
    n = len(data["user_id"])
    indices = np.random.choice(n, size=n, replace=True)
    return {k: [v[i] for i in indices] for k, v in data.items()}

# === IRT Helpers ===
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def train_irt(train_data):
    # optimal learning rate = 0.001 and iteration = 500 found in item_response
    # Note: number of iterations is reduced to avoid long computation time
    theta, beta, _, _, _ = irt(train_data, train_data, lr=0.001, iterations=250)
    return theta, beta

def get_irt_preds(theta, beta, data):
    # predict correctness using sigmoid (theta - beta)
    preds = []
    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        x = theta[u] - beta[q]
        p = sigmoid(x)
        preds.append(p >= 0.5)
    return np.array(preds, dtype=int)

# === Ensemble Helpers ===
def majority_vote(predictions_list):
    # perform majority vote across three base models
    return np.round(np.mean(predictions_list, axis=0)).astype(int)

def ensemble_accuracy(true_labels, pred_labels):
    return np.mean(np.array(true_labels) == np.array(pred_labels))

def run_ensemble():
    # === Load data ===
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    sparse_matrix = load_train_sparse("./data").toarray()

    # === KNN ===
    print("Running model KNN...")
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=21)  # k*=21, optimal k found in knn
    # --- Apply bootstrap to training data ---
    knn_data = bootstrap_data(train_data)
    # create matrix filled with NaN values
    knn_matrix = np.full_like(sparse_matrix, fill_value=np.nan, dtype=np.float64)
    # fill in known user-question correctness values from the bootstrapped training data
    for user_id, question_id, is_correct in zip(knn_data["user_id"], knn_data["question_id"], knn_data["is_correct"]):
        knn_matrix[user_id, question_id] = is_correct
    
    # --- Fit and transform ---
    imputed_matrix = imputer.fit_transform(knn_matrix)
    def get_knn_preds(data, matrix):
        # threshold = 0.5
        preds = []
        for i in range(len(data["user_id"])):
            u = data["user_id"][i]
            q = data["question_id"][i]
            p = matrix[u, q]
            preds.append(p >= 0.5)
        return np.array(preds, dtype=int)
    knn_val_preds = get_knn_preds(val_data, imputed_matrix)     # validation set predictions
    knn_test_preds = get_knn_preds(test_data, imputed_matrix)   # test set predictions

    # === IRT ===
    print("Running model IRT...")
    # --- Apply bootstrap to training data ---
    irt_data = bootstrap_data(train_data)
    # --- Fit and transform ---
    theta, beta = train_irt(irt_data)
    irt_val_preds = get_irt_preds(theta, beta, val_data)     # validation set predictions
    irt_test_preds = get_irt_preds(theta, beta, test_data)   # test set predictions

    # === Neural Network ===
    print("Running model Neural Network...")
    zero_train_matrix, train_matrix, _, _ = load_nn_data("./data")
    # --- Apply bootstrap to training data ---
    nn_data = bootstrap_data(train_data)
    # --- Clean data ---
    # create mask matrix and initialized with zeros
    mask = np.zeros_like(train_matrix)  # track which (user, question) pairs have observed responses
    # fill known responses from bootstrapped training data and mark their positions in the mask
    for user_id, question_id, is_correct in zip(nn_data["user_id"], nn_data["question_id"], nn_data["is_correct"]):
        train_matrix[user_id, question_id] = is_correct  # store the observed value
        mask[user_id, question_id] = 1                   # mark this entry as observed
    # combine observed data with zero_train_matrix to avoid using unobserved entries
    train_matrix = train_matrix * mask + zero_train_matrix * (1 - mask)
    # --- Fit and transform ---
    model = AutoEncoder(num_question=train_matrix.shape[1], k=50)  # k*=50, optimal k found in neural_network
    from neural_network import train, evaluate
    train(model, lr=0.01, lamb=0.0, train_data=train_matrix, 
          zero_train_data=zero_train_matrix, valid_data=val_data, num_epoch=45)
    
    def get_nn_preds(data, model, zero_train_matrix): 
      # generate predictions
      model.eval()
      preds = []
      for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        inputs = zero_train_matrix[u].unsqueeze(0)
        with torch.no_grad():
            output = model(inputs)
            p = output[0][q].item()
            preds.append(p >= 0.5)
      return np.array(preds, dtype=int)
    nn_val_preds = get_nn_preds(val_data, model, zero_train_matrix)     # validation set predictions
    nn_test_preds = get_nn_preds(test_data, model, zero_train_matrix)   # test set predictions

    # === DEBUG PRINT ===
    print("Debug print, please ignore:")
    print("KNN:", len(knn_val_preds))
    print("IRT:", len(irt_val_preds))
    print("NN :", len(nn_val_preds))
    print("VAL:", len(val_data["user_id"]))

    # Sanity check to ensure prediction lengths match
    if not (len(knn_val_preds) == len(irt_val_preds) == len(nn_val_preds)):
        raise ValueError("Mismatch in prediction lengths. Cannot ensemble.")

    # === Ensemble ===
    print("Running ensemble...")
    ensemble_val_preds = majority_vote([knn_val_preds, irt_val_preds, nn_val_preds])
    ensemble_test_preds = majority_vote([knn_test_preds, irt_test_preds, nn_test_preds])
    ensemble_val_true = np.array(val_data["is_correct"], dtype=int)
    ensemble_test_true = np.array(test_data["is_correct"], dtype=int)
    ensemble_val_acc = ensemble_accuracy(ensemble_val_true, ensemble_val_preds)
    ensemble_test_acc = ensemble_accuracy(ensemble_test_true, ensemble_test_preds)
    print(f"Ensemble validation accuracy: {ensemble_val_acc:.4f}")
    print(f"Ensemble test accuracy: {ensemble_test_acc:.4f}")

if __name__ == "__main__":
    run_ensemble()