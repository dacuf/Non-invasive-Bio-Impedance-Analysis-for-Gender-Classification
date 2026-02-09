import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Create directories if they don't exist
result_dir = "training_result"
cv_dir = os.path.join(result_dir, "cross_validation")
cm_dir = os.path.join(result_dir, "confusion_matrix")

os.makedirs(cv_dir, exist_ok=True)
os.makedirs(cm_dir, exist_ok=True)

# Read input file and sheet name
input_filename = input("Enter the name of the data file (xlsx): ")
sheet_name = input("Enter the sheet name: ")
input_data = pd.read_excel(input_filename, sheet_name=sheet_name)

y = input_data.Gender  # Target variable

# Remove Gender and Number column
input_data.drop(['Gender', 'Number'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in input_data.columns if input_data[cname].dtype in ['int64', 'float64']]
X = input_data[numeric_cols].copy()

print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))
pd.concat([X, y], axis=1).head()  # Show first 5 training examples

# Function to perform cross-validation and plot confusion matrix
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, cv_dir, cm_dir, test_size, n_splits):
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)

    # Plot CV scores
    plt.figure()
    plt.plot(cv_scores)
    plt.title(f'Cross-Validation Scores for {model_name}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(cv_dir, f'{model_name}_cv_test_size_{test_size}_n_splits_{n_splits}.png'))
    plt.close()

    # Fit the model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(os.path.join(cm_dir, f'{model_name}_cm_test_size_{test_size}_n_splits_{n_splits}.png'))
    plt.close()

    return mean_cv_score, model

# List of test sizes and number of splits to try
test_sizes = [0.2, 0.3, 0.4]
n_splits_list = [4, 5, 6, 7, 8, 9, 10]

# Loop through test sizes and number of splits
for test_size in test_sizes:
    for n_splits in n_splits_list:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # Stratified K-Fold cross-validation
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

        # Decision Tree Classifier
        param_grid_tree = {
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'ccp_alpha': np.arange(0.001, 0.1, 0.001)
        }
        tree_clas = DecisionTreeClassifier(random_state=0)
        grid_search_tree = GridSearchCV(estimator=tree_clas, param_grid=param_grid_tree, cv=kf, verbose=True, scoring='accuracy')
        best_tree_score, best_tree_model = evaluate_model(grid_search_tree, X_train, X_test, y_train, y_test, f"Decision Tree (test_size={test_size}, n_splits={n_splits})", cv_dir, cm_dir, test_size, n_splits)

        # Support Vector Machine (SVM) Classifier
        param_grid_svm = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        svm_clas = SVC(random_state=0)
        grid_search_svm = GridSearchCV(estimator=svm_clas, param_grid=param_grid_svm, cv=kf, verbose=True, scoring='accuracy')
        best_svm_score, best_svm_model = evaluate_model(grid_search_svm, X_train, X_test, y_train, y_test, f"SVM (test_size={test_size}, n_splits={n_splits})", cv_dir, cm_dir, test_size, n_splits)

        # Neural Network Classifier
        param_grid_nn = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive']
        }
        nn_clas = MLPClassifier(max_iter=100, random_state=0)
        grid_search_nn = GridSearchCV(estimator=nn_clas, param_grid=param_grid_nn, cv=kf, verbose=True, scoring='accuracy')
        best_nn_score, best_nn_model = evaluate_model(grid_search_nn, X_train, X_test, y_train, y_test, f"Neural Network (test_size={test_size}, n_splits={n_splits})", cv_dir, cm_dir, test_size, n_splits)

        # Ensemble Model - Voting Classifier
        voting_clas = VotingClassifier(estimators=[
            ('svm', grid_search_svm.best_estimator_),
            ('tree', grid_search_tree.best_estimator_),
            ('nn', grid_search_nn.best_estimator_)
        ], voting='hard')
        best_voting_score, best_voting_model = evaluate_model(voting_clas, X_train, X_test, y_train, y_test, f"Voting Classifier (test_size={test_size}, n_splits={n_splits})", cv_dir, cm_dir, test_size, n_splits)

        # Ensemble Model - Stacking Classifier
        stacking_clas = StackingClassifier(estimators=[
            ('svm', grid_search_svm.best_estimator_),
            ('tree', grid_search_tree.best_estimator_),
            ('nn', grid_search_nn.best_estimator_)
        ], final_estimator=DecisionTreeClassifier())
        best_stacking_score, best_stacking_model = evaluate_model(stacking_clas, X_train, X_test, y_train, y_test, f"Stacking Classifier (test_size={test_size}, n_splits={n_splits})", cv_dir, cm_dir, test_size, n_splits)

        # Save results to a text file with best model and score for each classifier
        output_filename = os.path.join(result_dir, os.path.splitext(os.path.basename(input_filename))[0] + f"_training_test_size_{test_size}_n_splits_{n_splits}.txt")
        with open(output_filename, 'w') as f:
            f.write(f"Best Decision Tree Model:\n{best_tree_model}\n")
            f.write(f"Mean CV Score: {best_tree_score}\n")
            f.write(f"Best Model (best score): {best_tree_model} with score {best_tree_score}\n\n")
            
            f.write(f"Best SVM Model:\n{best_svm_model}\n")
            f.write(f"Mean CV Score: {best_svm_score}\n")
            f.write(f"Best Model (best score): {best_svm_model} with score {best_svm_score}\n\n")
            
            f.write(f"Best Neural Network Model:\n{best_nn_model}\n")
            f.write(f"Mean CV Score: {best_nn_score}\n")
            f.write(f"Best Model (best score): {best_nn_model} with score {best_nn_score}\n\n")
            
            f.write(f"Best Voting Classifier Model:\n{best_voting_model}\n")
            f.write(f"Mean CV Score: {best_voting_score}\n")
            f.write(f"Best Model (best score): {best_voting_model} with score {best_voting_score}\n\n")
