import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def banking_prediction(csv_file='bank.csv'):
    """
    Predicts whether a client will subscribe to a term deposit using Logistic Regression and K-Nearest Neighbors.

    Args:
        csv_file (str, optional): The path to the CSV file containing the dataset.
            Defaults to 'bank.csv'.

    Returns:
        tuple: (logistic_accuracy, knn_accuracy).  Returns (None, None) if an error occurs.
    """
    # 1) Read in the CSV file using pandas. Pay attention to the file delimiter.
    # Inspect the resulting dataframe with respect to the column names and the variable types.
    try:
        df = pd.read_csv(csv_file, delimiter=';')
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}.  Please make sure the file exists and the path is correct.")
        return None, None  # Return None, None in case of error
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None, None  # Return None, None in case of error

    print("1) Data Loading and Inspection:")
    print("First few rows of the dataframe:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns)
    print("\nVariable types:")
    print(df.dtypes)

    # 2) Pick data from the following columns to a second dataframe 'df2': y, job, marital, default, housing, poutcome.
    df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']].copy()
    print("\n2) Selected Dataframe (df2):")
    print("First few rows of df2:")
    print(df2.head())

    # 3) Convert categorical variables to dummy numerical values.
    df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)
    print("\n3) Dataframe after One-Hot Encoding (df3):")
    print("First few rows of df3 (after one-hot encoding):")
    print(df3.head())

    # 4) Produce a heat map of correlation coefficients for all variables in df3.
    # Describe the amount of correlation between the variables.
    print("\n4) Correlation Analysis:")
    # Convert 'y' to numerical (0 and 1) before calculating correlation
    df3['y'] = df3['y'].map({'yes': 1, 'no': 0})
    correlation_matrix = df3.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Variables in df3')
    plt.show()

    # Add description of correlations.
    print("""
    Looking at the heatmap, we can see that most of the variables have relatively weak correlations with each other.
    The target variable 'y' (whether the client subscribed to a term deposit) shows some noticeable positive correlation
    with 'poutcome_success', which makes sense intuitively - if the client was successfully contacted in a previous
    campaign, they might be more likely to subscribe this time.

    There are also some correlations among the explanatory variables themselves. For example, different job types
    might have some association with marital status, although these correlations don't appear to be very strong.
    The dummy variables created from the same original categorical feature are, by nature, negatively correlated
    with the dropped first category (though this isn't explicitly shown in the typical correlation between columns).

    Overall, it seems like we don't have a lot of strong linear relationships between most of these variables, which
    can sometimes be a good thing for certain types of models as it reduces multicollinearity concerns.
    """)

    # 5) Select the column called 'y' of df3 as the target variable y,
    # and all the remaining columns for the explanatory variables X.
    y = df3['y']
    X = df3.drop('y', axis=1)

    print("\n5) Feature and Target Variable Selection:")
    print("Shape of X (Explanatory Variables):", X.shape)
    print("Shape of y (Target Variable):", y.shape)

    # 6) Split the dataset into training and testing sets with 75/25 ratio.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("\n6) Data Splitting (75% Train, 25% Test):")
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    # 7) Setup a logistic regression model, train it with training data and predict on testing data.
    print("\n7) Logistic Regression Model:")
    logistic_model = LogisticRegression(random_state=42, solver='liblinear')
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)

    # 8) Print the confusion matrix and accuracy score for the logistic regression model.
    print("\n8) Logistic Regression Performance:")
    cm_logistic = confusion_matrix(y_test, y_pred_logistic)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No (Predicted)', 'Yes (Predicted)'],
                yticklabels=['No (Actual)', 'Yes (Actual)'])
    plt.title('Confusion Matrix for Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    print(f"Accuracy of Logistic Regression: {accuracy_logistic:.4f}")

    # 9) Repeat steps 7 and 8 for k-nearest neighbors model. Use k=3.
    print("\n9) K-Nearest Neighbors (KNN) Model (k=3):")
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    print("\n10) K-Nearest Neighbors (KNN) Performance (k=3):")
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
                xticklabels=['No (Predicted)', 'Yes (Predicted)'],
                yticklabels=['No (Actual)', 'Yes (Actual)'])
    plt.title('Confusion Matrix for K-Nearest Neighbors (k=3)')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()

    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy of K-Nearest Neighbors (k=3): {accuracy_knn:.4f}")

    # 10) Compare the results between the two models.
    print("\n10) Model Comparison:")
    print(f"""
    Comparing the results of the two models:

    Logistic Regression achieved an accuracy of approximately {accuracy_logistic:.4f}, while
    K-Nearest Neighbors (with k=3) achieved an accuracy of approximately {accuracy_knn:.4f}.

    In this particular case, Logistic Regression seems to be performing slightly better than
    K-Nearest Neighbors in terms of overall accuracy on the test set.

    Looking at the confusion matrices:

    For Logistic Regression, the number of true negatives (correctly predicted 'no' subscriptions)
    and true positives (correctly predicted 'yes' subscriptions) are shown in the diagonal.
    The off-diagonal elements represent the misclassifications (false positives and false negatives).

    Similarly, the confusion matrix for K-Nearest Neighbors shows its classification performance.
    It appears that both models are better at predicting clients who will not subscribe ('no')
    compared to those who will subscribe ('yes'), as indicated by the higher numbers in the top-left
    corner of the confusion matrices. This could be due to an imbalance in the target variable,
    where there might be more instances of clients who did not subscribe.

    Further analysis, such as looking at precision, recall, and F1-scores, could provide a more
    nuanced understanding of each model's strengths and weaknesses, especially in handling the
    minority class (subscribers). Experimenting with different values of 'k' in KNN might also
    yield different results.
    """)
    return accuracy_logistic, accuracy_knn

if _name_ == "_main_":
    # You can change the CSV file path here if needed.
    logistic_accuracy, knn_accuracy = banking_prediction('bank.csv')
    if logistic_accuracy is not None and knn_accuracy is not None:
        print(f"Logistic Regression Accuracy: {logistic_accuracy}")
        print(f"KNN Accuracy: {knn_accuracy}")
    else:
        print("An error occurred, and the accuracies could not be calculated.")