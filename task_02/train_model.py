"""
Train Logistic Regression model on 70% of the data and save it.
"""
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_loader import load_data


if __name__ == "__main__":
    # 1. Load data
    X, y = load_data('task_02/luka_babunadze_1_83598478.csv')

    # 2. Split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42
    )

    # 3. Initialize and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4. Save model and test split for evaluation
    joblib.dump(model, 'task_02/logistic_model.joblib')
    joblib.dump((X_test, y_test), 'task_02/test_data.joblib')

    # 5. Print coefficients
    print("Trained Logistic Regression model coefficients:")
    for feat, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feat}: {coef:.4f}")