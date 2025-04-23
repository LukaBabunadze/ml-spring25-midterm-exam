"""
Evaluate saved model on hold-out test set and plot confusion matrix.
"""
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


if __name__ == "__main__":
    # 1. Load model and test data
    model = joblib.load('task_02/logistic_model.joblib')
    X_test, y_test = joblib.load('task_02/test_data.joblib')

    # 2. Predict
    y_pred = model.predict(X_test)

    # 3. Metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('task_02/confusion_matrix.png')
    plt.show()