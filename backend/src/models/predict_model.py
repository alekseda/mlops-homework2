import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


def load_trained_model(model_path="best_model.pkl"):
    """
    Load the trained model from file

    Parameters:
    -----------
    model_path : str
        Path to the saved model

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Loaded trained model
    """
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def load_test_data(file_path="processed_multisim_dataset.parquet"):
    """
    Load and split data for prediction evaluation

    Parameters:
    -----------
    file_path : str
        Path to the processed dataset

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    try:
        df = pd.read_parquet(file_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        # Use same split as training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        print(f"✅ Test data loaded: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None, None, None


def make_predictions(model, X_test):
    """
    Make predictions using the trained model

    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model
    X_test : pd.DataFrame
        Test features

    Returns:
    --------
    tuple
        (predictions, prediction_probabilities)
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

        print(f"✅ Predictions generated for {len(y_pred)} samples")
        print(f"Positive predictions: {sum(y_pred)} ({sum(y_pred)/len(y_pred)*100:.1f}%)")

        return y_pred, y_pred_proba

    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return None, None


def plot_roc_curve(y_test, y_pred_proba, title="ROC Curve"):
    """
    Plot ROC curve and calculate AUC

    Parameters:
    -----------
    y_test : pd.Series
        True labels
    y_pred_proba : np.array
        Prediction probabilities
    title : str
        Plot title
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"AUC Score: {auc_score:.3f}")
    return auc_score


def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix

    Parameters:
    -----------
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predicted labels
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return cm


def predict_new_data(model, new_data):
    """
    Make predictions on new data

    """
    try:
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)[:, 1]

        print(f"✅ Predictions made for {len(predictions)} new samples")
        return predictions, probabilities

    except Exception as e:
        print(f"❌ Error predicting new data: {e}")
        return None, None


def create_prediction_summary(predictions, probabilities, telephone_numbers=None):
    """
    Create a summary dataframe of predictions

    """
    summary_df = pd.DataFrame(
        {
            "prediction": predictions,
            "probability": probabilities,
            "confidence": np.where(predictions == 1, probabilities, 1 - probabilities),
        }
    )

    if telephone_numbers is not None:
        summary_df.index = telephone_numbers
        summary_df.index.name = "telephone_number"

    # Sort by probability (highest risk first)
    summary_df = summary_df.sort_values("probability", ascending=False)

    print("Prediction Summary:")
    print(
        f"Positive predictions: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)"
    )
    print(f"Average confidence: {summary_df['confidence'].mean():.3f}")

    return summary_df


def prediction_analysis(y_test, y_pred, y_pred_proba):
    """
    Comprehensive analysis of predictions

    """
    print("\n" + "=" * 50)
    print("PREDICTION ANALYSIS")
    print("=" * 50)

    # Basic metrics
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.3f}")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred)

    # ROC curve
    auc_score = plot_roc_curve(y_test, y_pred_proba, "ROC Curve - Model Predictions")

    # Prediction probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label="Negative Class", color="red")
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label="Positive Class", color="blue")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Probabilities by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {"accuracy": accuracy, "auc_score": auc_score, "confusion_matrix": cm}
