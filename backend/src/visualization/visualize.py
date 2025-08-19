import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import learning_curve

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")


def load_data_and_results():
    """
    Load dataset and any saved results for visualization

    Returns:
    --------
    tuple
        (df, predictions_df) - dataset and predictions if available
    """
    try:
        # Load processed dataset
        df = pd.read_parquet("processed_multisim_dataset.parquet")
        print(f"✅ Dataset loaded: {df.shape}")

        # Try to load predictions
        predictions_df = None
        try:
            predictions_df = pd.read_csv("predictions_summary.csv", index_col=0)
            print(f"✅ Predictions loaded: {predictions_df.shape}")
        except FileNotFoundError:
            print(
                "No predictions file found - run predict_model.py first for prediction visualizations"
            )

        return df, predictions_df

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def plot_target_distribution(df):
    """
    Visualize target variable distribution

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with target variable
    """
    plt.figure(figsize=(12, 5))

    # Target distribution
    plt.subplot(1, 2, 1)
    target_counts = df["target"].value_counts()
    plt.pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Target Distribution")

    # Target count bar plot
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x="target")
    plt.title("Target Count Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def plot_categorical_analysis(df):
    """
    Create comprehensive categorical feature visualizations

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to visualize
    """
    categorical_cols = df.columns[df.dtypes == "object"].tolist()
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    if not categorical_cols:
        print("No categorical features found for visualization")
        return

    # Distribution plots
    n_cols = len(categorical_cols)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(categorical_cols):
        # Distribution plot
        sns.countplot(data=df, x=col, ax=axes[0, i])
        axes[0, i].set_title(f"Distribution of {col}")
        axes[0, i].tick_params(axis="x", rotation=45)

        # Target relationship plot
        if "target" in df.columns:
            crosstab = pd.crosstab(df[col], df["target"], normalize="index")
            crosstab.plot(kind="bar", ax=axes[1, i], stacked=True)
            axes[1, i].set_title(f"{col} vs Target")
            axes[1, i].tick_params(axis="x", rotation=45)
            axes[1, i].legend(title="Target")

    plt.tight_layout()
    plt.show()


def plot_numerical_analysis(df):
    """
    Create numerical feature visualizations

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to visualize
    """
    numeric_cols = df.columns[df.dtypes.isin(["float64", "int64"])].tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    if not numeric_cols:
        print("No numerical features found for visualization")
        return

    # Distribution plots
    n_cols = min(4, len(numeric_cols))  # Max 4 columns per row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols

        if n_rows == 1:
            ax = axes[col_idx] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col_idx]

        # Distribution plot
        df[col].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        if n_rows == 1:
            ax = axes[col_idx] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col_idx]
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

    # Box plots for outlier visualization
    if len(numeric_cols) > 0:
        plt.figure(figsize=(15, 8))
        df[numeric_cols].boxplot(figsize=(15, 8))
        plt.title("Box Plots - Outlier Detection")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df):
    """
    Create correlation heatmap focusing on key features

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    """
    # Select key features for correlation (same as your original code)
    correlation_cols = [
        "target",
        "tenure",
        "val4_1",
        "val6_1",
        "val8_1",
        "val10_1",
        "val19_1",
        "val20_1",
        "val21_1",
    ]

    # Filter columns that actually exist
    available_cols = [col for col in correlation_cols if col in df.columns]

    if len(available_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[available_cols].corr()

        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            mask=mask,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

        # Feature importance based on correlation with target
        if "target" in available_cols:
            target_corr = (
                correlation_matrix["target"]
                .drop("target")
                .abs()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(10, 6))
            target_corr.plot(kind="bar")
            plt.title("Feature Correlation with Target (Absolute Values)")
            plt.xlabel("Features")
            plt.ylabel("Absolute Correlation")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        print("Not enough features for correlation analysis")


def plot_model_performance_comparison(cv_results=None):
    """
    Visualize model performance comparison

    Parameters:
    -----------
    cv_results : dict, optional
        Cross-validation results from train_model.py
    """
    # If no cv_results provided, try to load model and generate simple comparison
    if cv_results is None:
        print("No CV results provided - loading model for basic evaluation")
        return

    # Model performance comparison
    model_names = list(cv_results.keys())
    cv_means = [scores.mean() for scores in cv_results.values()]
    cv_stds = [scores.std() for scores in cv_results.values()]

    plt.figure(figsize=(12, 6))

    # Bar plot with error bars
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    plt.title("Model Performance Comparison (Cross-Validation)")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, mean_val in zip(bars, cv_means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()

    # Box plot of CV scores
    plt.figure(figsize=(12, 6))
    cv_data = [scores for scores in cv_results.values()]
    plt.boxplot(cv_data, labels=model_names)
    plt.title("Cross-Validation Score Distribution by Model")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prediction_analysis(predictions_df):
    """
    Visualize prediction results and patterns

    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Dataframe with predictions and probabilities
    """
    if predictions_df is None:
        print("No predictions available for visualization")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Prediction distribution
    axes[0, 0].pie(
        predictions_df["prediction"].value_counts().values,
        labels=["Negative", "Positive"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0, 0].set_title("Prediction Distribution")

    # Probability distribution
    axes[0, 1].hist(
        predictions_df["probability"], bins=30, alpha=0.7, edgecolor="black"
    )
    axes[0, 1].set_title("Prediction Probability Distribution")
    axes[0, 1].set_xlabel("Probability")
    axes[0, 1].set_ylabel("Frequency")

    # Confidence distribution
    axes[1, 0].hist(
        predictions_df["confidence"],
        bins=30,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    axes[1, 0].set_title("Prediction Confidence Distribution")
    axes[1, 0].set_xlabel("Confidence")
    axes[1, 0].set_ylabel("Frequency")

    # Probability by prediction
    axes[1, 1].boxplot(
        [
            predictions_df[predictions_df["prediction"] == 0]["probability"],
            predictions_df[predictions_df["prediction"] == 1]["probability"],
        ],
        labels=["Negative", "Positive"],
    )
    axes[1, 1].set_title("Probability Distribution by Prediction")
    axes[1, 1].set_ylabel("Probability")

    plt.tight_layout()
    plt.show()

    # High-risk customers
    high_risk = predictions_df[predictions_df["probability"] > 0.7]
    print(
        f"High-risk customers (>70% probability): {len(high_risk)} ({len(high_risk)/len(predictions_df)*100:.1f}%)"
    )


def create_comprehensive_dashboard(df, predictions_df=None):
    """
    Create a comprehensive visualization dashboard

    Parameters:
    -----------
    df : pd.DataFrame
        Main dataset
    predictions_df : pd.DataFrame, optional
        Predictions dataframe
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VISUALIZATION DASHBOARD")
    print("=" * 60)

    # 1. Target Distribution
    plot_target_distribution(df)

    # 2. Feature Distributions
    plot_numerical_analysis(df)

    # 3. Categorical Analysis
    plot_categorical_analysis(df)

    # 4. Correlation Analysis
    plot_correlation_heatmap(df)

    # 5. Prediction Analysis (if available)
    if predictions_df is not None:
        plot_prediction_analysis(predictions_df)

    print("✅ Comprehensive dashboard created!")


def plot_feature_importance(model_path="best_model.pkl", X_test=None):
    """
    Plot feature importance if model supports it

    Parameters:
    -----------
    model_path : str
        Path to saved model
    X_test : pd.DataFrame, optional
        Test features for importance calculation
    """
    try:
        model = joblib.load(model_path)

        # Try to get feature importance
        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            # For tree-based models
            importances = model.named_steps["classifier"].feature_importances_

            # Get feature names after preprocessing
            if X_test is not None:
                # Transform features to get processed feature names
                X_processed = model.named_steps["preprocessor"].transform(X_test)
                n_features = X_processed.shape[1]
                feature_names = [f"Feature_{i}" for i in range(n_features)]
            else:
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            # Plot top 20 features
            indices = np.argsort(importances)[::-1][:20]

            plt.figure(figsize=(12, 8))
            plt.bar(range(len(indices)), importances[indices])
            plt.title("Top 20 Feature Importances")
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.xticks(
                range(len(indices)), [feature_names[i] for i in indices], rotation=45
            )
            plt.tight_layout()
            plt.show()

        elif hasattr(model.named_steps["classifier"], "coef_"):
            # For linear models
            coefficients = model.named_steps["classifier"].coef_[0]

            # Plot top coefficients
            indices = np.argsort(np.abs(coefficients))[::-1][:20]

            plt.figure(figsize=(12, 8))
            colors = ["red" if coef < 0 else "blue" for coef in coefficients[indices]]
            plt.bar(range(len(indices)), coefficients[indices], color=colors, alpha=0.7)
            plt.title("Top 20 Model Coefficients")
            plt.xlabel("Features")
            plt.ylabel("Coefficient Value")
            plt.xticks(
                range(len(indices)), [f"Feature_{i}" for i in indices], rotation=45
            )
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.tight_layout()
            plt.show()

        else:
            print("Model does not support feature importance visualization")

    except Exception as e:
        print(f"❌ Error plotting feature importance: {e}")


def plot_learning_curves(model_path="best_model.pkl", X=None, y=None):
    """
    Plot learning curves to analyze model performance vs training size

    Parameters:
    -----------
    model_path : str
        Path to saved model
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    """
    if X is None or y is None:
        print("No data provided for learning curves")
        return

    try:
        model = joblib.load(model_path)

        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=5,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring="accuracy",
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
        plt.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.1,
            color="blue",
        )

        plt.plot(train_sizes, val_mean, "o-", color="red", label="Validation Score")
        plt.fill_between(
            train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red"
        )

        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy")
        plt.title("Learning Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error plotting learning curves: {e}")


def plot_roc_comparison(df, model_path="best_model.pkl"):
    """
    Create ROC curve visualization

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    model_path : str
        Path to saved model
    """
    try:
        from sklearn.model_selection import train_test_split

        # Load model and data
        model = joblib.load(model_path)
        X = df.drop("target", axis=1)
        y = df["target"]

        # Split data (same as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # Get predictions
        y_scores = model.predict_proba(X_test)[:, 1]

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_score = roc_auc_score(y_test, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, label=f"Model AUC = {auc_score:.3f}", linewidth=2, color="blue"
        )
        plt.plot(
            [0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Model Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return auc_score

    except Exception as e:
        print(f"❌ Error creating ROC curve: {e}")
        return None


def save_all_visualizations():
    """
    Save all generated plots to files
    """
    # Note: This is a placeholder - matplotlib plots would need to be
    # explicitly saved using plt.savefig() in each plotting function
    print(
        "💡 To save plots, add plt.savefig('filename.png') before plt.show() in each plot function"
    )


def main():
    """
    Main function to create figures and charts
    """
    print("Creating figures and charts...")

    # Load data
    df, predictions_df = load_data_and_results()
    if df is None:
        print("Please run build_features.py first to create processed dataset")
        return

    # Create comprehensive dashboard
    create_comprehensive_dashboard(df, predictions_df)

    # Additional advanced visualizations if model exists
    if pd.io.common.file_exists("best_model.pkl"):
        print("\nCreating advanced model visualizations...")

        # ROC curve

        # Feature importance
        X = df.drop("target", axis=1)
        plot_feature_importance(X_test=X.head(100))  # Use sample for processing

        # Learning curves
        X = df.drop("target", axis=1)
        y = df["target"]
        plot_learning_curves(
            X=X.sample(1000), y=y.loc[X.sample(1000).index]
        )  # Use sample for speed

    else:
        print(
            "No trained model found - run train_model.py first for model-specific visualizations"
        )

    print("\n✅ All visualizations completed!")
    print(
        "💡 Tip: Run this script after train_model.py and predict_model.py for complete visualizations"
    )


if __name__ == "__main__":
    main()
