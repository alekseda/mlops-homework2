import pandas as pd

import joblib

from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from scipy.stats import ttest_rel
import optuna
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


def load_processed_data(file_path="processed_multisim_dataset.parquet"):
    """
    Load processed dataset

    """
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Processed dataset loaded: {df.shape}")

        X = df.drop("target", axis=1)
        y = df["target"]

        # Shuffle the data
        X, y = shuffle(X, y, random_state=42)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        return X, y

    except Exception as e:
        print(f"❌ Error loading processed dataset: {e}")
        return None, None


def create_preprocessing_pipeline(X):
    """
    Create preprocessing pipeline for features

    """
    # Identify feature types
    numeric_features = X.columns[X.dtypes.isin(["float64", "int64"])]
    categorical_features = X.columns[X.dtypes == "object"]

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Numeric preprocessing pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("yeo-johnson_transformer", PowerTransformer(method="yeo-johnson")),
            ("pca", PCA(n_components=2, random_state=42)),
        ]
    )

    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", CatBoostEncoder()),
            ("selector", SelectPercentile(chi2, percentile=50)),
        ]
    )

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def create_model_pipelines(preprocessor):
    """
    Create multiple model pipelines for comparison

    """
    models = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        penalty="l1", C=0.5, solver="liblinear", max_iter=1000, random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            [("preprocessor", preprocessor), ("classifier", KNeighborsClassifier(n_neighbors=5))]
        ),
        "SVM": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", DecisionTreeClassifier(random_state=42)),
            ]
        ),
    }

    return models


def optimize_xgboost(X, y, preprocessor, n_trials=100):
    """
    Optimize XGBoost hyperparameters using Optuna

    """
    print("\n" + "=" * 50)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 50)

    def objective(trial):
        param = {
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1e-2),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1e-2),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        clf = Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier(**param))])

        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best Parameters:", study.best_params)
    return study.best_params


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models


    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name} Test Accuracy: {accuracy:.3f}")
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))

        results[name] = {"model": model, "accuracy": accuracy, "predictions": y_pred}

    return results


def cross_validation_comparison(models, X, y, cv_folds=10):
    """
    Perform cross-validation comparison of models

    """
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=1)
    cv_results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = scores
        print(f"{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return cv_results


def statistical_comparison(cv_results):
    """
    Perform statistical comparison between models using paired t-tests

    """
    print("\n" + "=" * 50)
    print("STATISTICAL MODEL COMPARISON")
    print("=" * 50)

    baseline_model = "Logistic Regression"

    if baseline_model in cv_results:
        baseline_scores = cv_results[baseline_model]

        for name, scores in cv_results.items():
            if name != baseline_model:
                t_stat, p_value = ttest_rel(scores, baseline_scores)

                if t_stat > 0:
                    comparison = f"{name} is better than {baseline_model}"
                else:
                    comparison = f"{baseline_model} is better than {name}"

                print(f"{name} vs {baseline_model}:")
                print(f"  t-statistic: {t_stat:.4f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Result: {comparison}")
                print()


def save_best_model(models, cv_results, X_train, y_train, model_path="best_model.pkl"):
    """
    Save the best performing model


    """
    # Find best model based on CV scores
    best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k].mean())
    best_model = models[best_model_name]

    # Retrain on full training data
    best_model.fit(X_train, y_train)

    # Save model
    try:
        joblib.dump(best_model, model_path)
        print(f"✅ Best model ({best_model_name}) saved to {model_path}")
        print(f"   CV Accuracy: {cv_results[best_model_name].mean():.4f}")
        return best_model, best_model_name
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None, None


def main():
    """
    Main function to train models
    """
    print("Training models...")

    # Load processed data
    X, y = load_processed_data()
    if X is None or y is None:
        print("Please run build_features.py first to create processed dataset")
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)

    # Create model pipelines
    models = create_model_pipelines(preprocessor)

    # Optimize XGBoost
    best_xgb_params = optimize_xgboost(X, y, preprocessor, n_trials=100)

    # Add optimized XGBoost to models
    best_xgb_params.update(
        {"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42}
    )
    models["XGBoost (Optimized)"] = Pipeline(
        [("preprocessor", preprocessor), ("classifier", XGBClassifier(**best_xgb_params))]
    )

    # Evaluate all models
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Cross-validation comparison
    cv_results = cross_validation_comparison(models, X, y)

    # Statistical comparison
    statistical_comparison(cv_results)

    # Save best model
    best_model, best_model_name = save_best_model(models, cv_results, X_train, y_train)

    # Summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Best model: {best_model_name}")
    print(f"Cross-validation accuracy: {cv_results[best_model_name].mean():.4f}")
    print(f"Test accuracy: {results[best_model_name]['accuracy']:.4f}")

    return best_model, results, cv_results


if __name__ == "__main__":
    best_model, model_results, cv_scores = main()
