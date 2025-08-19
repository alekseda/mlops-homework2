import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from make_dataset import load_dataset

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


def analyze_categorical_features(df):

    # Get categorical columns
    categorical_cols = df.columns[df.dtypes == "object"].tolist()
    if "target" in categorical_cols:
        categorical_cols.remove("target")

    print(f"Categorical features: {categorical_cols}")

    # Create visualizations for key categorical features
    fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 5))
    if len(categorical_cols) == 1:
        axes = [axes]

    for i, col in enumerate(categorical_cols):
        if col in df.columns:
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Cross-tabulation analysis with target
    print("\nCategorical vs Target Analysis:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col} vs target (normalized by row):")
            crosstab = pd.crosstab(df[col], df["target"], normalize="index")
            print(crosstab)


def detect_outliers(df, column):
    """
    Detect outliers using IQR method
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)]
    return len(outliers)


def analyze_numerical_features(df):
    """
    Analyze numerical features including outlier detection
    """
    print("\n" + "=" * 50)
    print("NUMERICAL FEATURE ANALYSIS")
    print("=" * 50)

    # Get numerical columns
    numeric_cols = df.columns[df.dtypes.isin(["float64", "int64"])].tolist()
    if "target" in numeric_cols:
        numeric_cols.remove("target")

    print(f"Numerical features: {numeric_cols}")

    # Outlier analysis
    print("\nOutlier Analysis:")
    for col in numeric_cols:
        if col in df.columns:
            outlier_count = detect_outliers(df, col)
            outlier_percentage = (outlier_count / len(df)) * 100
            print(f"{col}: {outlier_count} outliers ({outlier_percentage:.2f}%)")

    # Basic statistics
    print("\nNumerical Features Summary:")
    print(df[numeric_cols].describe())


def analyze_correlations(df):
    """
    Analyze correlations between features and target

    """

    # Select key numerical features for correlation analysis
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

    # Filter columns that actually exist in the dataset
    available_cols = [col for col in correlation_cols if col in df.columns]

    if len(available_cols) > 1:
        correlation_matrix = df[available_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
        )
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.show()

        # Print correlations with target
        if "target" in available_cols:
            target_corr = (
                correlation_matrix["target"]
                .drop("target")
                .sort_values(key=abs, ascending=False)
            )
            print("Features most correlated with target:")
            print(target_corr)
    else:
        print("Not enough numerical features for correlation analysis")


def feature_engineering(df):
    """
    Create new features based on existing ones

    """

    df_engineered = df.copy()

    # Example feature engineering based on common patterns
    # You can customize these based on your domain knowledge

    # Tenure-based features
    if "tenure" in df.columns:
        df_engineered["tenure_years"] = df_engineered["tenure"] / 12
        df_engineered["is_long_tenure"] = (
            df_engineered["tenure"] > df_engineered["tenure"].median()
        ).astype(int)
        print("✅ Created tenure-based features")

    # Value-based feature combinations
    val_cols = [
        col for col in df.columns if col.startswith("val") and col.endswith("_1")
    ]
    if len(val_cols) >= 2:
        # Create sum and mean of value features
        df_engineered["val_sum"] = df_engineered[val_cols].sum(axis=1)
        df_engineered["val_mean"] = df_engineered[val_cols].mean(axis=1)
        df_engineered["val_std"] = df_engineered[val_cols].std(axis=1)
        print(f"✅ Created aggregated features from {len(val_cols)} value columns")

    # Binary feature interactions
    if "is_dualsim" in df.columns and "gndr" in df.columns:
        df_engineered["dualsim_gender_interaction"] = (
            df_engineered["is_dualsim"].astype(str)
            + "_"
            + df_engineered["gndr"].astype(str)
        )
        print("✅ Created interaction features")

    print(f"Original features: {df.shape[1]}")
    print(f"Features after engineering: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")

    return df_engineered


def save_processed_data(df, output_path="processed_multisim_dataset.parquet"):
    """
    Save the processed dataset

    """
    try:
        df.to_parquet(output_path)
        print(f"✅ Processed dataset saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving processed dataset: {e}")


def main():
    """
    Main function to transform raw data into features
    """
    print("Transforming raw data into features...")

    # Load the raw dataset
    raw_file_path = "multisim_dataset.parquet"

    if not pd.io.common.file_exists(raw_file_path):
        print(f"❌ Raw dataset not found at {raw_file_path}")
        print("Please run make_dataset.py first to download the data")
        return None

    # Load dataset
    df = load_dataset(raw_file_path)
    if df is None:
        return None

    # Analyze categorical features
    analyze_categorical_features(df)

    # Analyze numerical features
    analyze_numerical_features(df)

    # Analyze correlations
    analyze_correlations(df)

    # Engineer new features
    df_processed = feature_engineering(df)

    # Save processed dataset
    save_processed_data(df_processed)

    return df_processed


if __name__ == "__main__":
    processed_dataset = main()
