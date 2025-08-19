import os

import boto3
import pandas as pd


def download_from_s3(
    bucket_name, s3_file_key, local_file_path, region_name="us-east-1"
):
    """
    Download a file from S3 bucket to local directory

    Parameters:
    -----------
    bucket_name : str
        Name of the S3 bucket
    s3_file_key : str
        Path to the file in S3 bucket
    local_file_path : str
        Local destination path for the downloaded file
    region_name : str
        AWS region name (default: 'us-east-1')

    Returns:
    --------
    bool
        True if download successful, False otherwise
    """
    try:
        # Create an S3 client
        s3 = boto3.client(
            "s3",
            region_name=region_name,
            # aws_access_key_id='your_access_key',
            # aws_secret_access_key='your_secret_key'
        )

        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(
            f"✅ File downloaded successfully from s3://{bucket_name}/{s3_file_key} to {local_file_path}"
        )
        return True

    except Exception as e:
        print("❌ Error downloading file:", e)
        return False


def load_dataset(file_path):
    """
    Load the parquet dataset and set telephone_number as index

    Parameters:
    -----------
    file_path : str
        Path to the parquet file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with telephone_number as index
    """
    try:
        df = pd.read_parquet(file_path)
        df.set_index("telephone_number", inplace=True)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def get_dataset_info(df):
    """
    Print basic information about the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    """
    print("\n" + "=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes.value_counts())

    print(f"\nObject columns: {list(df.columns[df.dtypes=='object'])}")
    print(f"Float64 columns: {list(df.columns[df.dtypes=='float64'])}")
    print(f"Int64 columns: {list(df.columns[df.dtypes=='int64'])}")

    print("\nMissing values:")
    missing_counts = df.isna().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
    else:
        print("No missing values found")

    print("\nFirst few rows:")
    print(df.head())


def main():
    """
    Main function to download and load the multisim dataset
    """
    print("Making/downloading raw data...")

    # Configuration
    bucket_name = "dataminds-warehouse"
    s3_file_key = "multisim_dataset.parquet"
    local_file_path = "multisim_dataset.parquet"

    # Check if file already exists locally
    if os.path.exists(local_file_path):
        print(f"File {local_file_path} already exists locally. Skipping download.")
    else:
        # Download from S3
        success = download_from_s3(bucket_name, s3_file_key, local_file_path)
        if not success:
            print("Failed to download dataset. Exiting.")
            return None

    # Load and examine the dataset
    df = load_dataset(local_file_path)
    if df is not None:
        get_dataset_info(df)
        return df
    else:
        return None


if __name__ == "__main__":
    dataset = main()
