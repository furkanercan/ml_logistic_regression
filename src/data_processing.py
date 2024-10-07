import numpy as np
import pandas as pd
from scipy import stats


def clear_nulls(df,columns):
    for col in columns:
        null_counter=df[col].isnull().sum()
        if null_counter>0:
             df[col]=df[col].fillna(df[col].mean())

def infer_feature_type(df, threshold=10):
    """
    Infers whether each feature in a DataFrame is categorical or continuous.
    
    Args:
        df (DataFrame): Input DataFrame.
        threshold (int): Maximum number of unique values for a feature to be considered categorical.
        
    Returns:
        dict: Dictionary with feature names as keys and 'categorical' or 'continuous' as values.
    """
    feature_types = {}
    
    for col in df.columns:
        # Get the number of unique values in the column
        num_unique = df[col].nunique()
        dtype = df[col].dtype
        
        # If it's an object or category type, assume categorical
        if dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            feature_types[col] = 'categorical'
        
        # If it's numeric but has a small number of unique values, treat it as categorical (e.g., binary features)
        elif pd.api.types.is_numeric_dtype(df[col]) and num_unique <= threshold:
            feature_types[col] = 'categorical'
        
        # Otherwise, it's continuous
        else:
            feature_types[col] = 'continuous'
    
    return feature_types

def outlier_handler_zscore(df, features, threshold=3):
    """
    Detects and optionally handles outliers using the Z-score method.

    Args:
        df (DataFrame): The input DataFrame.
        features (list): List of column names to check for outliers.
        threshold (float): Z-score threshold to define an outlier. Default is 3.

    Returns:
        DataFrame: A DataFrame with outliers removed or handled.
    """
    total_outliers_detected = 0
    total_outliers_handled = 0

    for col in features:
        
        df[col] = df[col].astype(float)  # Cast the column to float

        # Calculate Z-scores for each value in the column
        z_scores = np.abs(stats.zscore(df[col]))

        # Identify indices of outliers based on the Z-score threshold
        outliers = np.where(z_scores > threshold)
        num_outliers = len(outliers[0])
        total_outliers_detected += num_outliers
        
        # Option 1: Remove the outliers
        # df = df.drop(index=outliers[0])

        # Option 2: Cap the outliers (at the threshold boundary)
        df.loc[z_scores > threshold, col] = np.sign(df[col]) * threshold * df[col].std() + df[col].mean()
        total_outliers_handled += num_outliers  # Assuming all detected outliers were handled
        
        # Option 3: Replace the outliers with NaN
        # df.loc[z_scores > threshold, col] = np.nan
        
    print(f"Total outliers detected: {total_outliers_detected}")
    print(f"Total outliers handled: {total_outliers_handled}")
    return df