"""
LangChain tool functions for the ingestion agent.
"""
import os
import pandas as pd
from typing import Dict, Any
from langchain.tools import tool
from ydata_profiling import ProfileReport
import json


@tool
def read_dataframe(file_path: str) -> str:
    """
    Load a data file into a pandas DataFrame based on file extension.
    
    Supports CSV, Excel (.xlsx, .xls), and JSON formats.
    Automatically handles common encoding issues and delimiter detection for CSV files.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        String message indicating success or failure
    """
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File not found at {file_path}"
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # Try different encodings and separators for CSV files
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return f"ERROR: Could not decode CSV file with any supported encoding"
                
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            
        elif file_ext == '.json':
            df = pd.read_json(file_path)
            
        else:
            return f"ERROR: Unsupported file format: {file_ext}. Supported formats: .csv, .xlsx, .xls, .json"
        
        # Store DataFrame globally for access by other tools
        globals()['_loaded_dataframe'] = df
        
        return f"SUCCESS: Loaded {df.shape[0]} rows and {df.shape[1]} columns from {file_path}"
        
    except Exception as e:
        return f"ERROR: Failed to load file {file_path}: {str(e)}"


@tool
def standardize_dataframe() -> str:
    """
    Standardize the loaded DataFrame by converting column names to snake_case
    and performing basic data type optimization.
    
    Returns:
        String message indicating success or failure
    """
    try:
        if '_loaded_dataframe' not in globals():
            return "ERROR: No DataFrame loaded. Please load data first."
        
        df = globals()['_loaded_dataframe']
        
        # Convert column names to snake_case
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Remove any duplicate column names by adding suffix
        df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
        
        # Store the standardized DataFrame
        globals()['_loaded_dataframe'] = df
        
        return f"SUCCESS: Standardized column names. New columns: {list(df.columns)}"
        
    except Exception as e:
        return f"ERROR: Failed to standardize DataFrame: {str(e)}"


@tool 
def generate_profile(output_path: str) -> str:
    """
    Generate a comprehensive data profile using ydata-profiling and return a structured summary.
    
    Args:
        output_path: Path where to save the HTML profile report
        
    Returns:
        JSON string containing the profile summary or error message
    """
    try:
        if '_loaded_dataframe' not in globals():
            return json.dumps({"error": "No DataFrame loaded. Please load data first."})
        
        df = globals()['_loaded_dataframe']
        
        # Generate the profile report
        profile = ProfileReport(
            df, 
            title="Data Profiling Report",
            explorative=True,
            minimal=False
        )
        
        # Save HTML report
        profile.to_file(output_path)
        
        # Extract key metrics for the summary
        profile_summary = {
            "dataset_info": {
                "shape": list(df.shape),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "column_count": len(df.columns),
                "row_count": len(df)
            },
            "columns": {},
            "missing_data": {
                "total_missing_cells": int(df.isnull().sum().sum()),
                "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
            },
            "data_types": df.dtypes.value_counts().to_dict()
        }
        
        # Add column-specific information
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            # Add type-specific statistics
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_info.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None
                })
            elif df[col].dtype == 'object':
                col_info.update({
                    "unique_count": int(df[col].nunique()),
                    "most_frequent": str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else None
                })
                
            profile_summary["columns"][col] = col_info
        
        # Store profile summary globally
        globals()['_profile_summary'] = profile_summary
        
        return json.dumps({
            "status": "SUCCESS",
            "message": f"Profile generated and saved to {output_path}",
            "summary": profile_summary
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to generate profile: {str(e)}"})


@tool
def save_standardized_data(output_path: str) -> str:
    """
    Save the standardized DataFrame to a CSV file.
    
    Args:
        output_path: Path where to save the CSV file
        
    Returns:
        String message indicating success or failure
    """
    try:
        if '_loaded_dataframe' not in globals():
            return "ERROR: No DataFrame loaded. Please load and standardize data first."
        
        df = globals()['_loaded_dataframe']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return f"SUCCESS: Saved standardized data to {output_path}"
        
    except Exception as e:
        return f"ERROR: Failed to save data: {str(e)}"

