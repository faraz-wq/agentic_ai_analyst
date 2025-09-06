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

    print(f"Attempting to read file: {file_path}")  # Debug statement
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
    
    print(f"Attempting to standardize DataFrame")  # Debug statement
    try:
        if '_loaded_dataframe' not in globals():
            return "ERROR: No DataFrame loaded. Please load data first."
        
        df = globals()['_loaded_dataframe']
        
        # Convert column names to snake_case
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # Remove any duplicate column names by adding suffix
        df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)
        
        print(f"Standardized columns: {list(df.columns)}")  # Debug: print standardized column names
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
    print(f"Attempting to generate profile report at: {output_path}")  # Debug statement
    try:
        if '_loaded_dataframe' not in globals():
            return json.dumps({"error": "No DataFrame loaded. Please load data first."})
        
        df = globals()['_loaded_dataframe']
        
        # Fix: Convert nullable dtypes to standard dtypes
        df = df.convert_dtypes()
        df = df.astype({col: 'float64' for col in df.select_dtypes(include=['Float64', 'Float32']).columns})
        
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
            "data_types": {str(k): v for k, v in df.dtypes.value_counts().to_dict().items()}
        }
        
        # Add column-specific information
        for col in df.columns:
            col_info = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
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
    
    print(f"Attempting to save standardized DataFrame to: {output_path}")  # Debug statement
    try:
        if '_loaded_dataframe' not in globals():
            return "ERROR: No DataFrame loaded. Please load and standardize data first."
        
        df = globals()['_loaded_dataframe']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        print(globals()['_loaded_dataframe'].head())
        return f"SUCCESS: Saved standardized data to {output_path}"
        
    except Exception as e:
        return f"ERROR: Failed to save data: {str(e)}"

@tool
def analyze_data_quality(profile_json: str) -> str:
    """
    Analyze the data profile to identify and prioritize data quality issues.
    
    Args:
        profile_json: JSON string containing the data profile summary
        
    Returns:
        JSON string containing prioritized list of issues found
    """
    
    print(f"Attempting to analyze data quality")  # Debug statement
    try:
        profile = json.loads(profile_json)
        issues = []
        
        # Check for missing values
        for col_name, col_info in profile.get("columns", {}).items():
            null_pct = col_info.get("null_percentage", 0)
            
            if null_pct > 50:
                issues.append({
                    "issue": "high_missing_values",
                    "column": col_name,
                    "severity": "high",
                    "details": f"{null_pct}% null values",
                    "priority": 1
                })
            elif null_pct > 10:
                issues.append({
                    "issue": "moderate_missing_values",
                    "column": col_name,
                    "severity": "medium",
                    "details": f"{null_pct}% null values",
                    "priority": 2
                })
        
        # Check for data type issues (objects that might be numbers/dates)
        for col_name, col_info in profile.get("columns", {}).items():
            if col_info.get("dtype") == "object":
                # Check if it might be a numeric column stored as string
                if col_info.get("unique_count", 0) > 10:  # Likely not categorical
                    issues.append({
                        "issue": "potential_numeric_as_string",
                        "column": col_name,
                        "severity": "medium",
                        "details": f"Object column with {col_info.get('unique_count')} unique values",
                        "priority": 3
                    })
        
        # Check for potential duplicates (this is a heuristic)
        dataset_info = profile.get("dataset_info", {})
        row_count = dataset_info.get("row_count", 0)
        total_columns = dataset_info.get("column_count", 1)
        
        # If we have many rows, duplicates are likely
        if row_count > 1000:
            issues.append({
                "issue": "potential_duplicates",
                "column": "all",
                "severity": "low",
                "details": f"Large dataset ({row_count} rows) may contain duplicates",
                "priority": 4
            })
        
        # Sort by priority
        issues.sort(key=lambda x: x["priority"])
        
        # Store issues globally for access by other tools
        globals()['_detected_issues'] = issues
        
        return json.dumps({
            "status": "SUCCESS",
            "issues_found": len(issues),
            "issues": issues
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to analyze data quality: {str(e)}"})


@tool
def handle_missing_values(strategy: str, columns: str) -> str:
    """
    Handle missing values in specified columns using the given strategy.
    
    Args:
        strategy: Strategy to use ('drop', 'mean', 'median', 'mode', 'fillna_zero')
        columns: Comma-separated list of column names, or 'all' for all columns
        
    Returns:
        String message indicating success or failure
    """
    
    print(f"Attempting to handle missing values using strategy: {strategy} on columns: {columns}")  # Debug statement
    try:
        if '_working_dataframe' not in globals():
            return "ERROR: No working DataFrame available. Please load data first."
        
        df = globals()['_working_dataframe'].copy()
        actions = []
        
        # Parse columns
        if columns.lower() == 'all':
            target_columns = df.columns.tolist()
        else:
            target_columns = [col.strip() for col in columns.split(',')]
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            
            if strategy == 'drop':
                df = df.dropna(subset=[col])
                actions.append({
                    "action": "drop_missing",
                    "column": col,
                    "rows_removed": missing_count
                })
            elif strategy == 'mean' and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                actions.append({
                    "action": "impute_missing",
                    "column": col,
                    "strategy": "mean",
                    "value": fill_value
                })
            elif strategy == 'median' and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                actions.append({
                    "action": "impute_missing",
                    "column": col,
                    "strategy": "median",
                    "value": fill_value
                })
            elif strategy == 'mode':
                if len(df[col].mode()) > 0:
                    fill_value = df[col].mode().iloc[0]
                    df[col].fillna(fill_value, inplace=True)
                    actions.append({
                        "action": "impute_missing",
                        "column": col,
                        "strategy": "mode",
                        "value": fill_value
                    })
            elif strategy == 'fillna_zero':
                df[col].fillna(0, inplace=True)
                actions.append({
                    "action": "impute_missing",
                    "column": col,
                    "strategy": "zero",
                    "value": 0
                })
        
        # Update working dataframe
        globals()['_working_dataframe'] = df
        
        # Store actions
        if '_cleaning_actions' not in globals():
            globals()['_cleaning_actions'] = []
        globals()['_cleaning_actions'].extend(actions)
        
        return f"SUCCESS: Applied {strategy} strategy to {len(target_columns)} columns. Actions: {len(actions)}"
        
    except Exception as e:
        return f"ERROR: Failed to handle missing values: {str(e)}"


@tool
def correct_data_types(type_corrections: str) -> str:
    """
    Convert columns to correct data types based on provided mapping.
    
    Args:
        type_corrections: JSON string with column->type mapping, e.g. '{"col1": "int64", "col2": "datetime"}'
        
    Returns:
        String message indicating success or failure
    """
    try:
        if '_working_dataframe' not in globals():
            return "ERROR: No working DataFrame available. Please load data first."
        
        df = globals()['_working_dataframe'].copy()
        type_map = json.loads(type_corrections)
        actions = []
        
        for col, target_type in type_map.items():
            if col not in df.columns:
                continue
            
            original_type = str(df[col].dtype)
            
            try:
                if target_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif target_type == 'numeric':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif target_type in ['int64', 'int32']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(target_type)
                elif target_type in ['float64', 'float32']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(target_type)
                elif target_type == 'category':
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(target_type)
                
                actions.append({
                    "action": "convert_dtype",
                    "column": col,
                    "from": original_type,
                    "to": str(df[col].dtype)
                })
                
            except Exception as e:
                actions.append({
                    "action": "convert_dtype_failed",
                    "column": col,
                    "error": str(e)
                })
        
        # Update working dataframe
        globals()['_working_dataframe'] = df
        
        # Store actions
        if '_cleaning_actions' not in globals():
            globals()['_cleaning_actions'] = []
        globals()['_cleaning_actions'].extend(actions)
        
        return f"SUCCESS: Attempted type conversions for {len(type_map)} columns. Successful: {len([a for a in actions if 'failed' not in a['action']])}"
        
    except Exception as e:
        return f"ERROR: Failed to correct data types: {str(e)}"


@tool
def remove_duplicates() -> str:
    """
    Remove duplicate rows from the DataFrame.
    
    Returns:
        String message indicating success or failure
    """
    try:
        if '_working_dataframe' not in globals():
            return "ERROR: No working DataFrame available. Please load data first."
        
        df = globals()['_working_dataframe'].copy()
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = original_count - len(df)
        
        # Update working dataframe
        globals()['_working_dataframe'] = df
        
        # Store action
        action = {
            "action": "remove_duplicates",
            "column": "all",
            "duplicates_removed": duplicates_removed,
            "final_row_count": len(df)
        }
        
        if '_cleaning_actions' not in globals():
            globals()['_cleaning_actions'] = []
        globals()['_cleaning_actions'].append(action)
        
        return f"SUCCESS: Removed {duplicates_removed} duplicate rows. Final count: {len(df)} rows"
        
    except Exception as e:
        return f"ERROR: Failed to remove duplicates: {str(e)}"


@tool
def standardize_categorical_values(column: str, standardization_rules: str) -> str:
    """
    Standardize categorical values in a column based on provided rules.
    
    Args:
        column: Name of the column to standardize
        standardization_rules: JSON string with mapping, e.g. '{"USA": "United States", "U.S.A": "United States"}'
        
    Returns:
        String message indicating success or failure
    """
    try:
        if '_working_dataframe' not in globals():
            return "ERROR: No working DataFrame available. Please load data first."
        
        df = globals()['_working_dataframe'].copy()
        
        if column not in df.columns:
            return f"ERROR: Column '{column}' not found in DataFrame"
        
        rules = json.loads(standardization_rules)
        
        # Apply standardization rules
        df[column] = df[column].replace(rules)
        
        # Update working dataframe
        globals()['_working_dataframe'] = df
        
        # Store action
        action = {
            "action": "standardize_categorical",
            "column": column,
            "rules_applied": len(rules),
            "unique_values_after": df[column].nunique()
        }
        
        if '_cleaning_actions' not in globals():
            globals()['_cleaning_actions'] = []
        globals()['_cleaning_actions'].append(action)
        
        return f"SUCCESS: Applied {len(rules)} standardization rules to column '{column}'"
        
    except Exception as e:
        return f"ERROR: Failed to standardize categorical values: {str(e)}"


@tool
def validate_cleaning_results() -> str:
    """
    Compare data before and after cleaning to validate improvements.
    
    Returns:
        JSON string containing validation metrics
    """
    try:
        if '_working_dataframe' not in globals():
            return json.dumps({"error": "No working DataFrame available"})
        
        # Get original data from raw_data or loaded dataframe
        original_df = None
        if '_original_dataframe' in globals():
            original_df = globals()['_original_dataframe']
        else:
            return json.dumps({"error": "No original DataFrame available for comparison"})
        
        cleaned_df = globals()['_working_dataframe']
        
        # Calculate metrics
        validation_metrics = {
            "row_count_change": {
                "before": len(original_df),
                "after": len(cleaned_df),
                "change": len(cleaned_df) - len(original_df)
            },
            "missing_values": {
                "before": int(original_df.isnull().sum().sum()),
                "after": int(cleaned_df.isnull().sum().sum()),
                "reduction": int(original_df.isnull().sum().sum()) - int(cleaned_df.isnull().sum().sum())
            },
            "data_types": {
                "before": original_df.dtypes.value_counts().to_dict(),
                "after": cleaned_df.dtypes.value_counts().to_dict()
            }
        }
        
        # Store validation results
        globals()['_validation_results'] = validation_metrics
        
        return json.dumps({
            "status": "SUCCESS",
            "validation_metrics": validation_metrics
        })
        
    except Exception as e:
        return json.dumps({"error": f"Failed to validate cleaning results: {str(e)}"})


@tool
def initialize_cleaning_dataframe() -> str:
    """
    Initialize the working dataframe for cleaning operations from the loaded raw data.
    
    Returns:
        String message indicating success or failure
    """
    try:
        # Check if we have raw data loaded (from ingestion agent)
        if '_loaded_dataframe' not in globals():
            return "ERROR: No raw data available. Please run ingestion first."
        
        # Copy the raw data to working dataframe
        if '_loaded_dataframe' not in globals():
            return "ERROR: No raw DataFrame loaded. Please load data first."
        raw_df = globals()['_loaded_dataframe']
        globals()['_working_dataframe'] = raw_df.copy()
        globals()['_original_dataframe'] = raw_df.copy()  # Keep original for comparison
        
        # Initialize cleaning actions list
        globals()['_cleaning_actions'] = []
        
        return f"SUCCESS: Initialized cleaning dataframe with {raw_df.shape[0]} rows and {raw_df.shape[1]} columns"
        
    except Exception as e:
        return f"ERROR: Failed to initialize cleaning dataframe: {str(e)}"


@tool
def save_cleaned_data(output_path: str) -> str:
    """
    Save the cleaned DataFrame to a CSV file.
    
    Args:
        output_path: Path where to save the cleaned CSV file
        
    Returns:
        String message indicating success or failure
    """
    try:
        if '_working_dataframe' not in globals():
            return "ERROR: No cleaned DataFrame available. Please perform cleaning operations first."
        
        df = globals()['_working_dataframe']
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return f"SUCCESS: Saved cleaned data to {output_path}. Shape: {df.shape}"
        
    except Exception as e:
        return f"ERROR: Failed to save cleaned data: {str(e)}"

