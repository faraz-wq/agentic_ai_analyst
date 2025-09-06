# Data Cleaning Agent

This commit introduces significant enhancements to a data processing pipeline by adding a new data cleaning and transformation agent, updating the ingestion agent, expanding the core state management, and improving the toolset for data handling. Below is a detailed breakdown of the changes made in each modified file, organized by file and the purpose of the changes.

---

## 1. `.gitignore`

**Changes:**

- Added additional file extensions to be ignored in the `output/` directory:
  - `output/*.json`
  - `output/*.txt`
  - `output/*.xlsx`
  - `output/*.md`
  - `output/*.pdf`

**Purpose:**

- These additions ensure that various output file types generated during data processing (e.g., JSON data, text logs, Excel files, Markdown reports, and PDFs) are excluded from version control, keeping the repository clean and focused on source code.

---

## 2. `agents/cleaning_agent.py`

**Changes:**

- **New File Creation**: This is a newly added file that implements a `Data Cleaning & Transformation Agent` to handle data quality issues systematically.
- **Key Components**:
  - **Agent Creation (`create_cleaning_agent`)**:
    - Configures a `ChatOpenAI` model using the Cerebras API (`qwen-3-235b-a22b-instruct-2507`) with a low temperature (0.1) for consistent responses and a high token limit (2000).
    - Defines a detailed system prompt outlining a strict workflow:
      1. Initialize a cleaning DataFrame.
      2. Analyze data quality issues from a provided profile.
      3. Formulate a cleaning plan based on issue severity.
      4. Execute cleaning operations in a specific order (e.g., handle missing values, correct data types, remove duplicates, standardize categories).
      5. Validate results and save the cleaned dataset.
    - Specifies decision-making guidelines for handling missing values, data types, duplicates, and categorical standardization.
    - Uses a set of tools (defined in `core/tools.py`) for cleaning operations.
    - Returns an `AgentExecutor` configured with verbose output, error handling, and a maximum of 15 iterations.
  - **Agent Execution (`run_cleaning_agent`)**:
    - Takes an `AgentState` object as input, which contains raw data and a data profile from the ingestion phase.
    - Validates prerequisites (e.g., ingestion completion, presence of raw data and profile).
    - Ensures the output directory exists and logs its path.
    - Copies raw data to a global variable (`_loaded_dataframe`) for use by cleaning tools.
    - Prepares a detailed input for the agent, including dataset shape, output directory, and data profile.
    - Executes the agent and logs intermediate steps, validation metrics, and errors.
    - Loads the cleaned DataFrame from the saved file (`cleaned_data.csv`) with retry logic (up to 3 attempts with 0.5-second delays).
    - Generates a comprehensive cleaning report in Markdown format, including:
      - Summary (issues detected, actions performed, dataset shapes).
      - Detailed issues and actions.
      - Validation metrics (row count changes, missing value reduction, data type changes).
      - Recommendations for future data collection.
    - Saves the report as `cleaning_report.md` in the output directory.
    - Updates the `AgentState` with cleaned data, cleaning report, issues, actions, and logs.
    - Cleans up global variables to prevent memory leaks.
  - **Report Generation (`generate_cleaning_report_markdown`)**:
    - Creates a formatted Markdown report detailing the cleaning process.
    - Includes sections for executive summary, issues detected, actions performed, validation results, and recommendations.
    - Uses tables and structured formatting for clarity.

**Purpose:**

- Introduces a robust data cleaning pipeline that systematically addresses data quality issues, validates improvements, and generates detailed reports.
- Enhances the pipeline’s ability to produce high-quality, clean datasets for downstream analysis.
- Ensures transparency and traceability through comprehensive logging and reporting.

---

## 3. `agents/ingestion_agent.py`

**Changes:**

- **LLM Configuration Update**:
  - Changed the base URL from `https://openrouter.ai/api/v1` to `https://api.cerebras.ai/v1`.
  - Updated the model to `qwen-3-235b-a22b-instruct-2507` (previously `qwen/qwen3-coder:free`).
- **State Handling**:
  - Initialized `updated_state['logs']` to ensure logs are always available.
- **Result Extraction**:
  - Modified how the data profile is extracted from intermediate steps (instead of relying on global `_profile_summary`).
  - Added logic to load the standardized DataFrame from `raw_data.csv` in the output directory, avoiding reliance on global `_loaded_dataframe`.
  - Added debug logging for the DataFrame head (commented out).
- **Ingestion Completion Logic**:
  - Updated the `ingestion_complete` flag to be set only if both `raw_data` and `data_profile` are present, improving robustness.
- **Global Variable Cleanup**:
  - Ensured cleanup of `_loaded_dataframe` and `_profile_summary` if they exist.
- **Documentation**:
  - Removed redundant comments about the ingestion process to streamline the code.

**Purpose:**

- Improves the ingestion agent’s reliability by reducing dependency on global variables and enhancing error handling.
- Aligns the LLM configuration with the cleaning agent for consistency.
- Enhances logging and state management to better integrate with the new cleaning agent.

---

## 4. `core/state.py`

**Changes:**

- **New Fields Added**:
  - `cleaned_data`: Stores the cleaned and transformed DataFrame.
  - `cleaning_report`: Stores a structured report of cleaning actions.
  - `issues_detected`: Lists identified data quality issues.
  - `actions_performed`: Lists cleaning actions taken.
- **Code Cleanup**:
  - Removed unnecessary blank lines for better readability.

**Purpose:**

- Expands the `AgentState` model to support the cleaning agent’s outputs, enabling seamless state transfer between ingestion and cleaning phases.
- Provides a structured way to store and pass cleaning-related metadata.

---

## 5. `core/tools.py`

**Changes:**

- **Debug Statements**:
  - Added `print` statements for debugging in `read_dataframe`, `standardize_dataframe`, `generate_profile`, and `save_standardized_data`.
- **Data Type Handling in `generate_profile`**:
  - Added conversion of nullable dtypes (`Float64`, `Float32`) to standard `float64` to avoid serialization issues.
  - Converted data type dictionary keys to strings in the profile summary for JSON compatibility.
- **New Cleaning Tools**:
  - **analyze_data_quality**:
    - Analyzes a data profile to identify issues (e.g., high/moderate missing values, potential numeric columns stored as strings, potential duplicates).
    - Prioritizes issues by severity (high, medium, low) and stores them in `_detected_issues`.
    - Returns a JSON string with issue details.
  - **handle_missing_values**:
    - Handles missing values using strategies like `drop`, `mean`, `median`, `mode`, or `fillna_zero`.
    - Supports specific columns or all columns.
    - Logs actions (e.g., rows dropped, imputation values) in `_cleaning_actions`.
  - **correct_data_types**:
    - Converts column data types based on a provided JSON mapping (e.g., to `int64`, `float64`, `datetime`, `category`).
    - Handles errors gracefully and logs actions in `_cleaning_actions`.
  - **remove_duplicates**:
    - Removes duplicate rows from the DataFrame and logs the number of duplicates removed in `_cleaning_actions`.
  - **standardize_categorical_values**:
    - Standardizes categorical values in a column using a provided JSON mapping (e.g., mapping “USA” to “United States”).
    - Logs the number of rules applied and unique values after standardization in `_cleaning_actions`.
  - **validate_cleaning_results**:
    - Compares original and cleaned DataFrames to calculate metrics (row count changes, missing value reduction, data type changes).
    - Converts nullable dtypes to standard types for consistency.
    - Stores metrics in `_validation_results` and returns them as JSON.
  - **initialize_cleaning_dataframe**:
    - Initializes a working DataFrame (`_working_dataframe`) from the raw data (`_loaded_dataframe`) and keeps a copy of the original (`_original_dataframe`) for comparison.
    - Initializes `_cleaning_actions` for tracking cleaning operations.
  - **save_cleaned_data**:
    - Saves the cleaned DataFrame to a CSV file at the specified path.
    - Ensures the output directory exists and logs the shape of the saved DataFrame.
- **Global Variables**:
  - Introduced `_working_dataframe`, `_original_dataframe`, `_detected_issues`, `_cleaning_actions`, and `_validation_results` for tracking cleaning operations.

**Purpose:**

- Expands the toolset to support comprehensive data cleaning operations, including missing value handling, type correction, duplicate removal, and categorical standardization.
- Enhances debugging with print statements and improves data type handling for robustness.
- Provides structured outputs and global storage for cleaning metadata, enabling validation and reporting.

---

## 6. `orchestrator.py`

**Changes:**

- **Pipeline Expansion**:
  - Extended the orchestration logic to include both ingestion and cleaning agents.
  - Runs the ingestion agent first, checks for errors, and proceeds to the cleaning agent only if ingestion is successful.
- **Output Directory**:
  - Updated `output_dir` to `./output/` for clarity and consistency.
- **Result Display**:
  - Prints detailed results after ingestion (data shape) and cleaning (issues detected, actions performed, original/cleaned shapes).
  - Lists generated files (`raw_data.csv`, `profile.html`, `cleaned_data.csv`, `cleaning_report.md`).
- **Logging**:
  - Prints all logs from the final state for transparency.

**Purpose:**

- Integrates the new cleaning agent into the pipeline, creating a complete workflow from data ingestion to cleaning.
- Improves user feedback by displaying detailed results and generated files.
- Ensures robust error handling and conditional execution to prevent running cleaning on failed ingestion.

---

## Summary of Key Enhancements

1. **New Cleaning Agent**:
   - A dedicated agent for data cleaning and transformation was added, with a systematic workflow for analyzing, planning, executing, and validating cleaning operations.
   - Generates detailed Markdown reports for transparency and traceability.
2. **Improved Ingestion Agent**:
   - Reduced reliance on global variables, improved error handling, and aligned LLM configuration with the cleaning agent.
3. **Expanded State Management**:
   - Added fields to `AgentState` to support cleaning results, ensuring seamless data and metadata transfer between agents.
4. **Enhanced Toolset**:
   - Added a suite of cleaning tools to handle common data quality issues (missing values, incorrect types, duplicates, inconsistent categories).
   - Improved data type handling and debugging for robustness.
5. **Orchestration**:
   - Extended the pipeline to include cleaning, with clear feedback on results and generated outputs.
6. **Gitignore Updates**:
   - Expanded `.gitignore` to cover additional output file types, keeping the repository clean.

---

## Overall Impact

This commit transforms the project into a comprehensive data processing pipeline that not only ingests and profiles data but also systematically cleans and transforms it. The cleaning agent follows a rigorous workflow, ensuring high-quality data output, while the updated ingestion agent and tools enhance reliability and integration. The orchestrator ties these components together, providing a clear and user-friendly interface for the entire process. The detailed logging and reporting mechanisms ensure transparency, making the pipeline suitable for production-grade data processing tasks.
