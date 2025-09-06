import os
import json
from typing import Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from core.state import AgentState
from core.tools import (
    analyze_data_quality,
    handle_missing_values,
    correct_data_types,
    remove_duplicates,
    standardize_categorical_values,
    validate_cleaning_results,
    initialize_cleaning_dataframe,
    save_cleaned_data,
)


def create_cleaning_agent() -> AgentExecutor:
    """
    Create and configure the Data Cleaning & Transformation Agent.

    This agent is responsible for:
    1. Analyzing data quality issues from the profile
    2. Formulating a cleaning plan
    3. Executing cleaning operations
    4. Validating results
    5. Generating detailed reports

    Returns:
        AgentExecutor: Configured agent ready for execution
    """

    # Initialize the LLM
    llm = ChatOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="qwen-3-235b-a22b-instruct-2507",
        temperature=0.1,  # Low temperature for consistent, factual responses
        max_tokens=2000,
    )

    # Define the sophisticated system prompt
    system_prompt = """
    You are the Data Cleaning & Transformation Agent, an expert data quality specialist in a multi-agent analytics system.

    Your mission is to intelligently analyze data quality issues and execute a comprehensive cleaning strategy.

    WORKFLOW SEQUENCE (MUST FOLLOW EXACTLY):
    
    1. INITIALIZATION:
       - First, use initialize_cleaning_dataframe to set up the working dataset
    
    2. ANALYSIS PHASE:
       - Use analyze_data_quality with the provided data profile to identify issues
       - Prioritize issues by severity and impact
    
    3. PLANNING PHASE:
       - Formulate a logical sequence of cleaning operations
       - Consider dependencies (e.g., fix data types before statistical operations)
    
    4. EXECUTION PHASE:
       - Execute cleaning operations in the correct order:
         a) Handle severe missing values first (drop or impute)
         b) Correct data types (convert strings to numbers/dates)
         c) Remove duplicates
         d) Standardize categorical values
         e) Handle remaining missing values
    
    5. VALIDATION PHASE:
       - Use validate_cleaning_results to compare before/after metrics
       - Save the cleaned dataset using save_cleaned_data
    
    DECISION MAKING GUIDELINES:
    
    Missing Values:
    - >50% missing: Consider dropping the column or rows
    - 10-50% missing: Use appropriate imputation (mean for numeric, mode for categorical)
    - <10% missing: Use median for numeric, mode for categorical
    
    Data Types:
    - Look for numeric data stored as strings (high unique count in object columns)
    - Detect date columns (object columns with date-like patterns)
    - Convert categorical columns with low unique counts to category type
    
    Duplicates:
    - Always check for and remove exact duplicates
    - Be cautious with partial duplicates
    
    Categorical Standardization:
    - Look for obvious inconsistencies (case differences, abbreviations)
    - Only standardize when you're confident about the mapping
    
    CRITICAL REQUIREMENTS:
    - Document EVERY action taken with clear reasoning
    - If an operation fails, continue with other operations and note the failure
    - Always validate results before finishing
    - Provide detailed summary of all changes made
    
    ERROR HANDLING:
    - If any tool fails, document the error and continue with remaining tasks
    - Never stop the entire workflow due to one failed operation
    
    Your response should be comprehensive and include:
    - List of issues identified
    - Reasoning for chosen strategies
    - Summary of actions performed
    - Validation of improvements achieved
    """

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Define available tools
    tools = [
        initialize_cleaning_dataframe,
        analyze_data_quality,
        handle_missing_values,
        correct_data_types,
        remove_duplicates,
        standardize_categorical_values,
        validate_cleaning_results,
        save_cleaned_data,
    ]

    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)

    # Create and return the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15,  # More iterations for complex cleaning
        return_intermediate_steps=True,
    )

    return agent_executor


def run_cleaning_agent(state: AgentState) -> Dict[str, Any]:
    """
    Execute the Data Cleaning & Transformation Agent workflow.

    This function orchestrates the complete cleaning process:
    1. Validates prerequisite data
    2. Invokes the cleaning agent
    3. Updates the state with results
    4. Saves outputs and generates reports

    Args:
        state: The current AgentState containing data from ingestion

    Returns:
        Dict containing the updated state information
    """

    # Convert state to dict for easier manipulation
    updated_state = state.dict()

    try:
        # Validate prerequisites
        if not state.ingestion_complete:
            raise ValueError(
                "Ingestion must be completed before cleaning. Please run ingestion agent first."
            )

        if state.raw_data is None:
            raise ValueError(
                "No raw data available. Please ensure ingestion completed successfully."
            )

        if state.data_profile is None:
            raise ValueError(
                "No data profile available. Please ensure ingestion completed successfully."
            )

        # Ensure output directory exists
        os.makedirs(state.output_dir, exist_ok=True)

        # Store raw data globally for the cleaning tools to access
        globals()["_loaded_dataframe"] = state.raw_data.copy()

        # Create agent
        agent_executor = create_cleaning_agent()

        # Prepare agent input with data profile
        profile_json = json.dumps(state.data_profile)

        agent_input = {
            "input": f"""
            Please perform comprehensive data cleaning and transformation on the dataset.
            
            CONTEXT:
            - Dataset shape: {state.raw_data.shape}
            - Output directory: {state.output_dir}
            - Raw data is loaded and ready for processing
            
            DATA PROFILE SUMMARY:
            {profile_json}
            
            TASKS TO COMPLETE:
            1. Initialize the cleaning dataframe from the raw data
            2. Analyze the data profile to identify quality issues
            3. Execute appropriate cleaning operations based on the issues found
            4. Validate the cleaning results
            5. Save the cleaned dataset as 'cleaned_data.csv' in the output directory
            
            Follow the systematic approach outlined in your instructions and provide a comprehensive 
            summary of all issues found and actions taken.
            """
        }

        # Execute the agent
        result = agent_executor.invoke(agent_input)

        # Extract the agent's response
        agent_response = result.get("output", "")
        updated_state["logs"].append(f"Cleaning Agent: {agent_response}")

        # Get results from global variables set by tools
        if "_working_dataframe" in globals():
            updated_state["cleaned_data"] = globals()["_working_dataframe"]
            updated_state["logs"].append(
                f"Successfully cleaned data. New shape: {globals()['_working_dataframe'].shape}"
            )

        if "_detected_issues" in globals():
            updated_state["issues_detected"] = globals()["_detected_issues"]
            updated_state["logs"].append(
                f"Detected {len(globals()['_detected_issues'])} data quality issues"
            )

        if "_cleaning_actions" in globals():
            updated_state["actions_performed"] = globals()["_cleaning_actions"]
            updated_state["logs"].append(
                f"Performed {len(globals()['_cleaning_actions'])} cleaning actions"
            )

        if "_validation_results" in globals():
            validation_metrics = globals()["_validation_results"]
        else:
            validation_metrics = {}

        # Create comprehensive cleaning report
        cleaning_report = {
            "summary": {
                "issues_detected": len(updated_state.get("issues_detected", [])),
                "actions_performed": len(updated_state.get("actions_performed", [])),
                "original_shape": (
                    state.raw_data.shape if state.raw_data is not None else [0, 0]
                ),
                "cleaned_shape": (
                    updated_state["cleaned_data"].shape
                    if updated_state.get("cleaned_data") is not None
                    else [0, 0]
                ),
            },
            "issues_detected": updated_state.get("issues_detected", []),
            "actions_performed": updated_state.get("actions_performed", []),
            "validation_metrics": validation_metrics,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        updated_state["cleaning_report"] = cleaning_report

        # Generate and save markdown report
        report_path = os.path.join(state.output_dir, "cleaning_report.md")
        markdown_content = generate_cleaning_report_markdown(cleaning_report)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        updated_state["logs"].append(f"Generated cleaning report: {report_path}")

        # Mark cleaning as complete
        updated_state["cleaning_complete"] = True
        updated_state["logs"].append("Cleaning agent completed successfully")

        # Clean up global variables
        for var in [
            "_loaded_dataframe",
            "_working_dataframe",
            "_original_dataframe",
            "_detected_issues",
            "_cleaning_actions",
            "_validation_results",
        ]:
            if var in globals():
                del globals()[var]

    except Exception as e:
        error_msg = f"Cleaning agent failed: {str(e)}"
        updated_state["error"] = error_msg
        updated_state["logs"].append(error_msg)
        updated_state["cleaning_complete"] = False

    return updated_state


def generate_cleaning_report_markdown(cleaning_report: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report of the cleaning process.

    Args:
        cleaning_report: Dictionary containing cleaning results

    Returns:
        String containing markdown-formatted report
    """

    summary = cleaning_report.get("summary", {})
    issues = cleaning_report.get("issues_detected", [])
    actions = cleaning_report.get("actions_performed", [])
    validation = cleaning_report.get("validation_metrics", {})

    markdown = f"""# Data Cleaning & Transformation Report

Generated on: {cleaning_report.get('timestamp', 'Unknown')}

## Executive Summary

- **Issues Detected**: {summary.get('issues_detected', 0)}
- **Actions Performed**: {summary.get('actions_performed', 0)}
- **Original Dataset Shape**: {summary.get('original_shape', [0, 0])}
- **Cleaned Dataset Shape**: {summary.get('cleaned_shape', [0, 0])}

"""

    # Issues Section
    if issues:
        markdown += """## Issues Detected

                The following data quality issues were identified during analysis:

                | Issue Type | Column | Severity | Details |
                |------------|--------|----------|---------|
                """
        for issue in issues:
            markdown += f"| {issue.get('issue', 'Unknown')} | {issue.get('column', 'N/A')} | {issue.get('severity', 'Unknown')} | {issue.get('details', 'No details')} |\n"

    # Actions Section
    if actions:
        markdown += """

            ## Actions Performed

            The following cleaning operations were executed:

            """
        for i, action in enumerate(actions, 1):
            action_type = action.get("action", "Unknown")
            column = action.get("column", "N/A")

            markdown += f"### {i}. {action_type.replace('_', ' ').title()}\n\n"
            markdown += f"- **Column**: {column}\n"

            # Add specific details based on action type
            if action_type == "impute_missing":
                markdown += f"- **Strategy**: {action.get('strategy', 'Unknown')}\n"
                markdown += f"- **Value Used**: {action.get('value', 'N/A')}\n"
            elif action_type == "convert_dtype":
                markdown += f"- **From Type**: {action.get('from', 'Unknown')}\n"
                markdown += f"- **To Type**: {action.get('to', 'Unknown')}\n"
            elif action_type == "remove_duplicates":
                markdown += (
                    f"- **Duplicates Removed**: {action.get('duplicates_removed', 0)}\n"
                )
                markdown += f"- **Final Row Count**: {action.get('final_row_count', 'Unknown')}\n"
            elif action_type == "standardize_categorical":
                markdown += f"- **Rules Applied**: {action.get('rules_applied', 0)}\n"
                markdown += f"- **Unique Values After**: {action.get('unique_values_after', 'Unknown')}\n"

            markdown += "\n"

    # Validation Section
    if validation:
        markdown += """## Validation Results

### Data Quality Improvements

"""

        # Row count changes
        row_changes = validation.get("row_count_change", {})
        if row_changes:
            markdown += f"**Row Count**: {row_changes.get('before', 0)} → {row_changes.get('after', 0)} "
            change = row_changes.get("change", 0)
            if change > 0:
                markdown += f"(+{change} rows)\n\n"
            elif change < 0:
                markdown += f"({change} rows)\n\n"
            else:
                markdown += "(no change)\n\n"

        # Missing values improvements
        missing_changes = validation.get("missing_values", {})
        if missing_changes:
            before = missing_changes.get("before", 0)
            after = missing_changes.get("after", 0)
            reduction = missing_changes.get("reduction", 0)
            markdown += f"**Missing Values**: {before} → {after} "
            if reduction > 0:
                markdown += f"(-{reduction} missing values)\n\n"
            else:
                markdown += "(no change)\n\n"

        # Data types
        type_changes = validation.get("data_types", {})
        if type_changes:
            markdown += "**Data Types**:\n\n"
            before_types = type_changes.get("before", {})
            after_types = type_changes.get("after", {})

            markdown += "Before:\n"
            for dtype, count in before_types.items():
                markdown += f"- {dtype}: {count} columns\n"

            markdown += "\nAfter:\n"
            for dtype, count in after_types.items():
                markdown += f"- {dtype}: {count} columns\n"

    # Recommendations section
    markdown += """

        ## Recommendations

        Based on the cleaning process, consider the following for future data collection and processing:

        """

    if issues:
        high_severity_issues = [i for i in issues if i.get("severity") == "high"]
        if high_severity_issues:
            markdown += "### High Priority\n"
            for issue in high_severity_issues:
                if issue.get("issue") == "high_missing_values":
                    markdown += f"- **{issue.get('column')}**: Consider improving data collection processes to reduce missing values\n"

        medium_severity_issues = [i for i in issues if i.get("severity") == "medium"]
        if medium_severity_issues:
            markdown += "\n### Medium Priority\n"
            for issue in medium_severity_issues:
                if "data_type" in issue.get("issue", ""):
                    markdown += f"- **{issue.get('column')}**: Ensure proper data type validation during data entry\n"

    markdown += """
        ### General Recommendations
        - Implement data validation rules at the source
        - Regular data quality monitoring
        - Standardize data entry processes
        - Consider automated data cleaning pipelines for similar datasets

        ---
        *Report generated by the Data Cleaning & Transformation Agent*
        """

    return markdown
