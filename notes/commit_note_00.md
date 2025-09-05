# INIT commit

The problem statement is to build an RTGS style AI Analyst for Telangana state open datasets. The first directive suggests that we pick up a dataset but since the title suggests and what was later cleared in the meeting, the goal to my understanding is to make an agentic ai application which can analyze data and come up with insights basically automate the exploratory data analysis procedure so that we spend more time on what matters, crafting policies that make people's lives better.

## My Approach

I plan to build a multi agent platform that automates and simplifies the eda process.
By the end of this hackathon I hope to have 5 agents:

1. `Orchestrator Agent`: The brain. Parses the user's initial request, sequences the workflow, invokes the correct agents, handles errors, and compiles the final report.

2. `Data Ingestion & Profiling Agent`: Loads the data and creates a standardized, initial profile.

3. `Data Cleaning & Transformation Agent`: Diagnoses issues and generates code to fix them.

4. `Insights & Analysis Agent`: Performs EDA, statistical testing, and generates insights and visualizations.

5. `Documentation & Logging Agent`(this one i am not sure of but lets see): Continuously observes and documents the process.

## What I did in First Commit?

This is essentially a cli program which accepts a csv file and performs all of the steps that we need to call our project succesful but its missing the agentic ai workflow making these steps not very useful. It is a very barebones implementation, since all it is doing is using pandas and conditional operators to decide what to do. And as a result the output and insights we are getting are not really useful but it is what we need to get started. It provides me with necessary structure to go about the complete implementation.

## Setup Instructions

1. **Install required dependencies**:

```bash
pip install click pandas numpy matplotlib tabulate
```

2. **Make the script executable** (on Unix-like systems):

```bash
chmod +x ai_data_analytics_tool.py
```

3. **Run the tool**:

```bash
# Show help
python ai_data_analytics_tool.py --help

# Run full analysis on a CSV file
python ai_data_analytics_tool.py full-analysis data.csv

# Run individual commands
python ai_data_analytics_tool.py ingest data.csv
python ai_data_analytics_tool.py insights
python ai_data_analytics_tool.py docs
```

## Key Features

1. **Comprehensive CLI Interface**: Uses Click for a professional command-line interface
2. **Data Ingestion**: Supports CSV, Excel, and JSON formats with automatic standardization
3. **Data Cleaning**: Handles missing values, duplicates, and data type standardization
4. **Insights Generation**: Identifies trends, patterns, gaps, and imbalances
5. **Visualization**: Creates histograms and bar charts saved as image files
6. **Documentation**: Generates detailed Markdown documentation of the entire process
7. **Logging**: Comprehensive logging to both file and console
8. **Error Handling**: Robust error handling with informative messages

## Output Structure

The tool creates the following output structure:

```
output/
├── analysis_ready_data.csv
├── cleaned_transformed_data.csv
├── data_insights.txt
├── process_documentation.md
└── visualizations/
    ├── column1_distribution.png
    ├── column2_distribution.png
    └── ...
```
