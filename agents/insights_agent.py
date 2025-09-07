# agents/insights_agent.py
# Uses a custom agent loop to avoid response_format and agent_scratchpad issues.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os
from core.state import AgentState
from core.tools import (
    perform_eda_summary,
    analyze_correlations,
    identify_trends,
    detect_imbalances,
    run_statistical_tests,
    create_visualization,
    generate_insights_summary,
)
from typing import Dict, List
import pandas as pd
import json
import re


def create_insights_agent() -> ChatOpenAI:
    """
    Creates the Insights & Analysis Agent LLM using qwen-3-235b-a22b-instruct-2507.
    Returns the raw LLM for custom agent logic.
    """
    llm = ChatOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL"),
        temperature=0.1,
        max_tokens=2000,
    )
    return llm


def run_insights_agent(state: AgentState) -> Dict:
    """
    Main function to run the Insights & Analysis Agent with a custom loop.
    """
    if state.cleaned_data is None:
        raise ValueError("Cleaned data is required for analysis.")

    # Create visualizations subdirectory
    viz_dir = os.path.join(state.output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Serialize DataFrame to JSON
    df = state.cleaned_data.copy()
    for col in df.select_dtypes(include=["datetime64"]).columns:
        df[col] = df[col].astype(str)
    df_json = df.to_json(orient="records")

    # Define tools
    tools = {
        "perform_eda_summary": perform_eda_summary,
        "analyze_correlations": analyze_correlations,
        "identify_trends": identify_trends,
        "detect_imbalances": detect_imbalances,
        "run_statistical_tests": run_statistical_tests,
        "create_visualization": create_visualization,
        "generate_insights_summary": generate_insights_summary,
    }
    tool_names = list(tools.keys())

    # Define system prompt
    system_prompt = """You are a Senior Data Analyst for Policy, specializing in extracting actionable insights from data for policymakers.

Your role:
1. Explore the data systematically using the provided tools to understand distributions and relationships.
2. Identify statistically significant patterns, focusing on policy-relevant factors like equity, trends, and correlations.
3. Detect gaps, imbalances, and equity concerns across demographic or geographic dimensions.
4. Validate findings with appropriate statistical tests (e.g., t-tests, ANOVA, chi-square).
5. Visualize key insights using the create_visualization tool for clear communication.
6. Formulate actionable recommendations for policymakers based on findings.

Available tools: {tools}
Tool names: {tool_names}

The input data is provided as a JSON-serialized DataFrame (df_json). Use the tools to process this data and generate structured outputs.
Reason step-by-step, selecting the appropriate tools to complete the task. Use the tools by specifying their names and inputs in your response using the format: [tool_name]({{"arg1": "value1", "arg2": "value2"}}).
Return a JSON object with:
- insights: List of insight dictionaries with title, description, confidence, evidence, implications, recommendations
- analysis_report: Dictionary summarizing analysis results
- visualizations_generated: List of visualization file paths

Example tool usage:
- To perform EDA: [perform_eda_summary]({{"df_json": "..."}})
- To create a visualization: [create_visualization]({{"df_json": "...", "viz_type": "scatter", "x_column": "col1", "y_column": "col2", "output_path": "path.png"}})

Wrap the final output in ```json\n...\n```.
"""

    # Format tools description for the prompt
    tools_description = "\n".join(
        [f"- {name}: {tool.description}" for name, tool in tools.items()]
    )
    prompt_template = system_prompt.format(
        tools=tools_description, tool_names=", ".join(tool_names)
    )

    # Initialize LLM
    llm = create_insights_agent()

    # Custom agent loop
    messages = [
        HumanMessage(
            content=prompt_template
            + f"""\nInput: {json.dumps({
        'df_json': df_json,
        'data_profile': state.data_profile,
        'cleaning_report': state.cleaning_report,
        'output_dir': viz_dir,
        'policy_context': 'general'
    })}"""
        )
    ]
    max_iterations = 10
    analysis_results = {}
    visualizations_generated = []

    for iteration in range(max_iterations):
        try:
            response = llm.invoke(messages)
            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Check for tool calls in the response
            tool_call_pattern = r"\[(.*?)\]\((.*?)\)"
            tool_calls = re.findall(tool_call_pattern, content)

            if tool_calls:
                for tool_name, tool_args in tool_calls:
                    if tool_name not in tools:
                        messages.append(
                            AIMessage(content=f"Error: Tool {tool_name} not found")
                        )
                        continue

                    try:
                        # Parse tool arguments
                        args = json.loads(tool_args)
                        if (
                            "df_json" not in args
                            and tool_name != "generate_insights_summary"
                        ):
                            args["df_json"] = df_json
                        if (
                            "output_path" in args
                            and tool_name == "create_visualization"
                        ):
                            args["output_path"] = os.path.join(
                                viz_dir,
                                args.get("output_path", f"{tool_name}_{iteration}.png"),
                            )

                        # Call the tool
                        result = tools[tool_name](**args)
                        if (
                            tool_name == "create_visualization"
                            and isinstance(result, str)
                            and result.endswith(".png")
                        ):
                            visualizations_generated.append(result)
                        if tool_name != "generate_insights_summary":
                            analysis_results[tool_name] = result

                        messages.append(
                            AIMessage(
                                content=f"Tool {tool_name} result: {json.dumps(result)}"
                            )
                        )
                    except Exception as e:
                        messages.append(
                            AIMessage(content=f"Error in tool {tool_name}: {str(e)}")
                        )
            else:
                # Check for final JSON output
                if content.startswith("```json\n") and content.endswith("\n```"):
                    try:
                        output = json.loads(content[8:-4])
                        state.insights = output.get(
                            "insights", []
                        ) or generate_insights_summary(analysis_results).get(
                            "insights", []
                        )
                        state.analysis_report = output.get(
                            "analysis_report", {"summary": "Analysis completed"}
                        )
                        state.visualizations_generated = output.get(
                            "visualizations_generated", visualizations_generated
                        )
                        state.logs.append(
                            "Insights & Analysis Agent completed successfully."
                        )

                        # Save statistical_analysis.json
                        with open(
                            os.path.join(state.output_dir, "statistical_analysis.json"),
                            "w",
                        ) as f:
                            json.dump(state.analysis_report, f, indent=2)

                        # Generate policy_insights_report.md
                        report_content = generate_policy_report(state)
                        with open(
                            os.path.join(state.output_dir, "policy_insights_report.md"),
                            "w",
                        ) as f:
                            f.write(report_content)

                        return {"next": "completed", "state": state}
                    except json.JSONDecodeError as e:
                        messages.append(
                            AIMessage(
                                content=f"Error: Invalid JSON output: {str(e)}. Please provide valid JSON wrapped in ```json\n...\n```."
                            )
                        )
                else:
                    messages.append(
                        AIMessage(
                            content="Please provide the final output in JSON format wrapped in ```json\n...\n```."
                        )
                    )

        except Exception as e:
            state.logs.append(f"Insights Agent failed: {str(e)}")
            return {"error": str(e), "state": state}

    state.logs.append(
        "Insights Agent failed: Max iterations reached without valid JSON output"
    )
    return {"error": "Max iterations reached without valid JSON output", "state": state}


def generate_policy_report(state) -> str:
    """
    Generates the markdown policy insights report.
    """
    md = "# Policy Insights Report\n\n"

    # Executive Summary
    md += "## Executive Summary\n"
    if state.insights:
        for insight in state.insights[:5]:
            md += f"- {insight['title']}: {insight['description']}\n"
    else:
        md += "- No insights generated.\n"
    md += "\n"

    # Key Insights
    md += "## Key Insights\n"
    for insight in state.insights or []:
        md += f"### {insight['title']}\n"
        md += f"{insight['description']}\n\n"
        if "evidence" in insight:
            md += f"**Evidence:** {json.dumps(insight['evidence'])}\n\n"
        md += f"**Implications:** {insight['implications']}\n\n"
        if "recommendations" in insight:
            md += "**Recommendations:**\n"
            # Ensure recommendations is a list; convert string to list if necessary
            recommendations = insight["recommendations"]
            if isinstance(recommendations, str):
                recommendations = [recommendations]
            elif not isinstance(recommendations, list):
                recommendations = []
            for rec in recommendations:
                md += f"- {rec}\n"
        md += "\n"

    # Data Gaps
    md += "## Data Gaps\n"
    gaps = (
        state.cleaning_report.get("data_gaps", "No major gaps identified.")
        if state.cleaning_report
        else "No major gaps identified."
    )
    md += f"{gaps}\n\n"

    # Recommendations
    md += "## Recommendations\n"
    if state.insights:
        recommendations = set()
        for insight in state.insights:
            # Ensure recommendations is a list; convert string to list if necessary
            recs = insight.get("recommendations", [])
            if isinstance(recs, str):
                recs = [recs]
            elif not isinstance(recs, list):
                recs = []
            recommendations.update(recs)
        if recommendations:
            for rec in sorted(recommendations):  # Sort for consistent output
                md += f"- {rec}\n"
        else:
            md += "No specific recommendations generated.\n"
    else:
        md += "No specific recommendations generated.\n"

    return md