"""
Ingestion & Profiling Agent for the multi-agent data analytics system.
"""
import os
import json
from typing import Dict, Any
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from core.state import AgentState
from core.tools import read_dataframe, standardize_dataframe, generate_profile, save_standardized_data
from openai import OpenAI

def create_ingestion_agent() -> AgentExecutor:
    """
    Create and configure the Ingestion & Profiling Agent.
    
    This agent is responsible for:
    1. Loading data from various file formats
    2. Standardizing the dataset (column names, data types)
    3. Generating comprehensive data profiles
    4. Saving processed data and reports
    
    Returns:
        AgentExecutor: Configured agent ready for execution
    """
    
    # Initialize the LLM
    llm = ChatOpenAI(
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL"),
        temperature=0.1,
        max_tokens=2000,
    )
    
    # Define the system prompt
    system_prompt = """
    You are the Ingestion & Profiling Agent, a specialized component of a multi-agent data analytics system.

    Your primary responsibilities are:
    1. Load data from the specified file path using the read_dataframe tool
    2. Standardize the dataset using the standardize_dataframe tool  
    3. Generate a comprehensive data profile using the generate_profile tool
    4. Save the standardized data using the save_standardized_data tool

    CRITICAL WORKFLOW:
    You MUST follow this exact sequence:
    1. First, use read_dataframe with the provided file path
    2. Then, use standardize_dataframe to clean column names
    3. Next, use generate_profile to create the profile report (save as profile.html in output directory)
    4. Finally, use save_standardized_data to save the clean data (save as raw_data.csv in output directory)

    ERROR HANDLING:
    - If any step fails, document the error clearly and stop the workflow
    - Always provide informative messages about what went wrong
    - Never proceed if a previous step failed

    RESPONSE FORMAT:
    Always provide a clear summary of:
    - What actions were taken
    - Any errors encountered  
    - Key statistics about the loaded data
    - File paths where outputs were saved

    Remember: You are part of a larger system. Your outputs will be used by other agents, so accuracy and consistency are paramount.
    """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # Define available tools
    tools = [
        read_dataframe,
        standardize_dataframe, 
        generate_profile,
        save_standardized_data
    ]
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create and return the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True
    )
    
    return agent_executor

def run_ingestion_agent(state: AgentState) -> Dict[str, Any]:
    """
    Execute the Ingestion & Profiling Agent workflow.
    
    Args:
        state: The current AgentState containing input parameters
        
    Returns:
        Dict containing the updated state information
    """
    
    # Convert state to dict for easier manipulation
    updated_state = state.dict()
    updated_state['logs'] = updated_state.get('logs', [])
    
    try:
        # Validate inputs
        if not state.input_file_path:
            raise ValueError("input_file_path is required but not provided")
        if not state.output_dir:
            raise ValueError("output_dir is required but not provided")
        
        # Ensure output directory exists
        os.makedirs(state.output_dir, exist_ok=True)
        
        # Create agent
        agent_executor = create_ingestion_agent()
        
        # Prepare agent input
        agent_input = {
            "input": f"""
            Please process the following data file for ingestion and profiling:
            
            Input file: {state.input_file_path}
            Output directory: {state.output_dir}
            
            Follow the complete workflow:
            1. Load the data from the input file
            2. Standardize the dataset (column names to snake_case)
            3. Generate a profile report and save as 'profile.html' in the output directory
            4. Save the standardized data as 'raw_data.csv' in the output directory
            
            Provide a comprehensive summary of the results.
            """
        }
        
        # Execute the agent
        result = agent_executor.invoke(agent_input)
        
        # Extract the agent's response
        agent_response = result.get('output', '')
        updated_state['logs'].append(f"Ingestion Agent Response: {agent_response}")
        
        # Extract profile summary from intermediate steps
        profile_summary = None
        for step in result.get('intermediate_steps', []):
            action, output = step
            updated_state['logs'].append(f"Tool {action.tool}: {output}")
            if action.tool == 'generate_profile':
                try:
                    profile_data = json.loads(output).get('summary')
                    if profile_data:
                        profile_summary = profile_data
                        updated_state['data_profile'] = profile_summary
                        updated_state['logs'].append("Successfully extracted profile summary")
                except json.JSONDecodeError:
                    updated_state['logs'].append(f"Warning: Could not parse profile summary from {output}")
        
        # Since _loaded_dataframe is not reliably accessible via globals(),
        # reload the standardized DataFrame from the saved raw_data.csv
        raw_data_path = os.path.join(state.output_dir, 'raw_data.csv')
        if os.path.exists(raw_data_path):
            import pandas as pd
            updated_state['raw_data'] = pd.read_csv(raw_data_path)
            updated_state['logs'].append(f"Loaded standardized DataFrame from {raw_data_path} with shape {updated_state['raw_data'].shape}")
            # Debug: Print DataFrame head
            #print("DataFrame head:\n", updated_state['raw_data'].head())
        else:
            updated_state['logs'].append(f"Warning: Standardized data file {raw_data_path} not found")
        
        # Mark ingestion as complete if profile and data are available
        updated_state['ingestion_complete'] = bool(updated_state.get('raw_data') is not None and profile_summary is not None)
        if updated_state['ingestion_complete']:
            updated_state['logs'].append("Ingestion agent completed successfully")
        else:
            updated_state['logs'].append("Ingestion agent partially completed due to missing data or profile")
        
        # Clean up global variables (if they exist)
        if '_loaded_dataframe' in globals():
            del globals()['_loaded_dataframe']
        if '_profile_summary' in globals():
            del globals()['_profile_summary']
            
    except Exception as e:
        error_msg = f"Ingestion agent failed: {str(e)}"
        updated_state['error'] = error_msg
        updated_state['logs'].append(error_msg)
        updated_state['ingestion_complete'] = False
    
    return updated_state