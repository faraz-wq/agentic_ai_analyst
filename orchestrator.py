"""
orchestrator.py - Example usage of the ingestion agent
"""
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Example orchestration logic showing how to use the ingestion agent.
    """
    from core.state import AgentState
    from agents.ingestion_agent import run_ingestion_agent
    
    # Create initial state
    initial_state = AgentState(
        input_file_path="data/iris.csv",
        output_dir="output/"
    )
    
    # Run ingestion agent
    updated_state = run_ingestion_agent(initial_state)
    
    # Check results
    if updated_state.get('error'):
        print(f"Error occurred: {updated_state['error']}")
    else:
        print("Ingestion completed successfully!")
        print(f"Data shape: {updated_state['data_profile']['dataset_info']['shape'] if updated_state.get('data_profile') else 'Unknown'}")
        print(f"Columns: {list(updated_state['data_profile']['columns'].keys()) if updated_state.get('data_profile') else 'Unknown'}")
    
    # Print logs
    for log in updated_state['logs']:
        print(f"LOG: {log}")


if __name__ == "__main__":
    main()
