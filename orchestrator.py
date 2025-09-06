"""
orchestrator.py - Example usage of the ingestion agent
"""
from dotenv import load_dotenv

load_dotenv()

def main():
    """
    Example orchestration logic showing how to use both ingestion and cleaning agents.
    """
    from core.state import AgentState
    from agents.ingestion_agent import run_ingestion_agent
    from agents.cleaning_agent import run_cleaning_agent
    
    # Create initial state
    initial_state = AgentState(
        input_file_path="data/iris.csv",
        output_dir="./output/"
    )
    
    print("=== STEP 1: DATA INGESTION ===")
    
    # Run ingestion agent
    state_after_ingestion = run_ingestion_agent(initial_state)
    
    # Check ingestion results
    if state_after_ingestion.get('error'):
        print(f"Ingestion failed: {state_after_ingestion['error']}")
        return
    
    if not state_after_ingestion.get('ingestion_complete'):
        print("Ingestion did not complete successfully")
        return
    
    print("✅ Ingestion completed successfully!")
    print(f"Data shape: {state_after_ingestion['raw_data'].shape if state_after_ingestion.get('raw_data') is not None else 'Unknown'}")
    
    print("\n=== STEP 2: DATA CLEANING ===")
    
    # Convert back to AgentState object for cleaning
    state_obj = AgentState(**state_after_ingestion)
    
    # Run cleaning agent
    final_state = run_cleaning_agent(state_obj)
    
    # Check cleaning results
    if final_state.get('error'):
        print(f"Cleaning failed: {final_state['error']}")
        return
    
    if not final_state.get('cleaning_complete'):
        print("Cleaning did not complete successfully")
        return
    
    print("✅ Cleaning completed successfully!")
    
    # Display results
    if final_state.get('cleaning_report'):
        report = final_state['cleaning_report']
        summary = report.get('summary', {})
        
        print(f"Issues detected: {summary.get('issues_detected', 0)}")
        print(f"Actions performed: {summary.get('actions_performed', 0)}")
        print(f"Original shape: {summary.get('original_shape', [0, 0])}")
        print(f"Cleaned shape: {summary.get('cleaned_shape', [0, 0])}")
    
    # Print all logs
    print("\n=== DETAILED LOGS ===")
    for log in final_state.get('logs', []):
        print(f"LOG: {log}")
    
    print(f"\n✅ Pipeline completed! Check outputs in: ./output/")
    print("Files generated:")
    print("- raw_data.csv (from ingestion)")
    print("- profile.html (from ingestion)")  
    print("- cleaned_data.csv (from cleaning)")
    print("- cleaning_report.md (from cleaning)")


if __name__ == "__main__":
    main()
