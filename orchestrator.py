# orchestrator.py
# Orchestrates the full data analytics pipeline: ingestion, cleaning, and insights generation.

from dotenv import load_dotenv
load_dotenv()

def main():
    """
    Orchestrates the full data analytics pipeline:
    1. Data Ingestion
    2. Data Cleaning
    3. Insights & Analysis
    """
    from core.state import AgentState
    from agents.ingestion_agent import run_ingestion_agent
    from agents.cleaning_agent import run_cleaning_agent
    from agents.insights_agent import run_insights_agent
    
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
    state_after_cleaning = run_cleaning_agent(state_obj)
    
    # Check cleaning results
    if state_after_cleaning.get('error'):
        print(f"Cleaning failed: {state_after_cleaning['error']}")
        return
    
    if not state_after_cleaning.get('cleaning_complete'):
        print("Cleaning did not complete successfully")
        return
    
    print("✅ Cleaning completed successfully!")
    if state_after_cleaning.get('cleaning_report'):
        report = state_after_cleaning['cleaning_report']
        summary = report.get('summary', {})
        
        print(f"Issues detected: {summary.get('issues_detected', 0)}")
        print(f"Actions performed: {summary.get('actions_performed', 0)}")
        print(f"Original shape: {summary.get('original_shape', [0, 0])}")
        print(f"Cleaned shape: {summary.get('cleaned_shape', [0, 0])}")
    
    print("\n=== STEP 3: INSIGHTS & ANALYSIS ===")
    
    # Convert back to AgentState object for insights
    state_obj = AgentState(**state_after_cleaning)
    
    # Run insights agent
    final_state = run_insights_agent(state_obj)
    
    # Check insights results
    if final_state.get('error'):
        print(f"Insights generation failed: {final_state['error']}")
        return
    
    if final_state.get('next') != 'completed':
        print("Insights generation did not complete successfully")
        return
    
    print("✅ Insights & Analysis completed successfully!")
    
    # Display insights results
    if final_state.get('insights'):
        print("\nKey Insights Generated:")
        for idx, insight in enumerate(final_state['insights'], 1):
            print(f"{idx}. {insight['title']}: {insight['description']}")
            print(f"   Confidence: {insight.get('confidence', 'N/A')}")
            print(f"   Implications: {insight.get('implications', 'N/A')}")
            if insight.get('recommendations'):
                print("   Recommendations:")
                for rec in insight['recommendations']:
                    print(f"     - {rec}")
            print()
    
    if final_state.get('analysis_report'):
        print("Analysis Report Summary:")
        print(f"Summary: {final_state['analysis_report'].get('summary', 'No summary available')}")
    
    if final_state.get('visualizations_generated'):
        print("\nVisualizations Generated:")
        for viz_path in final_state['visualizations_generated']:
            print(f"- {viz_path}")
    
    # Print all logs
    print("\n=== DETAILED LOGS ===")
    for log in final_state.get('logs', []):
        print(f"LOG: {log}")
    
    print(f"\n✅ Pipeline completed! Check outputs in: {state_obj.output_dir}")
    print("Files generated:")
    print("- raw_data.csv (from ingestion)")
    print("- profile.html (from ingestion)")  
    print("- cleaned_data.csv (from cleaning)")
    print("- cleaning_report.md (from cleaning)")
    print("- statistical_analysis.json (from insights)")
    print("- policy_insights_report.md (from insights)")
    print("- visualizations/* (from insights)")

if __name__ == "__main__":
    main()