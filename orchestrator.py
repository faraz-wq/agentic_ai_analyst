# orchestrator.py
# CLI tool for orchestrating the full data analytics pipeline: ingestion, cleaning, and insights generation.

import click
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@click.command()
@click.argument('csv_file', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('--output-dir', '-o', 
              type=click.Path(path_type=Path), 
              default='./output/',
              help='Output directory for generated files (default: ./output/)')
@click.option('--verbose', '-v', 
              is_flag=True, 
              help='Enable verbose logging')
@click.option('--skip-ingestion', 
              is_flag=True, 
              help='Skip data ingestion step (useful if data is already processed)')
@click.option('--skip-cleaning', 
              is_flag=True, 
              help='Skip data cleaning step')
@click.option('--skip-insights', 
              is_flag=True, 
              help='Skip insights generation step')
def main(csv_file, output_dir, verbose, skip_ingestion, skip_cleaning, skip_insights):
    """
    Data Analytics Pipeline CLI Tool
    
    Orchestrates the full data analytics pipeline on a CSV file:
    1. Data Ingestion
    2. Data Cleaning  
    3. Insights & Analysis
    
    CSV_FILE: Path to the input CSV file to process
    """
    
    # Validate input file
    if not csv_file.suffix.lower() == '.csv':
        click.echo(click.style("Error: Input file must be a CSV file", fg='red'), err=True)
        raise click.Abort()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"🚀 Starting data analytics pipeline...")
    click.echo(f"📁 Input file: {csv_file}")
    click.echo(f"📂 Output directory: {output_dir}")
    
    try:
        from core.state import AgentState
        from agents.ingestion_agent import run_ingestion_agent
        from agents.cleaning_agent import run_cleaning_agent
        from agents.insights_agent import run_insights_agent
    except ImportError as e:
        click.echo(click.style(f"Error importing modules: {e}", fg='red'), err=True)
        click.echo("Make sure all required modules are available in your Python path.")
        raise click.Abort()
    
    # Create initial state
    initial_state = AgentState(
        input_file_path=str(csv_file),
        output_dir=str(output_dir)
    )
    
    current_state = initial_state
    
    # Step 1: Data Ingestion
    if not skip_ingestion:
        click.echo("\n" + "="*50)
        click.echo("📥 STEP 1: DATA INGESTION")
        click.echo("="*50)
        
        with click.progressbar(length=1, label='Processing ingestion') as bar:
            state_after_ingestion = run_ingestion_agent(current_state)
            bar.update(1)
        
        # Check ingestion results
        if state_after_ingestion.get('error'):
            click.echo(click.style(f"❌ Ingestion failed: {state_after_ingestion['error']}", fg='red'), err=True)
            raise click.Abort()
        
        if not state_after_ingestion.get('ingestion_complete'):
            click.echo(click.style("❌ Ingestion did not complete successfully", fg='red'), err=True)
            raise click.Abort()
        
        click.echo(click.style("✅ Ingestion completed successfully!", fg='green'))
        
        if verbose and state_after_ingestion.get('raw_data') is not None:
            data_shape = state_after_ingestion['raw_data'].shape
            click.echo(f"📊 Data shape: {data_shape[0]:,} rows × {data_shape[1]} columns")
        
        current_state = state_after_ingestion
    else:
        click.echo(click.style("⏭️  Skipping ingestion step", fg='yellow'))
    
    # Step 2: Data Cleaning
    if not skip_cleaning:
        click.echo("\n" + "="*50)
        click.echo("🧹 STEP 2: DATA CLEANING")
        click.echo("="*50)
        
        # Convert back to AgentState object for cleaning
        state_obj = AgentState(**current_state) if isinstance(current_state, dict) else current_state
        
        with click.progressbar(length=1, label='Processing cleaning') as bar:
            state_after_cleaning = run_cleaning_agent(state_obj)
            bar.update(1)
        
        # Check cleaning results
        if state_after_cleaning.get('error'):
            click.echo(click.style(f"❌ Cleaning failed: {state_after_cleaning['error']}", fg='red'), err=True)
            raise click.Abort()
        
        if not state_after_cleaning.get('cleaning_complete'):
            click.echo(click.style("❌ Cleaning did not complete successfully", fg='red'), err=True)
            raise click.Abort()
        
        click.echo(click.style("✅ Cleaning completed successfully!", fg='green'))
        
        if verbose and state_after_cleaning.get('cleaning_report'):
            report = state_after_cleaning['cleaning_report']
            summary = report.get('summary', {})
            
            click.echo(f"🔍 Issues detected: {summary.get('issues_detected', 0)}")
            click.echo(f"⚡ Actions performed: {summary.get('actions_performed', 0)}")
            click.echo(f"📏 Original shape: {summary.get('original_shape', [0, 0])}")
            click.echo(f"📐 Cleaned shape: {summary.get('cleaned_shape', [0, 0])}")
        
        current_state = state_after_cleaning
    else:
        click.echo(click.style("⏭️  Skipping cleaning step", fg='yellow'))
    
    # Step 3: Insights & Analysis
    if not skip_insights:
        click.echo("\n" + "="*50)
        click.echo("💡 STEP 3: INSIGHTS & ANALYSIS")
        click.echo("="*50)
        
        # Convert back to AgentState object for insights
        state_obj = AgentState(**current_state) if isinstance(current_state, dict) else current_state
        
        with click.progressbar(length=1, label='Generating insights') as bar:
            final_state = run_insights_agent(state_obj)
            bar.update(1)
        
        # Check insights results
        if final_state.get('error'):
            click.echo(click.style(f"❌ Insights generation failed: {final_state['error']}", fg='red'), err=True)
            raise click.Abort()
        
        if final_state.get('next') != 'completed':
            click.echo(click.style("❌ Insights generation did not complete successfully", fg='red'), err=True)
            raise click.Abort()
        
        click.echo(click.style("✅ Insights & Analysis completed successfully!", fg='green'))
        
        # Display insights results
        if final_state.get('insights'):
            click.echo(click.style("\n🎯 Key Insights Generated:", fg='cyan', bold=True))
            for idx, insight in enumerate(final_state['insights'], 1):
                click.echo(f"{idx}. {click.style(insight['title'], fg='yellow', bold=True)}: {insight['description']}")
                if verbose:
                    click.echo(f"   📈 Confidence: {insight.get('confidence', 'N/A')}")
                    click.echo(f"   💭 Implications: {insight.get('implications', 'N/A')}")
                    if insight.get('recommendations'):
                        click.echo("   📋 Recommendations:")
                        for rec in insight['recommendations']:
                            click.echo(f"     • {rec}")
                click.echo()
        
        if verbose and final_state.get('analysis_report'):
            click.echo(click.style("📊 Analysis Report Summary:", fg='cyan', bold=True))
            click.echo(f"{final_state['analysis_report'].get('summary', 'No summary available')}")
        
        if final_state.get('visualizations_generated'):
            click.echo(click.style("\n📈 Visualizations Generated:", fg='cyan', bold=True))
            for viz_path in final_state['visualizations_generated']:
                click.echo(f"• {click.style(viz_path, fg='blue')}")
        
        current_state = final_state
    else:
        click.echo(click.style("⏭️  Skipping insights step", fg='yellow'))
    
    # Print detailed logs if verbose
    if verbose and current_state.get('logs'):
        click.echo(click.style("\n📋 DETAILED LOGS", fg='magenta', bold=True))
        click.echo("="*50)
        for log in current_state.get('logs', []):
            click.echo(f"📝 {log}")
    
    # Final summary
    click.echo("\n" + "="*50)
    click.echo(click.style("🎉 PIPELINE COMPLETED SUCCESSFULLY!", fg='green', bold=True))
    click.echo("="*50)
    click.echo(f"📂 Output location: {click.style(str(output_dir), fg='blue', bold=True)}")
    
    click.echo(click.style("\n📄 Files generated:", fg='cyan'))
    generated_files = [
        "raw_data.csv (from ingestion)",
        "profile.html (from ingestion)",
        "cleaned_data.csv (from cleaning)",
        "cleaning_report.md (from cleaning)",
        "statistical_analysis.json (from insights)",
        "policy_insights_report.md (from insights)",
        "visualizations/* (from insights)"
    ]
    
    for file in generated_files:
        click.echo(f"• {file}")


@click.group()
def cli():
    """Data Analytics Pipeline CLI Tool"""
    pass


@cli.command('process')
@click.argument('csv_file', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('--output-dir', '-o', 
              type=click.Path(path_type=Path), 
              default='./output/',
              help='Output directory for generated files (default: ./output/)')
@click.option('--verbose', '-v', 
              is_flag=True, 
              help='Enable verbose logging')
@click.option('--skip-ingestion', 
              is_flag=True, 
              help='Skip data ingestion step (useful if data is already processed)')
@click.option('--skip-cleaning', 
              is_flag=True, 
              help='Skip data cleaning step')
@click.option('--skip-insights', 
              is_flag=True, 
              help='Skip insights generation step')
def process_command(csv_file, output_dir, verbose, skip_ingestion, skip_cleaning, skip_insights):
    """Process a CSV file through the analytics pipeline (alias for main command)"""
    ctx = click.get_current_context()
    ctx.invoke(main, 
              csv_file=csv_file, 
              output_dir=output_dir, 
              verbose=verbose,
              skip_ingestion=skip_ingestion,
              skip_cleaning=skip_cleaning,
              skip_insights=skip_insights)


@cli.command('version')
def version():
    """Show the version of the CLI tool"""
    click.echo("Data Analytics Pipeline CLI v1.0.0")


if __name__ == "__main__":
    # Support both direct execution and CLI group
    import sys
    if len(sys.argv) > 1 and sys.argv[1] not in ['process', 'version']:
        # Direct execution with CSV file
        main()
    else:
        # CLI group execution
        cli()