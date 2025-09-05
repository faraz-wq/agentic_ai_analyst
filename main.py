"""
Agentic AI Data Analytics Tool CLI

A comprehensive command-line interface for AI-powered data analytics
with ingestion, cleaning, insights generation, and documentation capabilities.
"""

import click
import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_analytics_tool.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ai_data_analytics_tool")

class DataAnalyticsTool:
    """Main class for the AI Data Analytics Tool"""
    
    def __init__(self):
        self.data = None
        self.metadata = {
            "ingestion_time": None,
            "source_file": None,
            "cleaning_operations": [],
            "transformations": [],
            "insights_generated": []
        }
    
    def ingest_data(self, file_path, output_dir):
        """Ingest and standardize data from various formats"""
        logger.info(f"Ingesting data from: {file_path}")
        
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Standardize column names
            self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Record metadata
            self.metadata["ingestion_time"] = datetime.now().isoformat()
            self.metadata["source_file"] = file_path
            self.metadata["original_shape"] = self.data.shape
            self.metadata["original_columns"] = list(self.data.columns)
            
            # Save analysis-ready data
            output_file = os.path.join(output_dir, "analysis_ready_data.csv")
            self.data.to_csv(output_file, index=False)
            
            logger.info(f"Data ingested successfully. Shape: {self.data.shape}")
            logger.info(f"Analysis-ready data saved to: {output_file}")
            
            return True, f"Data ingested successfully. Shape: {self.data.shape}"
            
        except Exception as e:
            logger.error(f"Error during data ingestion: {str(e)}")
            return False, f"Error during data ingestion: {str(e)}"
    
    def clean_and_transform(self, output_dir):
        """Clean and transform the data"""
        logger.info("Starting data cleaning and transformation")
        
        if self.data is None:
            return False, "No data loaded. Please ingest data first."
        
        cleaning_log = []
        transformations_log = []
        
        try:
            # Identify and handle missing values
            missing_summary = self.data.isnull().sum()
            cleaning_log.append(f"Missing values identified: {dict(missing_summary)}")
            
            # For simplicity, we'll fill numeric columns with median and categorical with mode
            for column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    if self.data[column].isnull().sum() > 0:
                        fill_value = self.data[column].median()
                        self.data[column].fillna(fill_value, inplace=True)
                        cleaning_log.append(f"Filled missing values in {column} with median: {fill_value}")
                else:
                    if self.data[column].isnull().sum() > 0:
                        fill_value = self.data[column].mode()[0] if len(self.data[column].mode()) > 0 else "Unknown"
                        self.data[column].fillna(fill_value, inplace=True)
                        cleaning_log.append(f"Filled missing values in {column} with mode: {fill_value}")
            
            # Remove duplicates
            initial_count = len(self.data)
            self.data.drop_duplicates(inplace=True)
            duplicates_removed = initial_count - len(self.data)
            if duplicates_removed > 0:
                cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
            
            # Standardize data types
            for column in self.data.select_dtypes(include=['object']).columns:
                self.data[column] = self.data[column].astype(str).str.strip()
                transformations_log.append(f"Standardized text in column: {column}")
            
            # Save cleaned data
            output_file = os.path.join(output_dir, "cleaned_transformed_data.csv")
            self.data.to_csv(output_file, index=False)
            
            # Update metadata
            self.metadata["cleaning_operations"] = cleaning_log
            self.metadata["transformations"] = transformations_log
            self.metadata["final_shape"] = self.data.shape
            
            logger.info("Data cleaning and transformation completed successfully")
            return True, "Data cleaning and transformation completed successfully"
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            return False, f"Error during data cleaning: {str(e)}"
    
    def generate_insights(self, output_dir):
        """Generate insights from the data"""
        logger.info("Generating insights from data")
        
        if self.data is None:
            return False, "No data loaded. Please ingest data first."
        
        insights = []
        
        try:
            # Basic statistics
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats = self.data[numeric_columns].describe()
                insights.append("Basic Statistics:")
                insights.append(tabulate(stats, headers='keys', tablefmt='grid'))
            
            # Identify trends and patterns
            for column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    # Check for trends if there's a time-related column
                    time_cols = [col for col in self.data.columns if 'date' in col or 'time' in col]
                    if time_cols:
                        for time_col in time_cols:
                            try:
                                # Simple trend analysis
                                self.data[time_col] = pd.to_datetime(self.data[time_col])
                                sorted_data = self.data.sort_values(time_col)
                                correlation = sorted_data[column].corr(pd.Series(range(len(sorted_data))))
                                
                                if abs(correlation) > 0.5:
                                    trend = "increasing" if correlation > 0 else "decreasing"
                                    insights.append(f"Strong {trend} trend in {column} over {time_col} (correlation: {correlation:.2f})")
                            except:
                                pass
            
            # Identify gaps and imbalances
            for column in self.data.select_dtypes(exclude=[np.number]).columns:
                if self.data[column].nunique() < 20:  # Avoid columns with too many unique values
                    value_counts = self.data[column].value_counts()
                    if len(value_counts) > 0:
                        imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 else 1
                        if imbalance_ratio > 10:
                            insights.append(f"Significant imbalance in {column}: {value_counts.iloc[0]} vs {value_counts.iloc[-1]} (ratio: {imbalance_ratio:.1f})")
            
            # Save insights to file
            insights_file = os.path.join(output_dir, "data_insights.txt")
            with open(insights_file, 'w') as f:
                f.write("DATA INSIGHTS REPORT\n")
                f.write("====================\n\n")
                for insight in insights:
                    f.write(insight + "\n\n")
            
            # Generate simple visualizations
            self._generate_visualizations(output_dir)
            
            # Update metadata
            self.metadata["insights_generated"] = insights
            self.metadata["insights_time"] = datetime.now().isoformat()
            
            logger.info("Insights generation completed successfully")
            return True, insights
            
        except Exception as e:
            logger.error(f"Error during insights generation: {str(e)}")
            return False, f"Error during insights generation: {str(e)}"
    
    def _generate_visualizations(self, output_dir):
        """Generate basic visualizations"""
        try:
            # Create visualizations directory
            viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Histograms for numeric columns
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            for column in numeric_columns[:5]:  # Limit to first 5 numeric columns
                plt.figure(figsize=(10, 6))
                self.data[column].hist(bins=30)
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(viz_dir, f'{column}_distribution.png'))
                plt.close()
            
            # Bar charts for categorical columns with limited unique values
            categorical_columns = self.data.select_dtypes(exclude=[np.number]).columns
            for column in categorical_columns:
                if self.data[column].nunique() <= 10:
                    plt.figure(figsize=(10, 6))
                    self.data[column].value_counts().plot(kind='bar')
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f'{column}_distribution.png'))
                    plt.close()
            
            logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    def generate_documentation(self, output_dir):
        """Generate comprehensive documentation"""
        logger.info("Generating documentation")
        
        try:
            # Create documentation file
            doc_file = os.path.join(output_dir, "process_documentation.md")
            
            with open(doc_file, 'w') as f:
                f.write("# Data Analytics Process Documentation\n\n")
                f.write("## Overview\n")
                f.write(f"- Process completed: {datetime.now().isoformat()}\n")
                f.write(f"- Source file: {self.metadata.get('source_file', 'Unknown')}\n")
                f.write(f"- Original shape: {self.metadata.get('original_shape', 'Unknown')}\n")
                f.write(f"- Final shape: {self.metadata.get('final_shape', 'Unknown')}\n\n")
                
                f.write("## Data Ingestion & Standardization\n")
                f.write(f"- Ingestion time: {self.metadata.get('ingestion_time', 'Unknown')}\n")
                f.write(f"- Original columns: {', '.join(self.metadata.get('original_columns', []))}\n\n")
                
                f.write("## Data Cleaning Operations\n")
                if self.metadata.get('cleaning_operations'):
                    for operation in self.metadata['cleaning_operations']:
                        f.write(f"- {operation}\n")
                else:
                    f.write("- No cleaning operations recorded\n")
                f.write("\n")
                
                f.write("## Data Transformations\n")
                if self.metadata.get('transformations'):
                    for transformation in self.metadata['transformations']:
                        f.write(f"- {transformation}\n")
                else:
                    f.write("- No transformations recorded\n")
                f.write("\n")
                
                f.write("## Insights Generated\n")
                if self.metadata.get('insights_generated'):
                    for insight in self.metadata['insights_generated']:
                        f.write(f"{insight}\n\n")
                else:
                    f.write("- No insights recorded\n")
            
            logger.info(f"Documentation saved to: {doc_file}")
            return True, f"Documentation saved to: {doc_file}"
            
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            return False, f"Error generating documentation: {str(e)}"

# CLI Setup using Click
@click.group()
def cli():
    """Agentic AI Data Analytics Tool CLI"""
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory for results')
def ingest(input_file, output_dir):
    """Ingest and standardize data from various formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    tool = DataAnalyticsTool()
    success, message = tool.ingest_data(input_file, output_dir)
    
    if success:
        click.echo(click.style("✓ " + message, fg='green'))
    else:
        click.echo(click.style("✗ " + message, fg='red'))

@cli.command()
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory for results')
def clean(output_dir):
    """Clean and transform the data"""
    os.makedirs(output_dir, exist_ok=True)
    
    tool = DataAnalyticsTool()
    # For simplicity, we're creating a new instance. In a real implementation,
    # you would persist the tool instance or load data from the output directory
    
    click.echo("This command requires data to be ingested first.")
    click.echo("Please run the ingest command or ensure data is available.")

@cli.command()
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory for results')
def insights(output_dir):
    """Generate insights from the data"""
    os.makedirs(output_dir, exist_ok=True)
    
    tool = DataAnalyticsTool()
    success, message = tool.generate_insights(output_dir)
    
    if success:
        click.echo(click.style("✓ Insights generated successfully", fg='green'))
        if isinstance(message, list):
            for insight in message:
                click.echo(insight)
        else:
            click.echo(message)
    else:
        click.echo(click.style("✗ " + message, fg='red'))

@cli.command()
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory for results')
def docs(output_dir):
    """Generate comprehensive documentation"""
    os.makedirs(output_dir, exist_ok=True)
    
    tool = DataAnalyticsTool()
    success, message = tool.generate_documentation(output_dir)
    
    if success:
        click.echo(click.style("✓ " + message, fg='green'))
    else:
        click.echo(click.style("✗ " + message, fg='red'))

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', 
              help='Output directory for results')
def full_analysis(input_file, output_dir):
    """Run the complete analysis pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    tool = DataAnalyticsTool()
    
    # Step 1: Ingest data
    click.echo(click.style("Step 1: Ingesting and standardizing data...", fg='blue'))
    success, message = tool.ingest_data(input_file, output_dir)
    if not success:
        click.echo(click.style("✗ " + message, fg='red'))
        return
    click.echo(click.style("✓ " + message, fg='green'))
    
    # Step 2: Clean and transform
    click.echo(click.style("Step 2: Cleaning and transforming data...", fg='blue'))
    success, message = tool.clean_and_transform(output_dir)
    if not success:
        click.echo(click.style("✗ " + message, fg='red'))
        return
    click.echo(click.style("✓ " + message, fg='green'))
    
    # Step 3: Generate insights
    click.echo(click.style("Step 3: Generating insights...", fg='blue'))
    success, message = tool.generate_insights(output_dir)
    if not success:
        click.echo(click.style("✗ " + message, fg='red'))
        return
    
    click.echo(click.style("✓ Insights generated successfully", fg='green'))
    if isinstance(message, list):
        for insight in message[:5]:  # Show first 5 insights
            click.echo(insight)
        if len(message) > 5:
            click.echo(f"... and {len(message) - 5} more insights")
    
    # Step 4: Generate documentation
    click.echo(click.style("Step 4: Generating documentation...", fg='blue'))
    success, message = tool.generate_documentation(output_dir)
    if not success:
        click.echo(click.style("✗ " + message, fg='red'))
        return
    click.echo(click.style("✓ " + message, fg='green'))
    
    click.echo(click.style("\nFull analysis completed successfully!", fg='green', bold=True))
    click.echo(f"Results saved to: {output_dir}")

if __name__ == '__main__':
    cli()