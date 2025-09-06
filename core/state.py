"""
Shared state management for the multi-agent data analytics system.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import pandas as pd

class AgentState(BaseModel):
    """
    Shared state object that gets passed between agents in the orchestration workflow.
    
    This state object maintains all the information needed across the entire
    data analytics pipeline, from ingestion through final documentation.
    """
    
    # Input parameters
    input_file_path: str = Field(..., description="Path to the input data file")
    output_dir: str = Field(..., description="Directory to save all outputs")
    
    # Data storage
    raw_data: Optional[pd.DataFrame] = Field(default=None, description="The loaded and standardized DataFrame")
    cleaned_data: Optional[pd.DataFrame] = Field(default=None, description="The cleaned and transformed DataFrame")
    data_profile: Optional[Dict[str, Any]] = Field(default=None, description="Structured summary of data profile")
    
    # Cleaning and transformation results
    cleaning_report: Optional[Dict[str, Any]] = Field(default=None, description="Structured report of all cleaning actions")
    issues_detected: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of identified data quality problems")
    actions_performed: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of cleaning actions taken")
    
    # Workflow tracking
    logs: List[str] = Field(default_factory=list, description="Human-readable log messages")
    error: Optional[str] = Field(default=None, description="Error message if any step fails")
    
    # Agent status tracking
    ingestion_complete: bool = Field(default=False, description="Whether ingestion agent completed successfully")
    cleaning_complete: bool = Field(default=False, description="Whether cleaning agent completed successfully")
    analysis_complete: bool = Field(default=False, description="Whether analysis agent completed successfully")
    documentation_complete: bool = Field(default=False, description="Whether documentation agent completed successfully")
    
    class Config:
        arbitrary_types_allowed = True  # Allows pandas DataFrame in Pydantic model
