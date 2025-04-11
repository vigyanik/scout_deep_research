# models/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Models from original search_content.py ---

class SearchReference(BaseModel):
    """Represents a reference found during a search, linked to specific text spans."""
    start_index: int
    end_index: int
    title: str
    url: str

class SearchContent(BaseModel):
    """Represents content generated from a search, including text and references."""
    text: str
    references: List[SearchReference] = Field(default_factory=list) # Default to empty list

# --- Models from original section_writer_child_merge.py ---

class Reference(BaseModel):
    """Represents a general reference, potentially for citation lists."""
    # Note: This is distinct from SearchReference. It was used for aggregated lists.
    url: str
    title: str
    info: Optional[str] = ""
    number: Optional[int] = None # Often assigned during post-processing

class SectionInput(BaseModel):
    """Input structure defining a section/topic to be researched."""
    # Original fields if needed, commented out as current logic uses research_topic primarily
    # title: str
    # purpose: str
    # goals: List[str]
    # additional_context: Optional[str] = None
    research_topic: str # Main input from user
    recursion_depth: int = 1
    skip_clarifications: bool = False
    # Adding the fields dynamically added during clarification
    clarifications: Optional[List[Dict[str, str]]] = None # Stores {question: ..., answer: ...} pairs
    generated_plan_summary: Optional[str] = None

class FurtherResearch(BaseModel):
    """Schema for extracting further research topics suggested by an agent."""
    further_research: List[str] = Field(default_factory=list)

class MergedWriteup(BaseModel):
    """Schema for representing a merged writeup (potentially from an agent)."""
    # Might not be strictly needed if agents just return strings,
    # but useful for consistency if an agent structures its merge output.
    writeup: str

class QuestionList(BaseModel):
    """Schema for extracting clarification questions from an agent."""
    questions: List[str] = Field(default_factory=list)

# --- Potentially add other shared data structures as needed ---
# Example: A model for status updates via WebSocket
class StatusUpdate(BaseModel):
    type: str = "status_update"
    run_id: str
    status: str
    error: Optional[str] = None
    # Add other relevant fields like progress percentage, current step etc.

class LogUpdateMessage(BaseModel):
     type: str = "log_update"
     run_id: str
     log_data: Dict[str, Any] # Contains nodes, tree, session_info

class LatestOutputMessage(BaseModel):
     type: str = "latest_output"
     run_id: str
     output_html: Optional[str] = None
     output_md: Optional[str] = None
     source_node_id: Optional[str] = None # ID of the node that generated this output