from typing import Dict, List, Optional, Union, Literal, TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from app.tools.doc_proccesing.gemini_vertex_tool import GeminiVertexDocumentTool

logger = logging.getLogger(__name__)

# State Management
class DocumentGroup(BaseModel):
    """Represents a group of related documents"""
    group_name: str
    file_paths: List[str]
    confidence: float
    document_type: str
    analysis: Optional[Dict] = None

class ChatMessage(BaseModel):
    """Represents a message in the chat interface"""
    role: Literal["human", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class FlowState(BaseModel):
    """Complete state for the ontology creation flow"""
    # Document and grouping state
    input_files: List[str] = Field(default_factory=list)
    document_groups: List[DocumentGroup] = Field(default_factory=list)
    grouping_approved: bool = False
    
    # Chat and interaction state
    chat_history: List[ChatMessage] = Field(default_factory=list)
    last_human_message: Optional[str] = None
    waiting_for_human: bool = False
    
    # Ontology creation state
    extraction_requirements: Optional[str] = None
    draft_ontology: Optional[Dict] = None
    ontology_approved: bool = False
    
    # Output state
    pydantic_classes: Optional[str] = None
    current_step: str = "init"
    errors: List[str] = Field(default_factory=list)

# Gemini Document Analysis Node
class DocumentAnalysisNode:
    def __init__(self):
        self.tool = GeminiVertexDocumentTool()
    
    def analyze_document_types(self, files: List[str]) -> Dict:
        """Ask Gemini to analyze if documents are of the same type"""
        system_prompt = """You are a document analysis expert. Analyze these documents and determine:
        1. If they are all the same type
        2. What type(s) they are
        3. Any potential groupings needed
        Return your analysis in a structured format."""
        
        try:
            result = self.tool.analyze_documents(
                file_locations=files,
                system_prompt=system_prompt,
                question="Are these documents of the same type? What types are they?",
                return_json_only=True
            )
            return result
        except Exception as e:
            logger.error(f"Error analyzing documents: {str(e)}")
            raise

    def __call__(self, state: FlowState) -> FlowState:
        try:
            # Analyze documents using Gemini
            analysis = self.analyze_document_types(state.input_files)
            
            # Create document groups based on analysis
            if analysis.get("documents_same_type", False):
                group = DocumentGroup(
                    group_name="main",
                    file_paths=state.input_files,
                    confidence=analysis.get("confidence", 0.0),
                    document_type=analysis.get("document_type", "unknown"),
                    analysis=analysis
                )
                state.document_groups = [group]
            else:
                # Handle multiple groups if documents are different types
                groups = analysis.get("groups", [])
                state.document_groups = [
                    DocumentGroup(
                        group_name=f"group_{i}",
                        file_paths=group["files"],
                        confidence=group.get("confidence", 0.0),
                        document_type=group.get("type", "unknown"),
                        analysis=group
                    )
                    for i, group in enumerate(groups)
                ]
            
            # Add analysis results to chat history
            state.chat_history.append(ChatMessage(
                role="assistant",
                content=f"I've analyzed the documents. Found {len(state.document_groups)} group(s)."
            ))
            
            state.waiting_for_human = True
            state.current_step = "document_analysis"
            
            return state
            
        except Exception as e:
            state.errors.append(f"Document analysis error: {str(e)}")
            state.chat_history.append(ChatMessage(
                role="assistant",
                content=f"Error analyzing documents: {str(e)}"
            ))
            return state

# HITL Interaction Node
class HITLInteractionNode:
    def __call__(self, state: FlowState) -> FlowState:
        if not state.last_human_message:
            state.waiting_for_human = True
            return state
            
        # Process human message based on current step
        if state.current_step == "document_analysis":
            if "approve" in state.last_human_message.lower():
                state.grouping_approved = True
                state.current_step = "requirements"
                state.chat_history.append(ChatMessage(
                    role="assistant",
                    content="Great! What information would you like to extract from these documents?"
                ))
            else:
                state.chat_history.append(ChatMessage(
                    role="assistant",
                    content="Please let me know if you'd like to regroup the documents differently."
                ))
                
        elif state.current_step == "requirements":
            state.extraction_requirements = state.last_human_message
            state.current_step = "ontology_creation"
            
        state.waiting_for_human = True
        return state

# Ontology Creation Node
class OntologyCreationNode:
    def __init__(self):
        self.tool = GeminiVertexDocumentTool()
    
    def __call__(self, state: FlowState) -> FlowState:
        if not state.extraction_requirements:
            return state
            
        try:
            # Use Gemini to create ontology based on requirements
            system_prompt = """You are an ontology creation expert. Create a Pydantic-based ontology that:
            1. Captures the requested information
            2. Uses appropriate data types and validation
            3. Follows Python best practices"""
            
            result = self.tool.analyze_documents(
                file_locations=[state.document_groups[0].file_paths[0]],  # Use first document as example
                system_prompt=system_prompt,
                question=f"Create an ontology for extracting: {state.extraction_requirements}",
                return_json_only=True
            )
            
            state.draft_ontology = result
            state.chat_history.append(ChatMessage(
                role="assistant",
                content="I've created a draft ontology. Would you like to review it?"
            ))
            
            state.waiting_for_human = True
            state.current_step = "ontology_review"
            
            return state
            
        except Exception as e:
            state.errors.append(f"Ontology creation error: {str(e)}")
            return state

def create_ontology_flow() -> StateGraph:
    """Create the ontology creation workflow"""
    
    workflow = StateGraph(FlowState)
    
    # Add nodes
    workflow.add_node("analyze", DocumentAnalysisNode())
    workflow.add_node("interact", HITLInteractionNode())
    workflow.add_node("create_ontology", OntologyCreationNode())
    
    # Add edges
    def should_continue(state: FlowState) -> str:
        if not state.grouping_approved:
            return "interact"
        elif not state.extraction_requirements:
            return "interact"
        elif not state.ontology_approved:
            return "create_ontology"
        else:
            return "end"
    
    workflow.add_edge("analyze", "interact")
    workflow.add_edge("interact", "create_ontology")
    workflow.add_conditional_edges(
        "create_ontology",
        should_continue,
        {
            "interact": "interact",
            "create_ontology": "create_ontology",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    return workflow.compile()