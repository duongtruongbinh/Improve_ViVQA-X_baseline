"""
LangChain-based Vietnamese VQA Architecture
Tuân thủ hoàn toàn đặc tả: 2 luồng song song + 6 vai trò rõ ràng

Refactored from original architecture to use LangChain/LangGraph
for better modularity, type safety, and architectural compliance.
"""

from .schemas import (
    VQAInput,
    VQAState, 
    FinalAnswer,
    AnswerCandidates,
    TextPrompts,
    ImageWithBoxes,
    DetailedDescription,
    RelevantIssue,
    Hypothesis,
    MVKB,
    VotingPool
)

from .agents import (
    ConceptExtractorAgent,
    InitiatorGuesserAgent,
    SubQuestionAnswererAgent,
    QuestionerAgent,
    HypothesisBuilderAgent,
    FinalDeciderExplainerAgent,
    SynthesizerAgent,
    create_vqa_agents
)

from .tools import (
    GroundingDINOTool,
    DAMTool,
    get_available_tools
)

from .workflows import (
    VietnameseVQAWorkflow,
    create_vietnamese_vqa_workflow
)

__version__ = "1.0.0"
__author__ = "VQA Team"
__description__ = "LangChain-based Vietnamese VQA Architecture with full specification compliance"

# Export main interface
__all__ = [
    # Main interface
    "VQAInput",
    "FinalAnswer", 
    "create_vietnamese_vqa_workflow",
    
    # Schemas
    "VQAState",
    "AnswerCandidates",
    "TextPrompts", 
    "ImageWithBoxes",
    "DetailedDescription",
    "RelevantIssue",
    "Hypothesis",
    "MVKB",
    "VotingPool",
    
    # Agents (6 roles according to specification)
    "ConceptExtractorAgent",          # Verifier role 1
    "InitiatorGuesserAgent",          # Verifier role 2  
    "SubQuestionAnswererAgent",       # Verifier role 3
    "QuestionerAgent",                # Strategist role 1
    "HypothesisBuilderAgent",         # Strategist role 2
    "FinalDeciderExplainerAgent",     # Strategist role 3
    "SynthesizerAgent",               # Algorithm
    "create_vqa_agents",
    
    # Tools
    "GroundingDINOTool",
    "DAMTool", 
    "get_available_tools",
    
    # Workflows
    "VietnameseVQAWorkflow"
] 