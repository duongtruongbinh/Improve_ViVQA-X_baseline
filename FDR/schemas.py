"""
Pydantic schemas for Vietnamese VQA Architecture
Tuân thủ đặc tả: https://github.com/user/VQA-Architecture
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class AnswerCandidates(BaseModel):
    """Ứng viên trả lời từ Verifier"""
    candidates: List[str] = Field(description="Danh sách 3 ứng viên trả lời")
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy")


class TextPrompts(BaseModel):
    """Text prompts cho GroundingDINO"""
    prompts: List[str] = Field(description="Danh sách các prompt phát hiện đối tượng")
    
    
class ImageWithBoxes(BaseModel):
    """Ảnh với bounding boxes từ GroundingDINO"""
    image_path: str = Field(description="Đường dẫn ảnh gốc")
    annotated_image_path: Optional[str] = Field(description="Đường dẫn ảnh có annotation")
    boxes_xyxy: Optional[List[List[float]]] = Field(description="Tọa độ bounding boxes")


class DetailedDescription(BaseModel):
    """Mô tả chi tiết từ DAM"""
    description: str = Field(description="Mô tả chi tiết vùng được phát hiện")
    region_analysis: Dict[str, Any] = Field(description="Phân tích từng vùng")


class RelevantIssue(BaseModel):
    """Câu hỏi phụ từ Strategist"""
    issue: str = Field(description="Câu hỏi phụ để phân biệt các ứng viên")
    answer: str = Field(description="Câu trả lời cho issue")
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy câu trả lời")


class Hypothesis(BaseModel):
    """Giả thuyết từ MVKB"""
    hypothesis: str = Field(description="Giả thuyết dạng IF-THEN")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Điểm tin cậy")
    confidence_word: Literal["Unlikely", "Possible", "Likely"] = Field(description="Từ mô tả tin cậy")
    answer_candidate: str = Field(description="Ứng viên trả lời liên quan")
    relevant_issue: str = Field(description="Câu hỏi phụ liên quan")
    issue_answer: str = Field(description="Câu trả lời cho câu hỏi phụ")


class MVKB(BaseModel):
    """Multi-View Knowledge Base"""
    hypotheses: List[Hypothesis] = Field(description="Danh sách các giả thuyết")
    
    
class VotingPool(BaseModel):
    """Kết quả bỏ phiếu từ Synthesizer"""
    votes: Dict[str, float] = Field(description="Điểm bỏ phiếu cho từng ứng viên")
    

class FinalAnswer(BaseModel):
    """Câu trả lời cuối cùng và giải thích"""
    answer: str = Field(description="Câu trả lời cuối cùng")
    causal_explanation: str = Field(description="Giải thích nhân quả chặt chẽ")
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy cuối cùng")


class VQAInput(BaseModel):
    """Input cho hệ thống VQA"""
    user_question: str = Field(description="Câu hỏi của người dùng")
    image_path: str = Field(description="Đường dẫn ảnh")


class VQAState(BaseModel):
    """State tổng thể cho LangGraph workflow"""
    # Input
    user_question: str
    image_path: str
    
    # Luồng A: Bottom-up Evidence Gathering
    text_prompts: Optional[TextPrompts] = None
    image_with_boxes: Optional[ImageWithBoxes] = None
    detailed_description: Optional[DetailedDescription] = None
    
    # Luồng B: Top-down Hypothesis Testing  
    answer_candidates: Optional[AnswerCandidates] = None
    relevant_issues: List[RelevantIssue] = Field(default_factory=list)
    mvkb: Optional[MVKB] = None
    
    # Tổng hợp cuối cùng
    voting_pool: Optional[VotingPool] = None
    final_answer: Optional[FinalAnswer] = None
    
    # Metadata
    processing_time: Optional[float] = None
    error_message: Optional[str] = None 