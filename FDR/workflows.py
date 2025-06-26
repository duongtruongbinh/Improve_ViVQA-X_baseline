"""
LangGraph Workflows for Vietnamese VQA Architecture
Triển khai 2 luồng song song: Bottom-up & Top-down
"""

import logging
import time
from typing import Dict, Any, List, TypedDict
from typing_extensions import Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableParallel

from .schemas import VQAState, VQAInput, FinalAnswer, MVKB, Hypothesis
from .agents import create_vqa_agents
from .tools import get_available_tools


class VQAGraphState(TypedDict):
    """State for LangGraph workflow"""
    # Input
    user_question: str
    image_path: str
    
    # Luồng A: Bottom-up Evidence Gathering
    text_prompts: Annotated[List[str], add_messages]
    image_with_boxes: Dict[str, Any]
    detailed_description: str
    
    # Luồng B: Top-down Hypothesis Testing  
    answer_candidates: List[str]
    answer_confidence: float
    relevant_issues: List[Dict[str, Any]]
    mvkb_hypotheses: List[Dict[str, Any]]
    
    # Tổng hợp cuối cùng
    voting_pool: Dict[str, float]
    final_answer: Dict[str, Any]
    
    # Metadata
    processing_time: float
    error_message: str


class VietnameseVQAWorkflow:
    """
    LangGraph workflow triển khai đặc tả Vietnamese VQA Architecture
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.agents = create_vqa_agents(model_name)
        self.tools = {tool.name: tool for tool in get_available_tools()}
        self.workflow = None
        self._build_workflow()
    
    def _build_workflow(self):
        """Xây dựng LangGraph workflow với 2 luồng song song"""
        
        # Initialize StateGraph
        workflow = StateGraph(VQAGraphState)
        
        # Add nodes cho từng component theo đặc tả
        workflow.add_node("concept_extractor", self._concept_extractor_node)
        workflow.add_node("grounding_dino", self._grounding_dino_node)
        workflow.add_node("dam_analysis", self._dam_analysis_node)
        workflow.add_node("initiator_guesser", self._initiator_guesser_node)
        workflow.add_node("questioner", self._questioner_node)
        workflow.add_node("sub_question_answerer", self._sub_question_answerer_node)
        workflow.add_node("hypothesis_builder", self._hypothesis_builder_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("final_decider", self._final_decider_node)
        
        # Define workflow edges theo đặc tả
        # Khởi tạo: song song Bottom-up và Top-down
        workflow.add_edge(START, "concept_extractor")  # Luồng A
        workflow.add_edge(START, "initiator_guesser")  # Luồng B
        
        # Luồng A: Bottom-up Evidence Gathering
        workflow.add_edge("concept_extractor", "grounding_dino")
        workflow.add_edge("grounding_dino", "dam_analysis")
        
        # Luồng B: Top-down Hypothesis Testing
        workflow.add_edge("initiator_guesser", "questioner")
        workflow.add_edge("questioner", "sub_question_answerer")
        workflow.add_edge("sub_question_answerer", "hypothesis_builder")
        
        # Tổng hợp: cả 2 luồng → Synthesizer
        workflow.add_edge("dam_analysis", "synthesizer")
        workflow.add_edge("hypothesis_builder", "synthesizer")
        
        # Quyết định cuối cùng
        workflow.add_edge("synthesizer", "final_decider")
        workflow.add_edge("final_decider", END)
        
        # Compile workflow
        self.workflow = workflow.compile()
        
        logging.info("✅ Vietnamese VQA workflow built successfully")
    
    def _concept_extractor_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 1: Verifier (VLM) - Concept Extractor
        Trích xuất khái niệm từ câu hỏi
        """
        try:
            logging.info("📝 Luồng A - Step 1: Concept Extraction")
            
            result = self.agents["concept_extractor"].extract_concepts(state["user_question"])
            
            return {
                "text_prompts": result.prompts
            }
        except Exception as e:
            logging.error(f"Concept extraction failed: {e}")
            return {
                "text_prompts": ["object", "item"],
                "error_message": str(e)
            }
    
    def _grounding_dino_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 2: GroundingDINO
        Phát hiện và định vị đối tượng
        """
        try:
            logging.info("🎯 Luồng A - Step 2: GroundingDINO Detection")
            
            if "grounding_dino" not in self.tools:
                raise ValueError("GroundingDINO tool not available")
            
            grounding_dino = self.tools["grounding_dino"]
            result = grounding_dino._run(
                image_path=state["image_path"],
                text_prompts=state.get("text_prompts", ["object"]),
                box_threshold=0.3,
                text_threshold=0.25
            )
            
            return {
                "image_with_boxes": {
                    "image_path": result.image_path,
                    "annotated_image_path": result.annotated_image_path,
                    "boxes_xyxy": result.boxes_xyxy
                }
            }
        except Exception as e:
            logging.error(f"GroundingDINO failed: {e}")
            return {
                "image_with_boxes": {
                    "image_path": state["image_path"],
                    "annotated_image_path": None,
                    "boxes_xyxy": None
                },
                "error_message": str(e)
            }
    
    def _dam_analysis_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 3: DAM (Describe Anything Model)
        Tạo mô tả chi tiết cho các vùng đã phát hiện
        """
        try:
            logging.info("🔍 Luồng A - Step 3: DAM Analysis")
            
            if "dam" not in self.tools:
                raise ValueError("DAM tool not available")
            
            dam = self.tools["dam"]
            image_with_boxes = state.get("image_with_boxes", {})
            
            result = dam._run(
                image_path=state["image_path"],
                question=state["user_question"],
                boxes_xyxy=image_with_boxes.get("boxes_xyxy"),
                temperature=0.2,
                max_tokens=256
            )
            
            return {
                "detailed_description": result.description
            }
        except Exception as e:
            logging.error(f"DAM analysis failed: {e}")
            return {
                "detailed_description": "Không thể phân tích chi tiết hình ảnh.",
                "error_message": str(e)
            }
    
    def _initiator_guesser_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 4: Verifier (VLM) - Initiator & Guesser
        Đưa ra các ứng viên trả lời ban đầu
        """
        try:
            logging.info("🤔 Luồng B - Step 1: Answer Candidates Generation")
            
            result = self.agents["initiator_guesser"].generate_candidates(
                state["user_question"], 
                state["image_path"]
            )
            
            return {
                "answer_candidates": result.candidates,
                "answer_confidence": result.confidence
            }
        except Exception as e:
            logging.error(f"Candidate generation failed: {e}")
            return {
                "answer_candidates": ["unknown", "uncertain", "unclear"],
                "answer_confidence": 0.1,
                "error_message": str(e)
            }
    
    def _questioner_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 5: Strategist (LLM) - Questioner
        Tạo câu hỏi phụ để phân biệt các ứng viên
        """
        try:
            logging.info("❓ Luồng B - Step 2: Relevant Issues Creation")
            
            # Use detailed description if available, otherwise use basic caption
            caption = state.get("detailed_description", "Image analysis not available")
            
            issues = self.agents["questioner"].create_relevant_issues(
                state["user_question"],
                state.get("answer_candidates", []),
                caption
            )
            
            return {
                "relevant_issues": [{"issue": issue} for issue in issues]
            }
        except Exception as e:
            logging.error(f"Issue creation failed: {e}")
            return {
                "relevant_issues": [{"issue": "What can you see in the image?"}],
                "error_message": str(e)
            }
    
    def _sub_question_answerer_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 6: Verifier (VLM) - Sub-question Answerer
        Trả lời các câu hỏi phụ
        """
        try:
            logging.info("💬 Luồng B - Step 3: Sub-question Answering")
            
            answered_issues = []
            for issue_dict in state.get("relevant_issues", []):
                issue = issue_dict["issue"]
                
                result = self.agents["sub_question_answerer"].answer_issue(
                    issue,
                    state["user_question"],
                    state["image_path"]
                )
                
                answered_issues.append({
                    "issue": result.issue,
                    "answer": result.answer,
                    "confidence": result.confidence
                })
            
            return {
                "relevant_issues": answered_issues
            }
        except Exception as e:
            logging.error(f"Sub-question answering failed: {e}")
            return {
                "relevant_issues": [{"issue": "Error", "answer": "uncertain", "confidence": 0.1}],
                "error_message": str(e)
            }
    
    def _hypothesis_builder_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 7: Strategist (LLM) - Hypothesis Builder
        Xây dựng MVKB với các giả thuyết
        """
        try:
            logging.info("🧠 Luồng B - Step 4: MVKB Construction")
            
            hypotheses = []
            answer_candidates = state.get("answer_candidates", [])
            relevant_issues = state.get("relevant_issues", [])
            
            for issue_dict in relevant_issues:
                for candidate in answer_candidates:
                    hypothesis = self.agents["hypothesis_builder"].build_hypothesis(
                        state["user_question"],
                        candidate,
                        issue_dict["issue"],
                        issue_dict["answer"]
                    )
                    
                    hypotheses.append({
                        "hypothesis": hypothesis.hypothesis,
                        "confidence_score": hypothesis.confidence_score,
                        "confidence_word": hypothesis.confidence_word,
                        "answer_candidate": hypothesis.answer_candidate,
                        "relevant_issue": hypothesis.relevant_issue,
                        "issue_answer": hypothesis.issue_answer
                    })
            
            return {
                "mvkb_hypotheses": hypotheses
            }
        except Exception as e:
            logging.error(f"Hypothesis building failed: {e}")
            return {
                "mvkb_hypotheses": [],
                "error_message": str(e)
            }
    
    def _synthesizer_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 8: Synthesizer (Algorithm)
        Tổng hợp kết quả bỏ phiếu
        """
        try:
            logging.info("⚖️ Tổng hợp: Weighted Voting")
            
            # Create MVKB object
            hypotheses = []
            for h_dict in state.get("mvkb_hypotheses", []):
                hypothesis = Hypothesis(
                    hypothesis=h_dict["hypothesis"],
                    confidence_score=h_dict["confidence_score"],
                    confidence_word=h_dict["confidence_word"],
                    answer_candidate=h_dict["answer_candidate"],
                    relevant_issue=h_dict["relevant_issue"],
                    issue_answer=h_dict["issue_answer"]
                )
                hypotheses.append(hypothesis)
            
            mvkb = MVKB(hypotheses=hypotheses)
            
            # Conduct voting
            voting_result = self.agents["synthesizer"].conduct_voting(
                state["user_question"],
                state["image_path"],
                state.get("answer_candidates", []),
                mvkb
            )
            
            return {
                "voting_pool": voting_result.votes
            }
        except Exception as e:
            logging.error(f"Voting failed: {e}")
            candidates = state.get("answer_candidates", ["unknown"])
            equal_score = 1.0 / len(candidates)
            return {
                "voting_pool": {candidate: equal_score for candidate in candidates},
                "error_message": str(e)
            }
    
    def _final_decider_node(self, state: VQAGraphState) -> Dict[str, Any]:
        """
        Component 9: Strategist (LLM) - Final Decider & Explainer
        Quyết định cuối cùng và giải thích
        """
        try:
            logging.info("🎯 Quyết định cuối cùng: Final Decision & Explanation")
            
            # Prepare data structures
            from .schemas import VotingPool, MVKB, Hypothesis
            
            voting_pool = VotingPool(votes=state.get("voting_pool", {}))
            
            hypotheses = []
            for h_dict in state.get("mvkb_hypotheses", []):
                hypothesis = Hypothesis(
                    hypothesis=h_dict["hypothesis"],
                    confidence_score=h_dict["confidence_score"],
                    confidence_word=h_dict["confidence_word"],
                    answer_candidate=h_dict["answer_candidate"],
                    relevant_issue=h_dict["relevant_issue"],
                    issue_answer=h_dict["issue_answer"]
                )
                hypotheses.append(hypothesis)
            
            mvkb = MVKB(hypotheses=hypotheses)
            
            # Final decision
            final_answer = self.agents["final_decider_explainer"].decide_and_explain(
                state["user_question"],
                voting_pool,
                state.get("detailed_description", ""),
                mvkb
            )
            
            return {
                "final_answer": {
                    "answer": final_answer.answer,
                    "causal_explanation": final_answer.causal_explanation,
                    "confidence": final_answer.confidence
                }
            }
        except Exception as e:
            logging.error(f"Final decision failed: {e}")
            # Fallback to highest voted answer
            voting_pool = state.get("voting_pool", {})
            if voting_pool:
                best_answer = max(voting_pool.items(), key=lambda x: x[1])[0]
            else:
                best_answer = "unknown"
            
            return {
                "final_answer": {
                    "answer": best_answer,
                    "causal_explanation": "Fallback: Chọn câu trả lời có điểm cao nhất hoặc mặc định.",
                    "confidence": 0.5
                },
                "error_message": str(e)
            }
    
    def invoke(self, vqa_input: VQAInput) -> FinalAnswer:
        """
        Chạy toàn bộ workflow Vietnamese VQA
        """
        start_time = time.time()
        
        try:
            logging.info(f"🚀 Starting Vietnamese VQA workflow for: {vqa_input.user_question}")
            
            # Initialize state
            initial_state = {
                "user_question": vqa_input.user_question,
                "image_path": vqa_input.image_path,
                "text_prompts": [],
                "image_with_boxes": {},
                "detailed_description": "",
                "answer_candidates": [],
                "answer_confidence": 0.0,
                "relevant_issues": [],
                "mvkb_hypotheses": [],
                "voting_pool": {},
                "final_answer": {},
                "processing_time": 0.0,
                "error_message": ""
            }
            
            # Run workflow
            result = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract final answer
            final_answer_dict = result.get("final_answer", {})
            
            final_answer = FinalAnswer(
                answer=final_answer_dict.get("answer", "unknown"),
                causal_explanation=final_answer_dict.get("causal_explanation", "Không có giải thích."),
                confidence=final_answer_dict.get("confidence", 0.0)
            )
            
            logging.info(f"✅ Vietnamese VQA completed in {processing_time:.2f}s: {final_answer.answer}")
            
            return final_answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            logging.error(f"❌ Vietnamese VQA workflow failed after {processing_time:.2f}s: {e}")
            
            return FinalAnswer(
                answer="error",
                causal_explanation=f"Lỗi xử lý: {str(e)}",
                confidence=0.0
            )
    
    def get_graph_visualization(self) -> Any:
        """Get Mermaid visualization of the workflow"""
        try:
            return self.workflow.get_graph().draw_mermaid()
        except Exception as e:
            logging.error(f"Graph visualization failed: {e}")
            return None


# Factory function
def create_vietnamese_vqa_workflow(model_name: str = "gpt-4o-mini") -> VietnameseVQAWorkflow:
    """Create Vietnamese VQA workflow"""
    return VietnameseVQAWorkflow(model_name=model_name) 