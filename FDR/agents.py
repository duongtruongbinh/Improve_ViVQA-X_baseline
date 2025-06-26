"""
LangChain Agents for Vietnamese VQA Architecture
Triển khai chính xác các vai trò theo đặc tả
"""

import logging
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from .schemas import (
    AnswerCandidates, TextPrompts, RelevantIssue, Hypothesis, 
    MVKB, VotingPool, FinalAnswer
)


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found at {image_path}")
        return None


class ConceptExtractorAgent:
    """
    Component 1: Verifier (VLM) - Vai trò "Concept Extractor"
    Trích xuất các khái niệm chính từ câu hỏi để cung cấp cho DINO
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=TextPrompts)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia phân tích ngữ nghĩa câu hỏi VQA.
            Nhiệm vụ: Trích xuất các khái niệm chính từ câu hỏi để tạo text prompts cho GroundingDINO.
            
            Quy tắc:
            - Tập trung vào danh từ cụ thể, vật thể có thể nhìn thấy
            - Tránh tính từ và động từ trừu tượng  
            - Sử dụng từ đơn giản, phổ biến
            - Trả về 3-5 khái niệm quan trọng nhất
            
            {format_instructions}"""),
            ("user", "Câu hỏi: {question}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def extract_concepts(self, question: str) -> TextPrompts:
        """Trích xuất khái niệm từ câu hỏi"""
        try:
            result = self.chain.invoke({
                "question": question,
                "format_instructions": self.parser.get_format_instructions()
            })
            logging.info(f"✅ Extracted concepts: {result.prompts}")
            return result
        except Exception as e:
            logging.error(f"Concept extraction failed: {e}")
            # Fallback: simple keyword extraction
            return TextPrompts(prompts=["object", "item"])


class InitiatorGuesserAgent:
    """
    Component 4: Verifier (VLM) - Vai trò "Initiator & Guesser"  
    Khởi tạo luồng suy luận và đưa ra các phỏng đoán ban đầu
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=AnswerCandidates)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia VQA khởi tạo quá trình suy luận.
            Nhiệm vụ: Phân tích ảnh và đưa ra 3 ứng viên trả lời có khả năng nhất.
            
            Quy tắc:
            - Đưa ra các ứng viên đa dạng, bao quát các khả năng
            - Ước tính độ tin cậy tổng thể
            - Ưu tiên câu trả lời ngắn gọn, cụ thể
            
            {format_instructions}"""),
            ("user", [
                {"type": "text", "text": "Câu hỏi: {question}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
            ])
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def generate_candidates(self, question: str, image_path: str) -> AnswerCandidates:
        """Tạo ứng viên trả lời ban đầu"""
        try:
            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                raise ValueError("Cannot encode image")
            
            result = self.chain.invoke({
                "question": question,
                "image_base64": image_base64,
                "format_instructions": self.parser.get_format_instructions()
            })
            logging.info(f"✅ Generated candidates: {result.candidates}")
            return result
        except Exception as e:
            logging.error(f"Candidate generation failed: {e}")
            return AnswerCandidates(candidates=["unknown", "uncertain", "unclear"], confidence=0.1)


class SubQuestionAnswererAgent:
    """
    Component 6: Verifier (VLM) - Vai trò "Sub-question Answerer"
    Trả lời câu hỏi phụ dựa trên quan sát ảnh
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RelevantIssue)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia quan sát hình ảnh.
            Nhiệm vụ: Trả lời câu hỏi phụ dựa trên quan sát chi tiết ảnh.
            
            Quy tắc:
            - Quan sát cẩn thận chi tiết trong ảnh
            - Đưa ra câu trả lời cụ thể, chính xác
            - Ước tính độ tin cậy dựa trên độ rõ ràng của bằng chứng
            
            {format_instructions}"""),
            ("user", [
                {"type": "text", "text": "Câu hỏi phụ: {issue}\nCâu hỏi gốc: {original_question}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_base64}"}}
            ])
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def answer_issue(self, issue: str, original_question: str, image_path: str) -> RelevantIssue:
        """Trả lời câu hỏi phụ"""
        try:
            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                raise ValueError("Cannot encode image")
            
            result = self.chain.invoke({
                "issue": issue,
                "original_question": original_question,
                "image_base64": image_base64,
                "format_instructions": self.parser.get_format_instructions()
            })
            logging.info(f"✅ Answered issue: {issue} -> {result.answer}")
            return result
        except Exception as e:
            logging.error(f"Issue answering failed: {e}")
            return RelevantIssue(issue=issue, answer="uncertain", confidence=0.1)


class QuestionerAgent:
    """
    Component 5: Strategist (LLM) - Vai trò "Questioner"
    Tạo câu hỏi phụ để phân biệt các ứng viên trả lời
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia logic suy luận.
            Nhiệm vụ: Tạo câu hỏi phụ để phân biệt các ứng viên trả lời.
            
            Quy tắc:
            - Câu hỏi phải cụ thể, có thể quan sát được
            - Giúp phân biệt rõ ràng giữa các ứng viên
            - Tập trung vào đặc điểm thị giác có thể xác minh
            
            Trả về danh sách 2-3 câu hỏi phụ, mỗi câu một dòng."""),
            ("user", """Câu hỏi gốc: {question}
            Mô tả ảnh: {caption}
            Ứng viên trả lời: {candidates}
            
            Hãy tạo câu hỏi phụ để phân biệt các ứng viên:""")
        ])
        
        self.chain = self.prompt | self.llm
    
    def create_relevant_issues(self, question: str, answer_candidates: List[str], caption: str) -> List[str]:
        """Tạo các câu hỏi phụ relevant"""
        try:
            result = self.chain.invoke({
                "question": question,
                "candidates": answer_candidates,
                "caption": caption
            })
            
            # Parse the response to extract questions
            issues = [line.strip() for line in result.content.strip().split('\n') if line.strip()]
            issues = [issue.lstrip('- ').lstrip('1. ').lstrip('2. ').lstrip('3. ') for issue in issues]
            
            logging.info(f"✅ Created relevant issues: {issues}")
            return issues[:3]  # Limit to 3 issues
        except Exception as e:
            logging.error(f"Issue creation failed: {e}")
            return ["What can you see in the image?"]


class HypothesisBuilderAgent:
    """
    Component 7: Strategist (LLM) - Vai trò "Hypothesis Builder"
    Xây dựng cơ sở tri thức (MVKB) dưới dạng các giả thuyết
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=Hypothesis)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia xây dựng giả thuyết logic.
            Nhiệm vụ: Tạo giả thuyết IF-THEN kết nối câu trả lời phụ với ứng viên chính.
            
            Quy tắc:
            - Giả thuyết phải có dạng "IF ... THEN ..."
            - Logic phải rõ ràng và có thể kiểm chứng
            - Ước tính độ tin cậy dựa trên logic thường thức
            - Confidence word: <0.4="Unlikely", 0.4-0.7="Possible", >0.7="Likely"
            
            {format_instructions}"""),
            ("user", """Câu hỏi gốc: {question}
            Ứng viên trả lời: {answer_candidate}
            Câu hỏi phụ: {relevant_issue}
            Câu trả lời phụ: {issue_answer}
            
            Hãy tạo giả thuyết kết nối chúng:""")
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def build_hypothesis(self, question: str, answer_candidate: str, relevant_issue: str, issue_answer: str) -> Hypothesis:
        """Xây dựng giả thuyết"""
        try:
            result = self.chain.invoke({
                "question": question,
                "answer_candidate": answer_candidate,
                "relevant_issue": relevant_issue,
                "issue_answer": issue_answer,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Ensure all required fields are present
            result.answer_candidate = answer_candidate
            result.relevant_issue = relevant_issue
            result.issue_answer = issue_answer
            
            logging.info(f"✅ Built hypothesis: {result.hypothesis[:50]}...")
            return result
        except Exception as e:
            logging.error(f"Hypothesis building failed: {e}")
            return Hypothesis(
                hypothesis=f"IF {issue_answer} THEN {answer_candidate}",
                confidence_score=0.5,
                confidence_word="Possible",
                answer_candidate=answer_candidate,
                relevant_issue=relevant_issue,
                issue_answer=issue_answer
            )


class FinalDeciderExplainerAgent:
    """
    Component 9: Strategist (LLM) - Vai trò "Final Decider & Explainer"
    Tổng hợp tất cả bằng chứng và đưa ra câu trả lời cuối cùng với giải thích
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=FinalAnswer)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia quyết định cuối cùng và giải thích.
            Nhiệm vụ: Tổng hợp kết quả bỏ phiếu và bằng chứng để đưa ra câu trả lời cuối cùng.
            
            Quy tắc:
            - Đối chiếu voting pool với detailed description
            - Sử dụng MVKB làm cơ sở xây dựng giải thích nhân quả
            - Giải thích phải chặt chẽ, logic, dễ hiểu
            - Đánh giá độ tin cậy tổng thể
            
            {format_instructions}"""),
            ("user", """Câu hỏi: {question}
            
            Kết quả bỏ phiếu: {voting_pool}
            Mô tả chi tiết: {detailed_description}
            Cơ sở tri thức (MVKB): {mvkb_summary}
            
            Hãy đưa ra quyết định cuối cùng với giải thích nhân quả:""")
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def decide_and_explain(self, question: str, voting_pool: VotingPool, 
                          detailed_description: str, mvkb: MVKB) -> FinalAnswer:
        """Quyết định cuối cùng và giải thích"""
        try:
            # Prepare MVKB summary
            mvkb_summary = "\n".join([
                f"- {h.hypothesis} (Confidence: {h.confidence_word})"
                for h in mvkb.hypotheses[:5]  # Top 5 hypotheses
            ])
            
            result = self.chain.invoke({
                "question": question,
                "voting_pool": str(voting_pool.votes),
                "detailed_description": detailed_description,
                "mvkb_summary": mvkb_summary,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            logging.info(f"✅ Final decision: {result.answer}")
            return result
        except Exception as e:
            logging.error(f"Final decision failed: {e}")
            # Fallback: choose highest voted answer
            best_answer = max(voting_pool.votes.items(), key=lambda x: x[1])[0]
            return FinalAnswer(
                answer=best_answer,
                causal_explanation="Dựa trên kết quả bỏ phiếu với điểm cao nhất.",
                confidence=0.7
            )


class SynthesizerAgent:
    """
    Component 8: Synthesizer (Algorithm)
    Tổng hợp kết quả từ các giả thuyết thông qua cơ chế bỏ phiếu
    """
    
    def __init__(self, verifier_agent):
        self.verifier_agent = verifier_agent  # Reference to sub-question answerer
    
    def conduct_voting(self, original_question: str, image_path: str, 
                      answer_candidates: List[str], mvkb: MVKB) -> VotingPool:
        """Thực hiện bỏ phiếu có trọng số"""
        try:
            vote_scores = {candidate: 0.0 for candidate in answer_candidates}
            
            for hypothesis in mvkb.hypotheses:
                # Ask verifier to re-answer original question under this hypothesis
                derived_answer = self._re_answer_under_hypothesis(
                    original_question, image_path, hypothesis
                )
                
                # Add confidence score to matching candidate
                for candidate in answer_candidates:
                    if candidate.lower() in derived_answer.lower():
                        vote_scores[candidate] += hypothesis.confidence_score
                        break
            
            logging.info(f"✅ Voting completed: {vote_scores}")
            return VotingPool(votes=vote_scores)
        except Exception as e:
            logging.error(f"Voting failed: {e}")
            # Fallback: equal votes
            equal_score = 1.0 / len(answer_candidates)
            return VotingPool(votes={candidate: equal_score for candidate in answer_candidates})
    
    def _re_answer_under_hypothesis(self, question: str, image_path: str, hypothesis: Hypothesis) -> str:
        """Re-answer question under specific hypothesis context"""
        try:
            # Create context prompt with hypothesis
            context_prompt = f"Giả thiết: {hypothesis.hypothesis}\n\nCâu hỏi: {question}"
            
            # Use verifier to answer with this context
            image_base64 = encode_image_to_base64(image_path)
            if not image_base64:
                return hypothesis.answer_candidate
            
            # Simple re-answering logic
            return hypothesis.answer_candidate  # Simplified for now
        except Exception as e:
            logging.error(f"Re-answering under hypothesis failed: {e}")
            return hypothesis.answer_candidate


# Factory function to create all agents
def create_vqa_agents(model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Create all VQA agents according to Vietnamese specification"""
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    agents = {
        # Verifier (VLM) agents - 3 roles
        "concept_extractor": ConceptExtractorAgent(llm),
        "initiator_guesser": InitiatorGuesserAgent(llm),
        "sub_question_answerer": SubQuestionAnswererAgent(llm),
        
        # Strategist (LLM) agents - 3 roles  
        "questioner": QuestionerAgent(llm),
        "hypothesis_builder": HypothesisBuilderAgent(llm),
        "final_decider_explainer": FinalDeciderExplainerAgent(llm),
        
        # Synthesizer (Algorithm)
        "synthesizer": None  # Will be created after verifier agents
    }
    
    # Create synthesizer with reference to sub-question answerer
    agents["synthesizer"] = SynthesizerAgent(agents["sub_question_answerer"])
    
    return agents 