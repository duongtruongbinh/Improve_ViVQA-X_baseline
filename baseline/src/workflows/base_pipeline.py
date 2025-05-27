import time
import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from ..image_utils import VLMImageProcessor
from ..evaluation import perform_direct_accuracy_check

@dataclass
class PipelineResult:
    """Standardized result structure for VQA pipelines."""
    question_id: str
    image_path: str
    question: str
    target_answers: Any
    question_type: str
    final_answer: str
    confidence_score: float
    processing_time_seconds: float
    error: Optional[str]
    direct_accuracy_check: Dict[str, Any]
    additional_metadata: Dict[str, Any]

class BaseVQAPipeline:
    """Base class for VQA pipelines with common functionality."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        f1_threshold: float = 0.5
    ):
        self.config = config
        self.f1_threshold = f1_threshold
        self.image_processor = VLMImageProcessor(config=config)
        
        # Setup logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("BaseVQAPipeline")
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
                
    def create_error_result(
        self,
        question_id: str,
        image_path: str,
        question: str,
        target_answer: Any,
        error_message: str,
        question_type: str = "other"
    ) -> PipelineResult:
        """Create a standardized error result."""
        return PipelineResult(
            question_id=str(question_id),
            image_path=image_path,
            question=question,
            target_answers=target_answer,
            question_type=question_type,
            final_answer="[Pipeline Error]",
            confidence_score=0.0,
            processing_time_seconds=0.0,
            error=error_message,
            direct_accuracy_check={},
            additional_metadata={}
        )
        
    def process_image(self, image_path: str) -> tuple[Optional[str], Optional[str]]:
        """Process image and return URL and any error message."""
        try:
            processed_url = self.image_processor.process(image_path)
            if not isinstance(processed_url, str) or not \
               (processed_url.startswith("data:image/") or processed_url.startswith("http")):
                raise ValueError(f"Invalid image URL/data URI: {str(processed_url)[:100]}")
            return processed_url, None
        except Exception as e:
            error_msg = f"ImageProcessingError: {str(e)}"
            self.logger.error(f"Error processing image '{os.path.basename(image_path)}': {e}", exc_info=True)
            return None, error_msg
            
    def evaluate_result(
        self,
        final_answer: str,
        target_answer: Any,
        question_type: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate result against target answer."""
        try:
            direct_check_results, _, _ = perform_direct_accuracy_check(
                final_answer_text=final_answer,
                target_answers=target_answer,
                question_type=question_type,
                current_error=error,
                verbose=self.logger.isEnabledFor(logging.DEBUG),
                f1_threshold_other=self.f1_threshold
            )
            return direct_check_results
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}", exc_info=True)
            return {}
            
    def create_success_result(
        self,
        question_id: str,
        image_path: str,
        question: str,
        target_answer: Any,
        final_answer: str,
        confidence_score: float,
        processing_time: float,
        question_type: str = "other",
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Create a standardized success result."""
        # Evaluate result
        direct_check_results = self.evaluate_result(
            final_answer=final_answer,
            target_answer=target_answer,
            question_type=question_type
        )
        
        return PipelineResult(
            question_id=str(question_id),
            image_path=image_path,
            question=question,
            target_answers=target_answer,
            question_type=question_type,
            final_answer=final_answer,
            confidence_score=confidence_score,
            processing_time_seconds=round(processing_time, 2),
            error=None,
            direct_accuracy_check=direct_check_results,
            additional_metadata=additional_metadata or {}
        )
        
    async def run_pipeline(
        self,
        image_path: str,
        question: str,
        question_id: str = "unknown_qid",
        target_answer: Any = None,
        question_type: str = "other"
    ) -> PipelineResult:
        """Run the VQA pipeline - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run_pipeline") 