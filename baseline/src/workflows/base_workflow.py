import time
import os
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..image_utils import VLMImageProcessor
from ..evaluation import perform_direct_accuracy_check

class WorkflowStatus(Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowResult:
    """Standardized result structure for VQA workflows."""
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
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: WorkflowStatus = WorkflowStatus.COMPLETED

    def is_success(self) -> bool:
        """Check if workflow completed successfully."""
        return self.status == WorkflowStatus.COMPLETED and self.error is None

    def get_accuracy_score(self) -> float:
        """Get the accuracy score from direct check results."""
        return self.direct_accuracy_check.get("accuracy", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "question_id": self.question_id,
            "image_path": self.image_path,
            "question": self.question,
            "target_answers": self.target_answers,
            "question_type": self.question_type,
            "final_answer": self.final_answer,
            "confidence_score": self.confidence_score,
            "processing_time_seconds": self.processing_time_seconds,
            "error": self.error,
            "direct_accuracy_check": self.direct_accuracy_check,
            "additional_metadata": self.additional_metadata,
            "timestamp": self.timestamp,
            "status": self.status.value
        }

class BaseVQAWorkflow:
    """Base class for VQA workflows with common functionality."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        f1_threshold: float = 0.5
    ):
        self.config = config
        self.f1_threshold = f1_threshold
        self.image_processor = VLMImageProcessor(config=config)
        self._current_status = WorkflowStatus.PENDING
        
        # Setup logging
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._current_status

    def _set_status(self, status: WorkflowStatus) -> None:
        """Set workflow status with logging."""
        self._current_status = status
        self.logger.info(f"Workflow status changed to: {status.value}")

    def _log_error(self, error: Exception, context: str) -> str:
        """Log error and return formatted error message."""
        error_msg = f"{context}: {str(error)}"
        self.logger.error(error_msg, exc_info=True)
        return error_msg
                
    def create_error_result(
        self,
        question_id: str,
        image_path: str,
        question: str,
        target_answer: Any,
        error_message: str,
        question_type: str = "other"
    ) -> WorkflowResult:
        """Create a standardized error result."""
        self._set_status(WorkflowStatus.FAILED)
        return WorkflowResult(
            question_id=str(question_id),
            image_path=image_path,
            question=question,
            target_answers=target_answer,
            question_type=question_type,
            final_answer="[Workflow Error]",
            confidence_score=0.0,
            processing_time_seconds=0.0,
            error=error_message,
            direct_accuracy_check={},
            additional_metadata={},
            status=WorkflowStatus.FAILED
        )
        
    def process_image(self, image_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Process image and return URL and any error message."""
        try:
            processed_url = self.image_processor.process(image_path)
            if not isinstance(processed_url, str) or not \
               (processed_url.startswith("data:image/") or processed_url.startswith("http")):
                raise ValueError(f"Invalid image URL/data URI: {str(processed_url)[:100]}")
            return processed_url, None
        except Exception as e:
            error_msg = self._log_error(e, f"Error processing image '{os.path.basename(image_path)}'")
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
            self._log_error(e, "Error during evaluation")
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
    ) -> WorkflowResult:
        """Create a standardized success result."""
        # Evaluate result
        direct_check_results = self.evaluate_result(
            final_answer=final_answer,
            target_answer=target_answer,
            question_type=question_type
        )
        
        self._set_status(WorkflowStatus.COMPLETED)
        return WorkflowResult(
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
            additional_metadata=additional_metadata or {},
            status=WorkflowStatus.COMPLETED
        )

    def validate_inputs(
        self,
        image_path: str,
        question: str,
        question_id: str,
        target_answer: Any
    ) -> Optional[str]:
        """Validate workflow inputs."""
        if not os.path.exists(image_path):
            return f"Image file not found: {image_path}"
        if not question or not isinstance(question, str):
            return "Invalid question format"
        if not question_id:
            return "Question ID is required"
        return None
        
    async def run_workflow(
        self,
        image_path: str,
        question: str,
        question_id: str = "unknown_qid",
        target_answer: Any = None,
        question_type: str = "other"
    ) -> WorkflowResult:
        """Run the VQA workflow - must be implemented by subclasses."""
        self._set_status(WorkflowStatus.RUNNING)
        
        # Validate inputs
        if error := self.validate_inputs(image_path, question, question_id, target_answer):
            return self.create_error_result(
                question_id=question_id,
                image_path=image_path,
                question=question,
                target_answer=target_answer,
                error_message=error,
                question_type=question_type
            )
            
        raise NotImplementedError("Subclasses must implement run_workflow") 