# debate_flow.py
import json
import time
import os
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple
import re

from ..config_loader import app_config
from ..image_utils import process_image_for_vlm_agent
from ..evaluation import perform_direct_accuracy_check
from ..agents.debate_agents import VQADebateSolverAgent
from ..vllm_clients import llm_config_vlm
from .base_workflow import BaseVQAWorkflow, WorkflowResult, WorkflowStatus

def parse_solver_output(raw_output: str) -> Dict[str, str | None]:
    """Parse solver output to extract answer and reasoning."""
    answer = None
    reasoning = None
    raw_output_cleaned = raw_output.strip()
    
    # Try to extract answer
    if raw_output_cleaned.lower().startswith("answer:"):
        answer = raw_output_cleaned.split(":",1)[-1].strip()
    elif "\n" in raw_output_cleaned:
        potential_answer = raw_output_cleaned.split("\n")[0].strip()
        if potential_answer.lower().startswith("answer:"):
            answer = potential_answer.split(":",1)[-1].strip()
        else:
            answer = potential_answer
    else:
        answer = raw_output_cleaned
        
    # Try to extract reasoning
    if "\n" in raw_output_cleaned:
        reasoning_lines = raw_output_cleaned.split("\n")[1:]
        reasoning = "\n".join(line.strip() for line in reasoning_lines if line.strip())
        
    return {"answer": answer, "reasoning": reasoning}

class DebateVQAWorkflow(BaseVQAWorkflow):
    """VQA workflow that uses multiple agents debating to reach consensus."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        f1_threshold: float = 0.5,
        num_solvers: int = 3,
        max_debate_rounds: int = 1
    ):
        super().__init__(config, logger, f1_threshold)
        self.num_solvers = num_solvers
        self.max_debate_rounds = max_debate_rounds
        
    async def _initialize_solvers(self, question_id: str) -> Tuple[List[VQADebateSolverAgent], Optional[str]]:
        """Initialize solver agents."""
        solver_agents: List[VQADebateSolverAgent] = []
        try:
            if not llm_config_vlm:
                raise ValueError("Solver LLM configuration is invalid or missing.")
                
            for i in range(self.num_solvers):
                solver_name = f"VQASolver_{i+1}_QID_{question_id}"
                agent = VQADebateSolverAgent(name=solver_name, llm_config=llm_config_vlm)
                solver_agents.append(agent)
                
            if not solver_agents:
                return [], "No solver agents were initialized"
                
            self.logger.info(f"{len(solver_agents)} VQADebateSolverAgents initialized for QID {question_id}.")
            return solver_agents, None
            
        except Exception as e:
            error_msg = self._log_error(e, "Error initializing solver agents")
            return [], error_msg
            
    async def _get_solver_response(
        self,
        solver_agent: VQADebateSolverAgent,
        processed_url: str,
        question: str,
        current_round: int,
        max_rounds: int,
        previous_round_outputs: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Get response from a solver agent."""
        try:
            solver_agent.reset()
            
            system_prompt = solver_agent.render_system_message(
                question=question,
                current_round=current_round,
                max_rounds=max_rounds,
                neighbor_responses=previous_round_outputs
            )
            
            solver_input = [
                {"type": "image_url", "image_url": {"url": processed_url}},
                {"type": "text", "text": question}
            ]
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": solver_input}
            ]

            self.logger.debug(f" {solver_agent.name} (Round {current_round}) System Prompt Preview: {system_prompt[:400]}...")
            self.logger.info(f" {solver_agent.name} (Round {current_round}): Calling VLM...")
            
            chat_res = await solver_agent.a_generate_reply(messages=messages, sender=None)
            raw_output = chat_res.get("content", "") if isinstance(chat_res, dict) else str(chat_res)
            
            if not raw_output:
                self.logger.warning(f" {solver_agent.name} (Round {current_round}) returned empty content.")
                return {
                    "solver_name": solver_agent.name,
                    "round_num": current_round,
                    "answer": "[SolverEmptyOutput]",
                    "reasoning": None,
                    "confidence": 0.0,
                    "raw_output": ""
                }, None
                
            parsed_output = parse_solver_output(raw_output)
            if not parsed_output.get("answer"):
                parsed_output["answer"] = "[SolverEmptyAnswerAfterParsing]"
                
            return {
                "solver_name": solver_agent.name,
                "round_num": current_round,
                "answer": parsed_output["answer"],
                "reasoning": parsed_output.get("reasoning"),
                "confidence": 0.95 if parsed_output["answer"] else 0.0,
                "raw_output": raw_output
            }, None
            
        except Exception as e:
            error_msg = self._log_error(e, f"Error in {solver_agent.name} (Round {current_round})")
            return {
                "solver_name": solver_agent.name,
                "round_num": current_round,
                "answer": f"[SolverError_R{current_round}]",
                "reasoning": None,
                "confidence": 0.0,
                "raw_output": ""
            }, error_msg
            
    def _get_majority_answer(self, solver_outputs: List[Dict[str, Any]]) -> Tuple[Optional[str], float, int]:
        """Get majority answer from solver outputs."""
        answers_for_vote = [
            sol_out.get("answer", "") for sol_out in solver_outputs
            if sol_out.get("answer") and isinstance(sol_out.get("answer"), str) and \
               not str(sol_out.get("answer")).startswith(("[SolverError", "[SolverEmpty", "[SolverEmptyAnswerAfterParsing"))
        ]
        
        if not answers_for_vote:
            return None, 0.0, 0
            
        vote_counts = Counter(answers_for_vote)
        if not vote_counts:
            return None, 0.0, 0
            
        majority_answer, majority_count = vote_counts.most_common(1)[0]
        
        confidences_for_majority = [
            s_out.get("confidence", 0.0) for s_out in solver_outputs
            if s_out.get("answer") == majority_answer and isinstance(s_out.get("confidence"), (float, int))
        ]
        
        final_confidence = sum(confidences_for_majority) / len(confidences_for_majority) if confidences_for_majority else 0.0
        
        return majority_answer, final_confidence, majority_count
        
    async def run_workflow(
        self,
        image_path: str,
        question: str,
        question_id: str = "unknown_qid",
        target_answer: Any = None,
        question_type: str = "other"
    ) -> WorkflowResult:
        start_time = time.time()
        
        # Process image
        processed_url, error = self.process_image(image_path)
        if error:
            return self.create_error_result(
                question_id=question_id,
                image_path=image_path,
                question=question,
                target_answer=target_answer,
                error_message=error,
                question_type=question_type
            )
            
        # Initialize solver agents
        solver_agents, error = await self._initialize_solvers(question_id)
        if error:
            return self.create_error_result(
                question_id=question_id,
                image_path=image_path,
                question=question,
                target_answer=target_answer,
                error_message=error,
                question_type=question_type
            )

        all_rounds_outputs: List[List[Dict[str, Any]]] = []
        previous_round_outputs: List[Dict[str, Any]] = []

        # Run debate rounds
        for current_round_idx in range(self.max_debate_rounds):
            current_round = current_round_idx + 1
            self.logger.info(f"\n--- Debate Round {current_round} (QID: {question_id}) ---")
            
            current_round_outputs: List[Dict[str, Any]] = []
            
            # Get responses from all solvers
            for solver_agent in solver_agents:
                output, error = await self._get_solver_response(
                    solver_agent,
                    processed_url,
                    question,
                    current_round,
                    self.max_debate_rounds,
                    previous_round_outputs
                )
                current_round_outputs.append(output)
                
                if error:
                    self.logger.error(f"Error from {solver_agent.name}: {error}")
            
            all_rounds_outputs.append(current_round_outputs)
            previous_round_outputs = current_round_outputs
            
            # Check if all solvers failed
            if not any(out.get('answer') and not str(out.get('answer', '')).startswith(("[SolverError", "[SolverEmpty", "[SolverEmptyAnswerAfterParsing")) for out in current_round_outputs):
                return self.create_error_result(
                    question_id=question_id,
                    image_path=image_path,
                    question=question,
                    target_answer=target_answer,
                    error_message=f"AllSolversFailed_R{current_round}",
                    question_type=question_type
                )

        # Get final result from last round
        if all_rounds_outputs and all_rounds_outputs[-1]:
            final_outputs = all_rounds_outputs[-1]
            majority_answer, final_confidence, majority_count = self._get_majority_answer(final_outputs)
            
            if majority_answer:
                return self.create_success_result(
                    question_id=question_id,
                    image_path=image_path,
                    question=question,
                    target_answer=target_answer,
                    final_answer=majority_answer,
                    confidence_score=final_confidence,
                    processing_time=time.time() - start_time,
                    question_type=question_type,
                    additional_metadata={
                        "debate_rounds": self.max_debate_rounds,
                        "num_solvers": self.num_solvers,
                        "majority_vote_count": majority_count,
                        "all_rounds_outputs": all_rounds_outputs
                    }
                )
                
            return self.create_error_result(
                question_id=question_id,
                image_path=image_path,
                question=question,
                target_answer=target_answer,
                error_message="No valid answers from solvers for majority vote",
                question_type=question_type
            )
            
        return self.create_error_result(
            question_id=question_id,
            image_path=image_path,
            question=question,
            target_answer=target_answer,
            error_message="No debate outputs from final round or no rounds completed",
            question_type=question_type
        )

# For backward compatibility
async def run_simplified_debate_vqa_pipeline(
    image_path: str,
    question: str,
    question_id: str = "unknown_qid",
    target_answer: Any = None,
    question_type: str = "other",
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Legacy function that uses DebateVQAWorkflow internally."""
    workflow = DebateVQAWorkflow(
        config=app_config,
        logger=logger_instance
    )
    result = await workflow.run_workflow(
        image_path=image_path,
        question=question,
        question_id=question_id,
        target_answer=target_answer,
        question_type=question_type
    )
    return result.to_dict()