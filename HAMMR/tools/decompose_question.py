from typing import List, Annotated
from langchain_google_vertexai import VertexAI
import os, json
from google.cloud import aiplatform
from autogen_core.tools import FunctionTool
from pathlib import Path

# Initialize Vertex AI
CREDENTIALS_PATH = Path(__file__).parent.parent / "credentials" / "bwa-agents-54872988b93e.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)
aiplatform.init(project='bwa-agents', location='us-central1')


def decompose_question(
    question: Annotated[str, "Complex question to decompose"]
) -> List[Annotated[str, "Simpler sub-question"]]:
    """
    Tool: DecomposeQuestion – Splits a complex question into two simpler sub-questions.
    """
    llm = VertexAI(model_name="gemini-2.0-flash-lite-001")
    prompt = (
            f"Decompose the following complex question into exactly two simpler, self-contained sub-questions. "
            f"The sub-questions, when answered sequentially, should help answer the original complex question. "
            f"Return the two sub-questions as a JSON array of strings.\n\n"
            f"Complex Question: \"What is the Köppen climate classification for the city where this mosque is located?\"\n"
            f"Output: [\"In which city is this mosque located?\", \"What is the Köppen climate classification for this city?\"]\n\n"
            f"Complex Question: \"Who is the CEO of the company that developed the game featuring a plumber who jumps on turtles?\"\n"
            f"Output: [\"Which company developed the game featuring a plumber who jumps on turtles?\", \"Who is the CEO of that company?\"]\n\n"
            f"Complex Question: \"{question}\"\n"
            f"Output:"
        )
    resp = llm.invoke(prompt)
    try:
        # Expecting JSON array
        result = json.loads(resp)
        if isinstance(result, list) and len(result) == 2:
            return result  # type: ignore
    except json.JSONDecodeError:
        pass
    # Fallback: take first two non-empty lines
    lines = [ln.strip(' -"') for ln in resp.splitlines() if ln.strip()]
    return lines[:2]

# Register tool
decompose_question_tool = FunctionTool(
    func=decompose_question,
    name="decompose_question",
    description=(
        "Tool: DecomposeQuestion – Split a complex question into two simpler sub-questions. "
        "Input: question (str). Output: list of two questions."
    )
)