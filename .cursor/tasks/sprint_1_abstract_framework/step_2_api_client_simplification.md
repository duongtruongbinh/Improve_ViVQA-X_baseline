# T.sprint_1/step_2_api_client_simplification.md
-   **Objective:** Refactor `api_tools/request_api_zoo.py` into a single-purpose OpenAI client.
-   **Checklist:**
    -   [x] Remove all logic and dependencies for Zhipu and SiliconFlow.
    -   [x] Hardcode the `model_name` check to only accept GPT variants.
    -   [x] Replace dynamic key selection with a single placeholder `YOUR_API_KEY_HERE`. 