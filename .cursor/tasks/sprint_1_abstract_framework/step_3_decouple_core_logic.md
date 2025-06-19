# T.sprint_1/step_3_decouple_core_logic.md
-   **Objective:** Remove all data loading and preprocessing calls from the core workflow scripts.
-   **Checklist:**
    -   [x] **Refactor `step_eval_mutil_process_api_zoo.py`:**
        -   Remove the function `vlm_assist_query` and its dependencies (`vis_processors`, `txt_processors`).
        -   Modify `get_assist_query_single` to remove all file I/O (e.g., `Image.open()`) and dataset-dependent path construction.
        -   The function should now assume that all necessary data (like pre-generated captions) is already present in the input `res` dictionary. The `assist_query_method` logic should be simplified to only support the `llm` path.
    -   [x] **Refactor `test_for_integration_rights_alloction.py`:**
        -   Remove all image loading and visual processing logic. The function `integration` should no longer receive `model`, `vis_processors`, or `txt_processors` related to a local VLM.
        -   Modify the function to work with abstract data, assuming the `vqa_answer` function is a placeholder for an external API call that doesn't require local image processing. 