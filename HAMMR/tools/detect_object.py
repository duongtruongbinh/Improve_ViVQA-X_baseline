from typing import Any, List, Annotated
from PIL import Image
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from autogen_core.tools import FunctionTool

# Load the OWLv2 processor and model once at module load
_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

def detect_object(
    image: Any,
    class_name: Annotated[str, "Target class to detect, e.g. 'cat', 'dog'"]
) -> List[Annotated[List[float], "Bounding box [x, y, width, height]"]]:
    """
    Tool: DetectObject – Returns bounding boxes for all instances of the given class name,
    in [x, y, width, height] format using OwlViTv2.
    """
    # Prepare the grounding query
    text_labels = [[f"a photo of a {class_name}"]]
    inputs = _processor(text=text_labels, images=image, return_tensors="pt")
    outputs = _model(**inputs)

    # Map normalized predictions back to pixel coordinates
    target_sizes = torch.tensor([(image.height, image.width)])
    results = _processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.1,
        text_labels=text_labels
    )

    # Extract Pascal VOC boxes and convert to [x, y, width, height]
    result = results[0]
    pascal_boxes = result["boxes"]
    xywh_boxes: List[List[float]] = []
    for box in pascal_boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        xywh_boxes.append([
            xmin,
            ymin,
            xmax - xmin,
            ymax - ymin
        ])

    return xywh_boxes

detect_object_tool = FunctionTool(
    func=detect_object,
    name="detect_object",
    description="Tool: DetectObject – Detect all instances of a specified class in the image and return their bounding boxes [x, y, width, height]."
)
