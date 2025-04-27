import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
import torch
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def get_sam_model(device: Optional[str] = None):
    model_type = "vit_t"

    # mobile_sam_dir = os.path.dirname(inspect.getfile(mobile_sam))
    save_path = os.path.expanduser("~/.cache/mobile_sam")
    # make sure the directory exists
    os.makedirs(save_path, exist_ok=True)
    sam_checkpoint = os.path.join(save_path, "mobile_sam.pt")

    if not os.path.exists(sam_checkpoint):
        # download the checkpoint
        weights_url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
        print(f"Downloading MobileSAM checkpoint from {weights_url}")
        import urllib.request

        urllib.request.urlretrieve(weights_url, sam_checkpoint)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    mobile_sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam_model.to(device=device)
    mobile_sam_model.eval()
    return mobile_sam_model


class SAM:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = get_sam_model(device=device)
        self._device = device
        self._sam_predictor = SamPredictor(self._model)
        self._sam_mask_generator = SamAutomaticMaskGenerator(self._model)

    def predict_with_prompt(
        self, image: np.ndarray, prompt: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._sam_predictor.set_image(image)
        masks, iou_predictions, _ = self._sam_predictor.predict(prompt)
        return masks, iou_predictions

    def predict_all(self, image: np.ndarray):
        mask_generator = SamAutomaticMaskGenerator(self._model)
        return mask_generator.generate(image=image)


def detections_to_sv_dectections(
    detections: List[Dict[str, Any]]
) -> sv.detection.core.Detections:
    """Convert detections from SAM to Supervision format.

    Args: detections: List of detections from SAM
        detections has the keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'

    Returns: Detections in Supervision format
        Detections is a class with arguments: xyxy, mask, confidence
    """
    # convert to Supervision format
    xyxys = []
    masks = []
    confidences = []
    for detection in detections:
        xyxy = detection["bbox"]
        mask = detection["segmentation"]
        confidence = detection["predicted_iou"]
        xyxys.append(xyxy)
        masks.append(mask)
        confidences.append(confidence)
    sv_detection = sv.detection.core.Detections(
        xyxy=np.array(xyxys), mask=np.array(masks), confidence=np.array(confidences)
    )
    return sv_detection


def main():
    import cv2

    image = cv2.imread("cameras/test_example3/image.png")
    assert image is not None
    model = SAM(device="cpu")
    image_numpy_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = model.predict_all(
        image=image_numpy_rgb,
    )
    detections = detections_to_sv_dectections(detections)

    # visualize detections
    # box_annotator = sv.BoxAnnotator()
    # annotated_image = box_annotator.annotate(scene=image, detections=detections)
    mask_annotator = sv.MaskAnnotator()
    annotated_image = mask_annotator.annotate(scene=image, detections=detections)
    cv2.imshow("annotated_image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    main()
