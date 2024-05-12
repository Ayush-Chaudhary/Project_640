from transformers import YolosImageProcessor, YolosForObjectDetection
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from ultralytics import YOLO


def get_yolo():
    """
    Load the Yolos model and processor
    Returns:
        model (YolosForObjectDetection): Yolos model
        processor (YolosImageProcessor): Yolos processor
    """

    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    return model, processor
    