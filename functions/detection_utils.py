import torch
from PIL import Image
import numpy as np
from functions.flow_utils import read_flow

def check_overlap(box1, box2):
    """
    Calculate the percentage of overlap between two bounding boxes.
    Args:
        box1 (list): List containing the coordinates of the first bounding box in the format [x1, y1, x2, y2]
        box2 (list): List containing the coordinates of the second bounding box in the format [x1, y1, x2, y2]

    Returns:
        overlap_area / total_area (float): Percentage of overlap between the two bounding boxes
    """

    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # find percentage of overlap
    x_overlap = max(0, min(x2, x4) - max(x1, x3))
    y_overlap = max(0, min(y2, y4) - max(y1, y3))
    overlap_area = x_overlap * y_overlap
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    total_area = box1_area + box2_area - overlap_area
    return overlap_area / total_area

def non_max_supress(boxes, threshold=0.5):
    """
    Perform non-maximum suppression on the detected bounding boxes.
    Args:
        boxes (dict): Dictionary containing the detected bounding boxes and their scores
        threshold (float): Threshold for overlap between bounding boxes

    Returns:
        boxes (dict): Dictionary containing the filtered bounding boxes
    """
    
    for item in boxes.items():
        # sort the boxes by score
        item[1].sort(key=lambda x: x[1], reverse=True)
        i = 0
        if len(item[1]) > 1:
            while i < len(item[1])-1:
                j = i + 1
                if j < len(item[1]):
                    while j < len(item[1]):
                        if check_overlap(item[1][i][0], item[1][j][0]) > threshold: item[1].pop(j)
                        else: j += 1
                i += 1
    return boxes

def remove_score(boxes):
    """
    Remove the score from the detected bounding boxes.
    Args:
        boxes (dict): Dictionary containing the detected bounding boxes and their scores

    Returns:
        boxes (dict): Dictionary containing the detected bounding boxes without scores
    """
    
    for item in boxes.items():
        for i in range(len(item[1])):
            item[1][i] = item[1][i][0]
    return boxes

# return the detected box in single output image of yolo 
def get_boxes_detection(model, processor, img, threshold=0.35):
    """
    Get the bounding boxes of the detected objects in an image.
    Args:
        model (Yolov5): Yolov5 model
        processor (Yolov5Processor): Yolov5 processor
        img (str): Path to the image
        threshold (float): Threshold for object detection confidence

    Returns:
        boxes (dict): Dictionary containing the detected bounding boxes and their scores
    """
    
    boxes = {}
    img = Image.open(img)
    # convert image to grayscale
    # img = img.convert('L')
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    
    target_sizes = torch.tensor([img.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        key = model.config.id2label[label.item()]
        value = [box, round(score.item(),2)]
        if key not in boxes: boxes[key] = [value]
        else: boxes[key].append(value)
        boxes = non_max_supress(boxes, threshold)
    return boxes

def average_pixel_value(image, mask):
    """
    Calculate the average pixel value within a masked region of an image.
    Args:
        image (PIL.Image): Input image
        mask (numpy.ndarray): Mask of the same size as the image

    Returns:
        average_value (numpy.ndarray): Average pixel value within the masked region along the RGB channels
    """
    
    # Convert the image to a numpy array
    image_array = np.array(image)

    # Apply the mask to the image
    masked_image = image_array * mask[:, :, np.newaxis]

    # Calculate the average pixel value within the masked region along the RGB channels of the non-zero pixels
    total_value = np.sum(masked_image, axis=(0, 1))
    total_area = np.sum(mask)
    average_value = total_value / total_area if total_area != 0 else np.zeros(2)

    return average_value

def generate_rectangular_mask(image, box):
    """
    Generate a rectangular mask based on the bounding box coordinates.
    Args:
        image (numpy.ndarray): Input image
        box (list): List containing the coordinates of the bounding box in the format [x1, y1, x2, y2]

    Returns:
        mask (numpy.ndarray): Mask of the same size as the image with the rectangular region filled with ones
    """
    
    # empty mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    box = [int(i) for i in box]
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 1
    return mask

def get_segmentation(model, feature_extractor, img):
    """
    Get the segmentation mask of an image.
    Args:
        model (ViT): Vision Transformer model
        feature_extractor (Transform): Feature extractor
        img (str): Path to the image

    Returns:
        logits (torch.Tensor): Logits of the segmentation mask
    """

    img = Image.open(img)
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    return logits

def get_avg_flow(flow_path, img_path, yolo_model, yolo_processor):
    """
    Get the average flow vectors for the detected bounding boxes in the images.
    Args:
        flow_path (list): List of paths to the optical flow files
        img_path (list): List of paths to the images
        yolo_model (Yolov5): Yolov5 model
        yolo_processor (Yolov5Processor): Yolov5 processor

    Returns:
        boxes (list): List of dictionaries containing the bounding boxes and flow vectors
        masks (list): List of masks for the detected objects
        flows (list): List of optical flow images
    """

    masks, flows = [], []
    boxes = []
    union = []
    for i in range(len(flow_path)):
        box = get_boxes_detection(yolo_model, yolo_processor, img_path[i])
        flow = read_flow(flow_path[i])

        temp_mask = []
        for item in box.items():
            for b in item[1]:
                mask = generate_rectangular_mask(flow, b[0])
                temp_mask.append(mask)
        # apply union of all masks
        union_mask = np.zeros(flow.shape[:2], dtype=np.uint8)
        for mask in temp_mask:
            union_mask = np.logical_or(union_mask, mask)

        for item in box.items():
            for b in item[1]:
                mask = generate_rectangular_mask(flow, b[0])
                vectors = list(average_pixel_value(flow, mask)-average_pixel_value(flow, 1-union_mask))
                b.append(vectors)
                b.append(np.arctan2(vectors[1], vectors[0]))

        flows.append(flow)
        masks.append(union_mask)
        boxes.append(box)
        union.append(union_mask)
    return boxes, masks, flows, union
                
                
