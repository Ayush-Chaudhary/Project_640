import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt    
import torch
import argparse

from functions.detection_utils import get_avg_flow
from functions.get_models import get_yolo
from functions.visualize import plot_boxes, draw_flow_vectors
from functions.detection_utils import get_avg_flow
from functions.flow_viz import flow_to_image
from functions.video_utils import save_vid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation Main')
    parser.add_argument('--img_path', type=str, help='Path to input image folder', default='FlowFormerPlusPlus-main/datasets/Sintel/test')
    parser.add_argument('--flow_path', type=str, help='Path to optical flow folder', default='FlowFormerPlusPlus-main/sintel_submission_multi8_768')
    parser.add_argument('--vid_name', type=str, help='Name of the test video folder', default='test_video')
    parser.add_argument('--model', type=str, help='Path to the model', default='inputs/yolov8n-seg.pt')

    args = parser.parse_args()

    # Get the paths to the images and optical flow files
    path_imgs = f'{args.img_path}/{args.vid_name}'
    img_path = [f'{path_imgs}/{img}' for img in os.listdir(path_imgs) if img.endswith('.png')][:-1]
    img_path.sort()

    path_flow = f'{args.flow_path}/{args.vid_name}'
    flow_path = [f'{path_flow}/{img}' for img in os.listdir(path_flow) if img.endswith('.flo')]
    flow_path.sort()

    # Load the YOLO model
    yolo_model, yolo_processor = get_yolo()        

    # Get the average flow vectors for the detected bounding boxes
    flow_boxes, masks, flows = get_avg_flow(flow_path, img_path, yolo_model, yolo_processor)

    # Draw the flow vectors on the images
    color_images, info = draw_flow_vectors(flow_boxes, img_path)

    # Save the video
    save_vid(f'results/{args.vid_name}combined.avi', color_images)
