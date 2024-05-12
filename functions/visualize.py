import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.detection_utils import get_boxes_detection, get_segmentation
from PIL import Image, ImageDraw

def draw_arrow(draw, x_start, y_start, x_end, y_end, head_width=10, head_length=10, fill="black", width=1):
    """
    Draw an arrow on an image.
    Args:
        draw (ImageDraw): ImageDraw object
        x_start (int): x-coordinate of the start point
        y_start (int): y-coordinate of the start point
        x_end (int): x-coordinate of the end point
        y_end (int): y-coordinate of the end point
        head_width (int): Width of the arrow head
        head_length (int): Length of the arrow head
        fill (str): Color of the arrow
        width (int): Width of the arrow shaft
    """
    
    # Draw arrow shaft
    draw.line((x_start, y_start, x_end, y_end), fill=fill, width=width)

    # Calculate head coordinates
    angle = np.arctan2(y_end - y_start, x_end - x_start)
    x_head1 = x_end - head_length * np.cos(angle) - head_width * np.sin(angle)
    y_head1 = y_end - head_length * np.sin(angle) + head_width * np.cos(angle)
    x_head2 = x_end - head_length * np.cos(angle) + head_width * np.sin(angle)
    y_head2 = y_end - head_length * np.sin(angle) - head_width * np.cos(angle)

    # Draw arrow head polygon
    head_points = [(x_end, y_end), (x_head1, y_head1), (x_head2, y_head2)]
    draw.polygon(head_points, fill=fill)


def draw_flow_vectors(flow_boxes, img_path):
    """
    Draw flow vectors on images.
    Args:
        flow_boxes (list): List of dictionaries containing flow vectors and bounding boxes
        img_path (list): List of paths to the images
    Returns:
        images (list): List of images with flow vectors drawn
        info (list): List of bounding boxes and flow vectors
    """

    images = []
    info = []
    for i in range(len(flow_boxes)):
        temp_info = []
        img = Image.open(img_path[i])
        draw = ImageDraw.Draw(img)
        for key, items in flow_boxes[i].items():
            for item in items:
                x1, y1, x2, y2 = item[0]
                u, v = item[2]
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                magnitude = round((u**2 + v**2)**0.5, 2)
                if magnitude > 0.8:
                    draw_arrow(draw, x, y, x + int(u*20), y + int(v*20), head_width=15, head_length=15, fill='red', width=8)
                    formatted_w = f'{magnitude:.2f}'
                    draw.text((x, y), f'{formatted_w}', fill='red')
                    temp_info.append([item[0], item[2]])
        # convert to cv2 format
        info.append(temp_info)
        img = np.array(img)
        images.append(img)
    return images, info

def plot_boxes(img_path, yolo_model, yolo_processor):
    """
    Plot bounding boxes on images.
    Args:
        img_path (str): Path to the image
        yolo_model (Yolov5): Yolov5 model
        yolo_processor (Yolov5Processor): Yolov5 processor
    """

    img = cv2.imread(img_path)
    boxes = get_boxes_detection(yolo_model, yolo_processor, img_path)
    for label, box in boxes.items():
        for b in box:
            x1, y1, x2, y2 = b[0]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {b[1]}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def plot_flow(avg_flow):
    """
    Plot average flow motion.
    Args:
        avg_flow (list): List of average flow motion
    """

    # plot the average flow
    y1 = [flow[0] for flow in avg_flow]
    y2 = [flow[1] for flow in avg_flow]
    x = range(len(avg_flow))

    plt.plot(x, y1, label='horizontal motion')
    plt.plot(x, y2, label='vertical motion')
    plt.xlabel('frame')
    plt.ylabel('flow')
    plt.title('Average flow motion of the person')
    plt.legend()
    plt.show()


def combine_images(img_path, union_mask, flow_imgs, color_images):
    """
    Combine images for visualization.
    Args:
        img_path (list): List of paths to the images
        union_mask (list): List of union masks
        flow_imgs (list): List of optical flow images
        color_images (list): List of color images
    Returns:
        images (list): List of combined images
    """

    images = []
    for i in range(len(img_path)):
        img = Image.open(img_path[i])
        img = np.array(img)

        mask_array = np.array(union_mask[i])

        # Apply the mask on the image
        masked_img_array = np.where(mask_array[..., None] > 0, img, 0)

        img1 = np.concatenate((img, masked_img_array), axis=1)

        img2 = np.concatenate((flow_imgs[i], color_images[i]), axis=1)
        
        final_img = np.concatenate((img1, img2), axis=0)
        images.append(final_img)
    return images