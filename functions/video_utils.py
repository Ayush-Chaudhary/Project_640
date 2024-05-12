import cv2
import os

def load_vid(path, target_fps = 10, target_res = (1024, 436)):
    """
    Load a video and return a list of frames
    Args:
        path (str): Path to the video file
    Returns:
        frames (list): List of frames in RGB format
    """
    cap = cv2.VideoCapture(path)

    # Check if the video is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video.")

    # empty list to store the frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize the frame to 1024x436
        frame = cv2.resize(frame, target_res)

        # store the frame in a list
        frames.append(frame)
    print('intial frames:', len(frames))
    print('FPS:', cap.get(cv2.CAP_PROP_FPS))
    # skip frames to get the desired FPS
    frames = frames[::target_fps]
    print('final frames:', len(frames))
    return frames

def save_vid_as_frames(path, frames):
    """
    Save a list of images as frames
    Args:
        path (str): Path to the directory where the frames will be saved
        frames (list): List of images to be saved as frames (each image should be in RGB format)
    """

    # check if the directory exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)
    # save the frames as images using OpenCV
    zeros = len(str(len(frames)))
    for i, frame in enumerate(frames):
        cv2.imwrite(f'{path}/frame_{i:0{zeros}d}.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

def save_vid(video_name, color):
    """
    Save a list of images as a video
    Args:
        video_name (str): Path of the output video along with the name
        color (list): List of images to be saved as a video (each image should be in RGB format)

    """

    # Read the first image to get frame dimensions
    frame = color[0]
    height, width, _ = frame.shape

    # Define the video writer with desired codec (XVID) and FPS (10)
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width,height))

    # Loop through each image and write it to the video
    for image in color:
        # convert the image to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Write the frame to the video
        video.write(image)

    # Release resources
    cv2.destroyAllWindows()
    video.release()

    print(f"Video created: {video_name}")