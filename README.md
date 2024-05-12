# 2D Motion Analysis in Videos using Optical Flow Techniques

This file contains instructions to run the code.

## Models
Download the pretrained FlowFormer++ [model](https://drive.google.com/drive/folders/1fyPZvcH4SuNCgnBvIJB2PktT5IN9PYPI?usp=sharing). The default path of the model for evaluation is:
```Shell
├── checkpoints
    ├── sintel.pth
```

## Evaluation

Store an example video file in the videos folder (preferably landscape) or use `test_video.mp4`.
```shell
├──videos
    ├──test_video.mp4
```

Run the `video_to_images.py` file to save the video as frames in `FlowFormerPlusPlus-main\datasets\Sintel\test`. Save any more frame that you want to test the alhorithm on in a new folder in the `FlowFormerPlusPlus-main\datasets\Sintel\test\personal` directory

```shell
python video_to_images.py
```

Run the `evaluate_FlowFormer_tile.py` file to get the flow values corresponding the frames.

```shell
cd ./FlowFormerPlusPlus-main
python evaluate_FlowFormerPlusPlus_tile.py --model ./checkpoints/sintel.pth --eval sintel_submission
cd ..
```

Run the `segmentation_main.py` to get the annotated videos after running YOLOv8 and postprocessing functions

```shell
python segmentation_main.py
```

Go through the [results](https://drive.google.com/drive/folders/1dXfLus2LjWmG3Mal3VqVbY2NNGlVuBEm?usp=sharing) on a few more test videos.