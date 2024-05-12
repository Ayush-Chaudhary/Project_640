from functions.video_utils import load_vid, save_vid_as_frames
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from a video')
    parser.add_argument('--video', type=str, help='path to the video folder', default='videos')
    parser.add_argument('--output', type=str, help='path to the output directory', default = 'FlowFormerPlusPlus-main/datasets/Sintel/test/kittiroad')
    parser.add_argument('--vid_name', type=str, help='name of the video file')
    parser.add_argument('--frame_skips', type=int, help='number of frames to skip', default=1)
    parser.add_argument('--extension', type=str, help='extension of the video file', default='avi')
    args = parser.parse_args()

    # Load the video
    vid = load_vid(f'{args.video}/{args.vid_name}.{args.extension}', args.frame_skips)#, (1242, 432))

    # Save the video as frames
    save_vid_as_frames(f'{args.output}/{args.vid_name}', vid)
