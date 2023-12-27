import os #help save image different folder
import argparse #help handle command line parameters
import yaml #read file yaml
import time #count time finish run program

import cv2 #handle image
# import matplotlib
# import matplotlib.pyplot as plt

from detector.detect import detect_face 
from detector.smooth import smooth_face
from utils.image import (load_image, 
                         save_image, 
                         save_steps, 
                         check_img_size,
                         get_height_and_width,
                         process_image,
                         check_if_adding_bboxes)
from utils.video import (split_video,
                         process_video)
from utils.types import (is_image,
                         is_video,
                         is_directory)


def parse_args():
    parser = argparse.ArgumentParser(description='Facial detection and \
                                     smoothing using OpenCV.')
    #add arguments 
    parser.add_argument('--input', 
                        type=str, 
                        help='Input file or folder',
                        default='data/images/hillary_clinton.jpg')
    parser.add_argument('--output', 
                        type=str, 
                        help='Output file or folder',
                        default='data/output')
    parser.add_argument('--show-detections', 
                        action='store_true',
                        help='Displays bounding boxes during inference.')
    parser.add_argument('--save-steps', 
                        action='store_true',
                        help='Saves each step of the image.')
    args = parser.parse_args() #parse parameters from command line
    return args


def load_configs():
    #read file yaml return dict 
    script_directory = os.path.dirname(os.path.abspath(__file__))
    #os.path.abspath: convert relative path to path
    #os.path.dirname: provide folder contain file
    config_path = os.path.join(script_directory, 'configs', 'configs.yaml')
    #os.path.join: connect file name
    with open(config_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def main(args):
    """Puts it all together."""
    # Start measure time
    tic = time.perf_counter()
    # Project config
    cfg = load_configs()
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], 
                                        cfg['net']['cfg_file'])
    # Input and load image
    input_file = args.input #get filename
    img_steps = None #init step handle image
    try:
        # If file is a video file
        if is_video(input_file):
            # Process video
            process_video(input_file, args, cfg, net)

        # If file is a image file
        elif is_image(input_file):
            # Load image
            input_img = load_image(input_file)
            # Process image 
            img_steps = process_image(input_img, cfg, net)
            # Save final image to specified output filename
            out_filename = os.path.join(args.output, cfg['image']['output'])
            # Check for --show-detections flag
            output_img = check_if_adding_bboxes(args, img_steps)
            # Save image
            img_saved = save_image(out_filename, output_img)

        # If input_file is a dir
        elif is_directory(input_file):
            # For each file in the dir
            for file in os.listdir(input_file):
                # Join input dir and file name
                file = os.path.join(input_file, file)
                # If file is a video file
                if is_video(file):
                    # Process video
                    process_video(file, args, cfg, net)
                # If file is a image file    
                if is_image(file):
                    # Load image
                    input_img = load_image(file)
                    # Process image
                    img_steps = process_image(input_img, cfg, net)
                    # Save final image to specified output filename
                    out_filename = os.path.join(args.output, cfg['image']['output'])
                     # Check for --show-detections flag
                    output_img = check_if_adding_bboxes(args, img_steps)
                    # Save image
                    img_saved = save_image(out_filename, output_img)

    except ValueError:
        print('Input must be a valid image, video, or directory.')
    
    # Save processing steps
    if args.save_steps and img_steps is not None:
        # Set image output height
        output_height = cfg['image']['img_steps_height']
        # Set output filename
        steps_filename = os.path.join(args.output, cfg['image']['output_steps'])
        # Save file
        save_steps(steps_filename, img_steps, output_height)

    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")


if __name__ == '__main__':
    args = parse_args()
    main(args)