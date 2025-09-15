"""
Fire Detection using YOLOv8

This script demonstrates how to use YOLOv8 for detecting fire in images.
It loads a pre-trained YOLOv8 model, takes an input image, and performs fire detection.
Ensure that the YOLOv8 model is trained or fine-tuned for fire detection before using.

Usage:
    python main.py --image <path_to_image>
"""

import argparse
import cv2
from ultralytics import YOLO

def detect_fire(image_path):
    """
    Detect fire in an input image using a YOLOv8 model.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    # Load the YOLOv8 model (replace with your fire detection model if needed)
    model = YOLO("yolov8n.pt")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open image from {image_path}")
        return

    # Perform fire detection
    results = model(image)

    # Display detection results
    print(f"Detection results for image: {image_path}")
    results.show()  # Opens a window with detection results

    # Save results image
    results.save(save_dir="runs/detect/fire_detection")

def main():
    parser = argparse.ArgumentParser(description="Fire detection using YOLOv8.")
    parser.add_argument("--image", required=True, help="Path to the input image for fire detection.")
    args = parser.parse_args()
    detect_fire(args.image)

if __name__ == "__main__":
    main()