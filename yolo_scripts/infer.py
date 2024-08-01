import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Infer YOLOv8 on a directory of images and save the outputs.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model.')
    parser.add_argument('--source-dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the model.
    model = YOLO(args.model_path)

    # Perform inference on the directory of images.
    results = model.predict(
        source=args.source_dir,
        imgsz=args.imgsz,
        save=True,
    )


if __name__ == "__main__":
    main()