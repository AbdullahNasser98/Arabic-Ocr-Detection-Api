import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model.")
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the YOLO model.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data configuration file.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training.')
    parser.add_argument('--batch', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--name', type=str, default='yolov8m', help='Name of the training run.')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for data loading.')
    return parser.parse_args()


def main():
    args = parse_args()
    
    model = YOLO('yolov8m.pt')

    model.train(
        data=args.data_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        workers=args.workers
    )

if __name__ == "__main__":
    main()