import argparse
from ultralytics import YOLO
from thop import profile
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model and extract evaluation metrics.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data configuration file.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the model.
    model = YOLO(args.model_path)

    # Evaluate the model.
    results = model.val(
        data=args.data_path,
        imgsz=640,
        name='bestn_eval'
    )

    # Extract and print evaluation metrics.
    metrics = results.results_dict

    precision = metrics['metrics/precision(B)']
    recall = metrics['metrics/recall(B)']
    map50 = metrics['metrics/mAP50(B)']
    map = metrics['metrics/mAP50-95(B)']

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"mAP@0.50: {map50:.4f}")
    print(f"mAP@0.50:0.95: {map:.4f}")

if __name__ == "__main__":
    main()