import argparse
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX or TRT format.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLO model.')
    parser.add_argument('--format', type=str, choices=['onnx', 'trt'], required=True, help='Format to convert the model to: onnx or trt.')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'], help='Precision for the conversion: fp16 or int8.')
    parser.add_argument('--calibration_data_path', type=str, help='Path to the calibration data for INT8 conversion.')
    return parser.parse_args()

def convert_model(model_path, format, precision, calibration_data_path):
    # Load the YOLO model
    model = YOLO(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    
    if format == 'onnx':
        model.export(format='onnx', device=device)

    elif format == 'trt':
        if precision == 'fp32':
            model.export(format='engine', dynamic=True, device=device, workspace=2)
        elif precision == 'fp16':
            model.export(format='engine', dynamic=True, half=True, device=device, workspace=2)
        elif precision == 'int8':
            model.export(format='engine', dynamic=True, int8=True, device=device, workspace=2, data=calibration_data_path)
    


if __name__ == "__main__":
    args = parse_args()
    convert_model(args.model_path, args.format, args.precision, args.calibration_data_path)