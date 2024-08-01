# OCR and Object Detection API

This project provides a FastAPI-based web service for performing object detection and OCR on uploaded images of arabic text. It uses a fine-tuned version of YOLOv8 for object detection and EasyOCR for text extraction and SpellChecker for text correction.

## Features

- Detects titles in pages using a pre-trained YOLOv8 model.
- Extracts and corrects text from the detected regions using EasyOCR and SpellChecker.
- Exposes an API endpoint to upload images and get the processed results.

## Requirements

- Docker
- Python 3.10 or later

## YOLOv8

### Models performance:

- Fine-tunning on title class only would produce a faster model. However, since Title & Body Text classes are pretty similar I have decided to train the model on both of the classes for it to be able to detect the subtle differences.
- Below are the results on the desired Title class

| Model name               | Version     |   mAP50-95    | Precision    | Recall     | 
|:------------------------ |:------------|:--------------|:------------|:------------|
| Fine-tuned Yolov8m       | PyTorch     |  0.492        | 0.649        | 0.575      |                
| Fine-tuned Yolov8m       | ONNX        |  0.486        | 0.659        | 0.569      |               
| Fine-tuned Yolov8m       | FP16        |  0.485        | 0.651        | 0.574      |               
| Fine-tuned Yolov8m       | INT8        |  0.5987       | 0.9043       | 0.8570     | 

### Training

1. Download the dataset from [here](https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset?resource=download)

2. Run this cell to update the dataset into the format excpected by YOLOv8 model
   ```bash
   python yolo_scripts/update_annotations.py
   ```

3. To train the model run the training script form command line with the desired arguments. For example:
   ```bash
   python yolo_scripts/train.py --data-path <data.yaml path>
   ```

### Evaluating 

1. To evaluate the model run the evaluate script form command line with the desired arguments. For example:
    ```bash
    python eval.py --model-path <model path> --data-path <data.yaml path>
    ```

### Exporting 

1. Export to ONNX:
   ```bash
   python convert.py --model-path <.pt model path> --output-path <desired output path> --format onnx
   ```

2. Export to TRT:
    - For FP32:
        ```bash
        python convert.py --model-path <model path> --output-path <desired output path> --format trt --precision fp32
        ```
    - For FP16:
        ```bash
        python convert.py --model-path <model path> --output-path <desired output path> --format trt --precision fp16
        ```
    - For INT8:
        ```bash
        python convert.py --model-path <model path> --output-path <desired output path> --format trt --precision int8 --calibration_data_path <data path>
        ```

### Infering 

1. Make sure that the model is present in ./models

2. To infer the model on a directory of images. Run the inference script form command line with the desired arguments. For example:
    ```bash
    python infer.py --model-path <model path> --source-dir <images folder path>
    ```

## Using OCR
1. To use OCR independently:
    ```bash
    python ocr_model.py
    ```


## Using Detector With OCR

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Arabic-Ocr-Detection-Api.git
   cd ocr-detection-api
   ```

2. **Dowmload the models**:
   - Download the models used from this [link](https://drive.google.com/drive/folders/1lulv528xWgTirbGkyivxTKoEwFNymFcA?usp=sharing)
   - Create a folder called models and place the required model in it.

3. **Build the Docker image**:
   ```bash
   docker build -t ocr-app .
   ```

4. **Run the Docker container**:
   ```bash
   docker run -d -p 8000:8000 --name ocr-app-container ocr-app
   ```

4. Once the application is running, you can interact with it via the `/predict/` endpoint, example: 
   ```bash
   curl -X POST "http://localhost:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
   ```


