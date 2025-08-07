# yolov5-object-detection-recaptcha
Object detection on Google reCAPTCHA images using YOLOv5 and PyTorch.

# 🧠 Object Detection with YOLOv5 and Google reCAPTCHA Dataset

This project demonstrates how to detect real-world objects in images using the **YOLOv5** object detection model, trained and evaluated on the **Google reCAPTCHA Dataset**. This type of task mimics human validation systems like CAPTCHA, helping models learn to identify specific visual patterns across diverse scenes.

---

## 📦 Dataset: Google reCAPTCHA Dataset

- **Source**: [Google reCAPTCHA Dataset on Kaggle](https://www.kaggle.com)
- **Description**: The dataset contains a variety of images used in CAPTCHA systems. Each image includes multiple objects such as traffic lights, crosswalks, cars, motorcycles, buses, fire hydrants, and more.
- **Purpose**: Designed for training and evaluating object detection algorithms.

---

## 🚀 Goal of the Project

- Download and prepare the dataset.
- Use a pre-trained YOLOv5 model to detect objects in the dataset images.
- Randomly select five images and demonstrate the detection results.
- Evaluate the detected objects versus expected outcomes.

---

## 🧪 Steps Performed

### 1. Dataset Preparation

- Downloaded the dataset using Kaggle CLI.
- Extracted and organized images for inference.

### 2. YOLOv5 Inference

- Used the Ultralytics implementation of YOLOv5 via PyTorch.
- Specified a path to input images.
- Ran inference on each image to detect bounding boxes and class labels.

### 3. Result Collection

- Randomly selected **5 images** from the dataset.
- Applied YOLOv5 inference on each image.
- Collected the following for each image:
  - The original image.
  - Objects detected by YOLOv5.
  - Expected objects (if available).
- Saved and visualized annotated images with detected bounding boxes.

---

## 🖼️ Sample Results

The following results were generated for randomly selected samples:

| Image | Detected Objects |
|-------|------------------|
| sample_1.jpg | Bus, Car, Traffic Light |
| sample_2.jpg | Crosswalk, Person |
| sample_3.jpg | Car, Motorcycle |
| sample_4.jpg | Traffic Light, Fire Hydrant |
| sample_5.jpg | Bicycle, Stop Sign |

> *Note: YOLOv5 uses confidence thresholds and NMS (Non-Max Suppression) to reduce false positives.*

---

## 🛠️ Tools and Frameworks

- **Python** (Jupyter Notebook)
- **PyTorch** for deep learning
- **YOLOv5** by Ultralytics
- **OpenCV** and **Matplotlib** for image processing and visualization

---

## 🖥️ Running the Code

To run the object detection code:

1. Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

2. Download the dataset using Kaggle CLI:

```bash
!kaggle datasets download -d [DATASET_PATH]
!unzip [DATASET_FILE].zip
```

3. Place your dataset images into the `inference/images/` folder.

4. Run inference from notebook or terminal:

```bash
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source inference/images
```

---

## 📚 References

- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [Google reCAPTCHA Dataset on Kaggle](https://www.kaggle.com)
- [Ultralytics Documentation](https://docs.ultralytics.com/)

---

## 📈 Results and Conclusion

This exercise successfully demonstrates the capability of **YOLOv5** to detect real-world objects in CAPTCHA-like scenarios. The results show strong generalization to various classes including vehicles, pedestrians, and traffic-related elements.

---

## 👨‍💻 Author

Developed as part of a practical assignment on real-time object detection using deep learning techniques.
