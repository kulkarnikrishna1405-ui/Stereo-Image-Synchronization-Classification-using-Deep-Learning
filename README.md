# Stereo-Image-Synchronization-Classification-using-Deep-Learning
Developed a deep learning pipeline to classify stereo image pairs as synchronized or asynchronous for ADAS, AR/VR, and 3D vision. Built an augmented dataset, applied digital image processing, and designed a CNN for spatial/appearance similarity, enabling robust real-time stereo validation.

# Stereo-Image-Synchronization-Classification-using-Deep-Learning  

This project implements a **Deep Learning pipeline** for classifying whether a stereo image pair is **synchronized (Sync)** or **asynchronous (Async)**. It uses a **ResNet18-based Siamese architecture** trained on stereo image datasets.  

---

## 📂 Project Structure  
.
├── DATASET.zip # Dataset (zipped)
├── DATASET/ # Extracted dataset (left/ and right/ folders)
├── StereoSyncNet.py # Main training + inference script
├── stereo_sync_model_fixed.pth # Trained model (saved after training)
├── README.md # Project documentation


---

## 🚀 Features  

- Automatically generates **Sync (matched)** and **Async (mismatched)** stereo pairs.  
- Uses **ResNet18 pretrained on ImageNet** as feature extractor.  
- Binary classification (Sync / Async) with **Sigmoid output**.  
- Runs smoothly on **Google Colab / Jupyter Notebook**.  
- Includes **UI** for uploading two images and checking synchronization.  

---

## 📦 Requirements  

Install dependencies:  

```bash
pip install torch torchvision matplotlib pillow ipywidgets
