# Image Processing Application 🖼️⚙️

This project is a **Python-based GUI application** for **image processing** built with **OpenCV, Tkinter, NumPy, and Matplotlib**.  
It allows users to load an image and apply a wide range of transformations, filters, and computer vision algorithms.

## 🚀 Features

The application is organized into **labs** and **assignments**, each covering different image processing operations:

### 🔹 Basic Transformations
- **Grayscale conversion**  
- **Black & White conversion** (thresholding)  
- **HSV color space conversion**  
- **Histograms**: RGB & grayscale  

### 🔹 Image Enhancements
- **Negative transformation**  
- **Gamma correction**  
- **Brightness adjustment**  
- **Contrast adjustment**  

### 🔹 Filtering
- **Averaging filter (mean)**  
- **Gaussian filter**  
- **Laplacian filter**  

### 🔹 Connected Components
- **Two-pass algorithm** for labeling connected components  
- **BFS-based labeling**  
- **Colored component visualization**  

### 🔹 Edge Detection
- **Gradient computation** with Sobel operators  
- **Adaptive thresholding**  
- **Binary edge detection**  

### 🔹 Quantization & Dithering
- **Multi-threshold quantization**  
- **Floyd-Steinberg dithering**  

### 🔹 Histogram & Thresholding
- **Global automatic binarization (iterative thresholding)**  
- **Histogram equalization**  

### 🔹 Fourier Transform Filters
- **Gaussian Low-Pass**  
- **Ideal Low-Pass**  
- **Gaussian High-Pass**  
- **Ideal High-Pass**  

### 🔹 Morphological Operations
- **Contour extraction**  
- **Region filling (flood fill)**  

### 🔹 Chain Codes
- **Contour chain code representation** with direction encoding  

## 🖥️ User Interface

- **Image panel**: Displays the loaded/processed image.  
- **Buttons panel**: Each button corresponds to an operation (grouped by labs & assignments).  
- **Chain code output**: Displays chain code for detected contours.  

## 📊 Example Workflow

1. Load an image using **"Incarcare imagine"**.  
2. Apply transformations like **Grayscale** or **Contrast adjustment**.  
3. Explore **filters** and **Fourier-based operations**.  
4. Visualize results with histograms or chain codes.  
5. Processed images are automatically saved.  
