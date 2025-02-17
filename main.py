import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import random
from collections import deque

img = None

def load_image():
    global img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        show_image(img)

def show_image(image):
    fixed_size = (400, 400)
    image = cv2.resize(image, fixed_size)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    
    panel.config(image=image)
    panel.image = image

# lab 1
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def grayscale():
    global img
    gray = to_grayscale(img)
    show_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Grayscale.jpg', gray)

def black_white():
    global img
    gray = to_grayscale(img)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    show_image(cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('BlackWhite.jpg', bw)

def hsv():
    global img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    show_image(hsv)
    cv2.imwrite('HSV.jpg', hsv)

def histograma_color():
    global img
    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.title("Color Histogram (RGB)")
    plt.show()

def histograma_gri():
    global img
    gray = to_grayscale(img)
    plt.figure()
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.title("Grayscale Histogram")
    plt.show()

# lab 3
def negativ():
    global img
    negative_img = 255 - img
    show_image(negative_img)
    cv2.imwrite('Negative.jpg', negative_img)

def gamma():
    global img
    gamma = 3.0
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    gamma_img = cv2.LUT(img, table)
    show_image(gamma_img)
    cv2.imwrite('Gamma_Corrected.jpg', gamma_img)

def luminozitate():
    global img
    brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=60)
    show_image(brightness_img)
    cv2.imwrite('Brightness_Adjusted.jpg', brightness_img)

def contrast():
    global img
    contrast_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
    show_image(contrast_img)
    cv2.imwrite('Contrast_Adjusted.jpg', contrast_img)

# lab 4
def media_aritmetica():
    global img
    kernel = np.ones((3, 3), np.float32) / 9
    avg_img = cv2.filter2D(img, -1, kernel)
    show_image(avg_img)
    cv2.imwrite('Averaging_Filter.jpg', avg_img)

def gauss():
    global img
    gauss_img = cv2.GaussianBlur(img, (3, 3), 0)
    show_image(gauss_img)
    cv2.imwrite('Gaussian_Filter.jpg', gauss_img)

def laplace():
    global img
    laplace = cv2.Laplacian(img, cv2.CV_64F)
    laplace = np.uint8(np.absolute(laplace))
    show_image(laplace)
    cv2.imwrite('Laplacian_Filter.jpg', laplace)

# lab 6
def doua_treceri(binary_image):
    height, width = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=int)
    label = 0
    edges = {i: [] for i in range(10000)}

    directions = [(-1, 0), (0, -1)]

    for r in range(height):
        for c in range(width):
            if binary_image[r, c] == 0 and labels[r, c] == 0:
                neighbors = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and labels[nr, nc] > 0:
                        neighbors.append(labels[nr, nc])

                if len(neighbors) == 0:
                    label += 1
                    labels[r, c] = label
                else:
                    min_label = min(neighbors)
                    labels[r, c] = min_label
                    for neighbor_label in neighbors:
                        if neighbor_label != min_label:
                            edges[min_label].append(neighbor_label)
                            edges[neighbor_label].append(min_label)

    new_labels = np.zeros(label + 1, dtype=int)
    new_label = 0
    for i in range(1, label + 1):
        if new_labels[i] == 0:
            new_label += 1
            queue = deque([i])
            new_labels[i] = new_label
            while queue:
                current = queue.popleft()
                for neighbor in edges[current]:
                    if new_labels[neighbor] == 0:
                        new_labels[neighbor] = new_label
                        queue.append(neighbor)

    for r in range(height):
        for c in range(width):
            labels[r, c] = new_labels[labels[r, c]]

    return labels

def colorare_componente(binary_image, labels):
    height, width = binary_image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    max_label = np.max(labels)
    colors = {i: tuple([random.randint(0, 255) for _ in range(3)]) for i in range(1, max_label + 1)}

    for r in range(height):
        for c in range(width):
            if binary_image[r, c] == 255:
                colored_image[r, c] = [255, 255, 255]
            elif labels[r, c] > 0:
                colored_image[r, c] = colors[labels[r, c]]

    return colored_image

def componente_conexe():
    global img
    gray_image = to_grayscale(img)
    
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    labels = doua_treceri(binary_image)
    colored_image = colorare_componente(binary_image, labels)

    show_image(colored_image)
    cv2.imwrite('Connected_Components.jpg', colored_image)

def etichetare_bfs(binary_image):
    height, width = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=int)
    label = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def bfs(r, c, current_label):
        queue = deque([(r, c)])
        labels[r, c] = current_label

        while queue:
            x, y = queue.popleft()
            for dr, dc in directions:
                nr, nc = x + dr, y + dc
                if (
                    0 <= nr < height and 0 <= nc < width
                    and binary_image[nr, nc] == 0
                    and labels[nr, nc] == 0
                ):
                    labels[nr, nc] = current_label
                    queue.append((nr, nc))

    for r in range(height):
        for c in range(width):
            if binary_image[r, c] == 0 and labels[r, c] == 0:
                label += 1
                bfs(r, c, label)

    return labels

def bfs_colorare(binary_image, labels):
    height, width = binary_image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    max_label = np.max(labels)
    colors = {i: tuple([random.randint(0, 255) for _ in range(3)]) for i in range(1, max_label + 1)}

    for r in range(height):
        for c in range(width):
            if binary_image[r, c] == 255:
                colored_image[r, c] = [255, 255, 255]
            elif labels[r, c] > 0:
                colored_image[r, c] = colors[labels[r, c]]

    return colored_image

def bfs_conexe():
    global img
    gray_image = to_grayscale(img)
    
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    labels = etichetare_bfs(binary_image)
    colored_image = bfs_colorare(binary_image, labels)

    show_image(colored_image)
    cv2.imwrite('BFS_Connected_Components.jpg', colored_image)

# lab 7
def gradient(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    hist, _ = np.histogram(gradient_magnitude, bins=256, range=(0, 255))
    return hist, gradient_magnitude

def prag_adaptativ(hist, image, p=0.01):
    height, width = image.shape
    num_pixels = height * width
    num_non_edge_pixels = (1 - p) * (num_pixels - hist[0])

    cumulative_sum = 0
    for i in range(1, 256):
        cumulative_sum += hist[i]
        if cumulative_sum > num_non_edge_pixels:
            return i
    return 255

def binarizare_adaptiva(image, p=0.01):
    hist, gradient_magnitude = gradient(image)
    adaptive_threshold = prag_adaptativ(hist, image, p)
    _, binary_image = cv2.threshold(gradient_magnitude, adaptive_threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def detectie_contur_binarizare(gradient_magnitude, low_threshold=50, high_threshold=150):
    edges = np.zeros_like(gradient_magnitude)
    edges[gradient_magnitude >= high_threshold] = 255
    edges[gradient_magnitude < low_threshold] = 0
    return edges

def detectie_contur_adaptiv():
    global img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    binarized_image = binarizare_adaptiva(gray, p=0.01)
    edges = detectie_contur_binarizare(binarized_image)

    show_image(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Adaptive_Threshold_Edges.jpg', edges)

# tema 1
def multi_threshold_quantization():
    global img
    gray = to_grayscale(img)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total_pixels = gray.size
    fdp = hist / total_pixels
    WH, TH = 5, 0.0003
    max_positions = [0] + [k for k in range(WH, 256 - WH) if fdp[k] > np.mean(fdp[k-WH:k+WH+1]) + TH] + [255]
    thresholds = [(max_positions[i] + max_positions[i+1]) // 2 for i in range(len(max_positions)-1)]
    quantized_img = np.zeros_like(gray)
    for i, threshold in enumerate(thresholds):
        quantized_img[gray >= threshold] = max_positions[i+1]
    show_image(cv2.cvtColor(quantized_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Quantized.jpg', quantized_img)

def floyd_steinberg_dithering():
    global img
    gray = to_grayscale(img)
    h, w = gray.shape
    fs_img = gray.copy().astype(float)
    for y in range(h):
        for x in range(w):
            old_pixel = fs_img[y, x]
            new_pixel = 0 if old_pixel < 128 else 255
            fs_img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x + 1 < w: fs_img[y, x + 1] += quant_error * 7 / 16
            if x - 1 >= 0 and y + 1 < h: fs_img[y + 1, x - 1] += quant_error * 3 / 16
            if y + 1 < h: fs_img[y + 1, x] += quant_error * 5 / 16
            if x + 1 < w and y + 1 < h: fs_img[y + 1, x + 1] += quant_error * 1 / 16
    fs_img = np.clip(fs_img, 0, 255).astype(np.uint8)
    show_image(cv2.cvtColor(fs_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Dithered.jpg', fs_img)

# tema 2
def binarizare_automata_globala():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    I_max = np.max(np.nonzero(hist)[0])
    I_min = np.min(np.nonzero(hist)[0])
    T = (I_max + I_min) // 2

    error = 1
    previous_T = T

    while True:
        G1 = gray[gray <= T]
        G2 = gray[gray > T]

        if len(G1) > 0:
            mu_G1 = np.mean(G1)
        else:
            mu_G1 = 0
        if len(G2) > 0:
            mu_G2 = np.mean(G2)
        else:
            mu_G2 = 0

        T = (mu_G1 + mu_G2) / 2

        if abs(T - previous_T) < error:
            break

        previous_T = T

    _, binary_img = cv2.threshold(gray, int(T), 255, cv2.THRESH_BINARY)

    show_image(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Otsu_Binarized.jpg', binary_img)
    print(f"Pragul final calculat: {T}")

def histogram_equalization():
    global img
    gray = to_grayscale(img)
    equalized_img = cv2.equalizeHist(gray)
    show_image(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Histogram_Equalized.jpg', equalized_img)

# tema 3
def transf_fourier(img):
    dft = np.fft.fft2(img)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted

def fourier_invers(dft_shifted):
    dft_ishifted = np.fft.ifftshift(dft_shifted)
    img_back = np.fft.ifft2(dft_ishifted)
    img_back = np.abs(img_back)
    return np.uint8(img_back)

def gaussian_trece_jos(img, cutoff=30):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    
    dft_shifted = transf_fourier(img)
    filtered_dft = dft_shifted * mask
    return fourier_invers(filtered_dft)

def ideal_trece_jos(img, cutoff=30):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    dft_shifted = transf_fourier(img)
    filtered_dft = dft_shifted * mask
    return fourier_invers(filtered_dft)

def gaussian_trece_sus(img, cutoff=30):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 - np.exp(-(distance ** 2) / (2 * (cutoff ** 2)))
    
    dft_shifted = transf_fourier(img)
    filtered_dft = dft_shifted * mask
    return fourier_invers(filtered_dft)

def ideal_trece_sus(img, cutoff=30):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.float32)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    dft_shifted = transf_fourier(img)
    filtered_dft = dft_shifted * mask
    return fourier_invers(filtered_dft)

def aplicare_gaussian_trece_jos():
    global img
    filtered_img = gaussian_trece_jos(img, cutoff=30)
    show_image(filtered_img)

def aplicare_ideal_trece_jos():
    global img
    filtered_img = ideal_trece_jos(img, cutoff=30)
    show_image(filtered_img)

def aplicare_gaussian_trece_sus():
    global img
    filtered_img = gaussian_trece_sus(img, cutoff=30)
    show_image(filtered_img)

def aplicare_ideal_trece_sus():
    global img
    filtered_img = ideal_trece_sus(img, cutoff=30)
    show_image(filtered_img)

# tema 4
def extragere_contur():
    global img
    gray = to_grayscale(img)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel, iterations=1)
    contours = cv2.subtract(gray, eroded)
    show_image(cv2.cvtColor(contours, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Contour.jpg', contours)

def umplere_regiuni():
    global img
    gray = to_grayscale(img)
    mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
    flood_filled = gray.copy()
    cv2.floodFill(flood_filled, mask, (0, 0), 255)
    filled = cv2.bitwise_or(gray, cv2.bitwise_not(flood_filled))
    show_image(cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR))
    cv2.imwrite('Filled.jpg', filled)

# tema 5
def coduri_inlantuite():
    global img
    gray = to_grayscale(img)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        chain_code = []
        current_point = contours[0][0][0]
        prev_direction = None 
        
        for i in range(1, len(contours[0])):
            next_point = contours[0][i][0]
            delta = (next_point[0] - current_point[0], next_point[1] - current_point[1])
            
            for direction_index, direction in enumerate(directions):
                if delta == direction:
                    if prev_direction is None or prev_direction != direction_index:
                        chain_code.append(direction_index)
                        chain_code.append(' ')
                        chain_code.append(direction_index)
                        chain_code.append(' ')
                        chain_code.append(direction_index)
                        chain_code.append(' ')
                    prev_direction = direction_index
                    break
            current_point = next_point
            
        chain_code_display.config(text="" + ''.join(map(str, chain_code)))
    else:
        chain_code_display.config(text="Nu a fost gasit niciun contur")

root = tk.Tk()
root.title("Aplicație PI")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

panel = tk.Label(root)
panel.grid(row=0, column=0, rowspan=6, padx=10, pady=10)

buttons = [
    ("Incarcare imagine", load_image),
    ("Grayscale", grayscale),
    ("Black & White", black_white),
    ("HSV", hsv),
    ("Histograma imaginii", histograma_color),
    ("Histograma imaginii gri", histograma_gri),
    ("Negativul imaginii", negativ),
    ("Ajustare luminozitate", luminozitate),
    ("Ajustare contrast", contrast),
    ("Corectie gamma", gamma),
    ("Filtrul medie aritmetica", media_aritmetica),
    ("Filtrul gaussian", gauss),
    ("Filtrul Laplace", laplace),
    ("Componentele conexe: 2 treceri", componente_conexe),
    ("Componentele conexe: BFS", bfs_conexe),
    ("Binarizare adaptiva", detectie_contur_adaptiv),
    ("Cuantizare cu praguri multiple", multi_threshold_quantization),
    ("Floyd-Steinberg", floyd_steinberg_dithering),
    ("Binarizare automata globala", binarizare_automata_globala),
    ("Egalizare histograma", histogram_equalization),
    ("Filtru gaussian trece jos", aplicare_gaussian_trece_jos),
    ("Filtru ideal trece jos", aplicare_ideal_trece_jos),
    ("Filtru gaussian trece sus", aplicare_gaussian_trece_sus),
    ("Filtru ideal trece sus", aplicare_ideal_trece_sus),
    ("Extragere contur", extragere_contur),
    ("Umplere regiuni", umplere_regiuni),
    ("Coduri inlantuite", coduri_inlantuite)
]

chain_code_display = tk.Label(root, text="", width=20, height=1)
chain_code_display.grid(row=3, column=5, columnspan=1, padx=10, pady=10)


for i, (text, command) in enumerate(buttons):
    tk.Button(root, text=text, command=command, width=25).grid(row=i % 6, column=1 + (i // 6), padx=5, pady=5)

root.mainloop()
