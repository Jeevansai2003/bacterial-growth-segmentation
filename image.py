import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

from skimage import io, color, filters, measure, transform, morphology
from skimage.filters import gaussian

import cv2
import numpy as np
import os

# ---------------- GUI WINDOW ----------------
window = tk.Tk()
window.title("Bacterial Growth Segmentation")

font_style = ("Comic Sans MS", 12)

file_path = tk.StringVar()
resize_factor = tk.DoubleVar(value=1.0)
manual_threshold_value = tk.DoubleVar(value=0.5)
segmentation_method = tk.StringVar(value="Multi-Threshold")


# ---------------- IMAGE LOADING ----------------
def load_image():
    file_path.set(
        filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )
    )
    update_image_preview()


def update_image_preview():
    if file_path.get():
        image = Image.open(file_path.get())
        image.thumbnail((400, 400))
        img_preview = ImageTk.PhotoImage(image)
        img_label.configure(image=img_preview)
        img_label.image = img_preview


# ---------------- THRESHOLD SELECTION ----------------
def get_binary_mask(preprocessed_image, color_mask):
    method = segmentation_method.get()

    if method == "Otsu":
        otsu_mask = preprocessed_image > filters.threshold_otsu(preprocessed_image)
        return otsu_mask

    elif method == "Adaptive":
        adaptive_mask = preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15)
        return adaptive_mask

    elif method == "Manual":
        manual_mask = preprocessed_image > manual_threshold_value.get()
        return manual_mask

    else:  # Multi-Threshold Fusion
        otsu_mask = preprocessed_image > filters.threshold_otsu(preprocessed_image)
        adaptive_mask = preprocessed_image > filters.threshold_local(preprocessed_image, block_size=15)
        manual_mask = preprocessed_image > manual_threshold_value.get()

        fusion_mask = (otsu_mask.astype(int) + adaptive_mask.astype(int) + manual_mask.astype(int)) >= 2
        return fusion_mask | color_mask


# ---------------- IMAGE PROCESSING ----------------
def process_image():
    if not file_path.get():
        return

    image = io.imread(file_path.get())

    # Resize
    resized_image = transform.resize(
        image,
        (int(image.shape[0] * resize_factor.get()),
         int(image.shape[1] * resize_factor.get())),
        mode='reflect',
        anti_aliasing=True
    )

    # -------- Color-based segmentation (HSV) --------
    hsv_image = color.rgb2hsv(resized_image)

    lower_yellow = (0.10, 0.4, 0.4)
    upper_yellow = (0.18, 1.0, 1.0)

    color_mask = (
        (hsv_image[:, :, 0] >= lower_yellow[0]) &
        (hsv_image[:, :, 0] <= upper_yellow[0]) &
        (hsv_image[:, :, 1] >= lower_yellow[1]) &
        (hsv_image[:, :, 2] >= lower_yellow[2])
    )

    # -------- Preprocessing --------
    gray_image = color.rgb2gray(resized_image)
    preprocessed_image = gaussian(gray_image, sigma=1)

    # -------- Selected threshold method --------
    binary_image = get_binary_mask(preprocessed_image, color_mask)

    # -------- Morphological refinement --------
    cleaned_image = morphology.opening(binary_image, morphology.disk(2))
    cleaned_image = morphology.closing(cleaned_image, morphology.disk(3))
    cleaned_image = morphology.remove_small_objects(cleaned_image, min_size=100)
    cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=500)
    cleaned_image = morphology.dilation(cleaned_image, morphology.disk(1))

    # -------- Final segmentation --------
    segmented_image = resized_image.copy()
    segmented_image[~cleaned_image] = 0

    # -------- Display images --------
    images = [resized_image, gray_image, preprocessed_image, binary_image, cleaned_image, segmented_image]

    titles = [
        "Resized Image",
        "Grayscale Image",
        "Preprocessed Image",
        f"Binary Mask ({segmentation_method.get()})",
        "Morphological Refinement",
        "Final Segmented Image"
    ]

    display_images(images, titles)

    # -------- Colony graphs only for Multi-Threshold --------
    if segmentation_method.get() == "Multi-Threshold":
        colony_size_analysis(cleaned_image)


# ---------------- COLONY SIZE ANALYSIS ----------------
def colony_size_analysis(binary_mask):

    mask_uint8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colony_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]

    if len(colony_areas) == 0:
        print("No colonies detected")
        return

    # Colony index vs area
    plt.figure()
    plt.plot(range(1, len(colony_areas) + 1), colony_areas, marker='o')
    plt.xlabel("Colony Index")
    plt.ylabel("Area (pixels)")
    plt.title("Colony Index vs Area")
    plt.grid(True)
    plt.show()

    # Histogram
    plt.figure()
    plt.hist(colony_areas, bins=10)
    plt.xlabel("Colony Area (pixels)")
    plt.ylabel("Frequency")
    plt.title("Colony Size Distribution")
    plt.show()


# ---------------- TIME-SERIES GROWTH CURVE ----------------
def growth_curve_analysis():

    folder = filedialog.askdirectory()
    if not folder:
        return

    days, total_areas = [], []

    image_files = sorted(os.listdir(folder))

    for idx, file in enumerate(image_files):

        path = os.path.join(folder, file)
        image = io.imread(path)

        resized_image = transform.resize(
            image,
            (int(image.shape[0] * resize_factor.get()),
             int(image.shape[1] * resize_factor.get())),
            mode='reflect',
            anti_aliasing=True
        )

        hsv_image = color.rgb2hsv(resized_image)

        lower_yellow = (0.10, 0.4, 0.4)
        upper_yellow = (0.18, 1.0, 1.0)

        color_mask = (
            (hsv_image[:, :, 0] >= lower_yellow[0]) &
            (hsv_image[:, :, 0] <= upper_yellow[0]) &
            (hsv_image[:, :, 1] >= lower_yellow[1]) &
            (hsv_image[:, :, 2] >= lower_yellow[2])
        )

        gray_image = color.rgb2gray(resized_image)
        preprocessed_image = gaussian(gray_image, sigma=1)

        binary_image = get_binary_mask(preprocessed_image, color_mask)

        cleaned_image = morphology.opening(binary_image, morphology.disk(2))
        cleaned_image = morphology.closing(cleaned_image, morphology.disk(3))
        cleaned_image = morphology.remove_small_objects(cleaned_image, min_size=100)
        cleaned_image = morphology.remove_small_holes(cleaned_image, area_threshold=500)
        cleaned_image = morphology.dilation(cleaned_image, morphology.disk(1))

        mask_uint8 = (cleaned_image.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20)

        days.append(idx + 1)
        total_areas.append(total_area)

    # Growth curve
    if days:
        plt.figure()
        plt.plot(days, total_areas, marker='o')
        plt.xlabel("Day")
        plt.ylabel("Total Colony Area (pixels)")
        plt.title("Bacterial Growth Curve Over Time")
        plt.grid(True)
        plt.show()


# ---------------- DISPLAY FUNCTION ----------------
def display_images(images, titles):
    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), constrained_layout=True)
    axes = axes.flatten()

    for i in range(n):
        if images[i].ndim == 2:
            axes[i].imshow(images[i], cmap='gray')
        else:
            axes[i].imshow(images[i])

        axes[i].set_title(titles[i])
        axes[i].axis('off')

    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


# ---------------- GUI CONTROLS ----------------
tk.Button(window, text="Browse Image", command=load_image, font=font_style).pack()

img_label = tk.Label(window)
img_label.pack()

resize_frame = tk.Frame(window)
resize_frame.pack()

tk.Label(resize_frame, text="Resize Factor:", font=font_style).pack(side=tk.LEFT)

for val in [1, 2, 3, 4]:
    tk.Button(resize_frame, text=str(val), font=font_style,
              command=lambda v=val: resize_factor.set(v), width=2).pack(side=tk.LEFT)

# -------- Dropdown for segmentation method --------
method_frame = tk.Frame(window)
method_frame.pack(pady=5)

tk.Label(method_frame, text="Segmentation Method:", font=font_style).pack(side=tk.LEFT)

method_menu = tk.OptionMenu(method_frame, segmentation_method,
                            "Otsu", "Adaptive", "Manual", "Multi-Threshold")
method_menu.pack(side=tk.LEFT)

# Manual threshold input
manual_frame = tk.Frame(window)
manual_frame.pack()

tk.Label(manual_frame, text="Manual Threshold:", font=font_style).pack(side=tk.LEFT)
tk.Entry(manual_frame, textvariable=manual_threshold_value, width=5).pack(side=tk.LEFT)

# Buttons
tk.Button(window, text="Process Image", command=process_image, font=font_style).pack(pady=10)

tk.Button(window, text="Time-Series Growth Curve",
          command=growth_curve_analysis, font=font_style).pack(pady=5)

# ---------------- RUN APP ----------------
window.mainloop()
