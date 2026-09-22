import cv2
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs("day_02/output", exist_ok=True)

# 1. Read the camera man noisy image using the exact full path
image_path = r"C:\Users\PC-LAB1\Desktop\image_processing_course\day_02\images\noisy_camera_man.webp"
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Image not found. Please check the file path.")
else:
    print("Image loaded successfully! Shape:", original_image.shape)

    # 2. Apply Median Blur filter with a 5x5 kernel size (Change 5 to 7 later to test the 7x7 filter)
    kernel_size = 7  # يمكنك تغيير هذا الرقم إلى 7 لاحقاً
    filtered_image = cv2.medianBlur(original_image, kernel_size)

    # Save the filtered image to the output folder
    cv2.imwrite(f"day_02/output/noisy_camera_man_filtered_{kernel_size}x{kernel_size}.jpg", filtered_image)

    # 3. Create a figure with a 2x2 grid for comparison
    plt.figure(figsize=(14, 10))

    # ---- Subplot 1: Original Noisy Image ----
    plt.subplot(2, 2, 1)
    rgb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_original)
    plt.title("Original Noisy Camera Man")
    plt.axis("off")

    # ---- Subplot 2: Filtered Image (Median Kernel Size) ----
    plt.subplot(2, 2, 2)
    rgb_filtered = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_filtered)
    plt.title(f"Filtered Image (Median {kernel_size}x{kernel_size})")
    plt.axis("off")

    # ---- Subplot 3: Histogram of Original Image ----
    plt.subplot(2, 2, 3)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([original_image], [i], None, [256], [0, 256])
        plt.plot(hist_orig, color=color, linewidth=1.5)
        plt.xlim([0, 256])
    plt.title("Histogram: Original Image")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Pixel Count")
    plt.grid(True, linestyle='--', alpha=0.6)

    # ---- Subplot 4: Histogram of Filtered Image ----
    plt.subplot(2, 2, 4)
    for i, color in enumerate(colors):
        hist_filt = cv2.calcHist([filtered_image], [i], None, [256], [0, 256])
        plt.plot(hist_filt, color=color, linewidth=1.5)
        plt.xlim([0, 256])
    plt.title(f"Histogram: Filtered Image ({kernel_size}x{kernel_size})")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Pixel Count")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Adjust layout and show the plot window
    plt.tight_layout()
    plt.show()
    
    print(f"Comparison analysis with {kernel_size}x{kernel_size} filter generated successfully!")