import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Lena5 image from the network path
image_path = r"\\192.168.1.10\shear all\Lena5.jpg"
image_bgr = cv2.imread(image_path)

# Convert BGR to HSV color space
image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

# Split HSV channels
h_channel, s_channel, v_channel = cv2.split(image_hsv)

# Normalize Hue channel for clear display (OpenCV Hue is 0-179, scaled to 0-255)
h_channel_normalized = cv2.normalize(h_channel, None, 0, 255, cv2.NORM_MINMAX)

# Display the HSV image and its individual channels side by side
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Full HSV Image (converted to RGB for proper visualization)
axes[0].imshow(cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB))
axes[0].set_title("Lena5 - Full HSV")
axes[0].axis("off")

# Hue Channel
axes[1].imshow(h_channel_normalized, cmap='gray')
axes[1].set_title("Hue (H) Channel")
axes[1].axis("off")

# Saturation Channel
axes[2].imshow(s_channel, cmap='gray')
axes[2].set_title("Saturation (S) Channel")
axes[2].axis("off")

# Value Channel
axes[3].imshow(v_channel, cmap='gray')
axes[3].set_title("Value (V) Channel")
axes[3].axis("off")

plt.tight_layout()
plt.show()