import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic color image programmatically (Height=300, Width=300, 3 Color Channels RGB)
# This guarantees it will run instantly without any missing file errors!
image = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw some colored shapes/regions on the image
image[50:250, 50:150] = [255, 0, 0]    # Red block
image[50:250, 150:250] = [0, 255, 0]  # Green block

# Print image statistics and properties
print("==============================")
print(" Image Analysis Report")
print("==============================")
print(f"Shape: {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Minimum Value: {image.min()}")
print(f"Maximum Value: {image.max()}")
print(f"Mean Value: {image.mean():.2f}")
print("==============================\n")

# Display the image using Matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("Synthetic Image Analysis Lab 3")
plt.axis("off")
plt.show()