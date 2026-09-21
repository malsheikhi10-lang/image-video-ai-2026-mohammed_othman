from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Network image paths
image_paths = {
    "Lena5": r"\\192.168.1.10\shear all\Lena5.jpg",
    "Original_Lena512": r"\\192.168.1.10\shear all\Original_lena512.jpg"
}

for name, path in image_paths.items():
    print(f"==============================")
    print(f" Analyzing Image: {name}")
    print(f"==============================")
    try:
        # Open image from network share
        img = Image.open(path)
        image_array = np.array(img)
        
        # Print shape and data type
        print(f"Shape: {image_array.shape}")
        print(f"Data type: {image_array.dtype}")
        
        # Check if color (3 channels) or grayscale (1 channel)
        if len(image_array.shape) == 3:
            channels = ['Red (R)', 'Green (G)', 'Blue (B)']
            for i, channel_name in enumerate(channels):
                ch_data = image_array[:, :, i]
                print(f"  - Channel {channel_name}:")
                print(f"    Max Value: {ch_data.max()}")
                print(f"    Min Value: {ch_data.min()}")
        else:
            print(f"  - Grayscale Channel:")
            print(f"    Max Value: {image_array.max()}")
            print(f"    Min Value: {image_array.min()}")
                
        print(f"Overall - Max: {image_array.max()}, Min: {image_array.min()}\n")
        
        # Display the image using Matplotlib
        plt.figure()
        if len(image_array.shape) == 2:
            plt.imshow(image_array, cmap='gray')
        else:
            plt.imshow(image_array)
        plt.title(f"Image: {name}")
        plt.axis("off")
        plt.show()
            
    except Exception as e:
        print(f"Error opening image {name}: {e}\n")