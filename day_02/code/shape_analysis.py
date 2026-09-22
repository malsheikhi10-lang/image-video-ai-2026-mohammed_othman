import cv2
import numpy as np
import os

# Ensure the output directory exists
os.makedirs("day_02/output", exist_ok=True)

# 1. Read the image with shapes using the exact full path
image_path = r"C:\Users\PC-LAB1\Desktop\image_processing_course\day_02\images\image_with_shapes_2.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found. Please check the file path.")
else:
    print("Image loaded successfully! Shape:", image.shape)

    # 2. Convert to Grayscale and apply thresholding to binary image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours (shapes boundaries)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_counts = {"Circle": 0, "Triangle": 0, "Rectangle": 0, "Square": 0, "Other": 0}
    shape_areas = {"Circle": [], "Triangle": [], "Rectangle": [], "Square": [], "Other": []}

    print("\n--- Shape Analysis Results ---")

    # 4. Loop through each detected shape to classify and calculate its area
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:  # Ignore tiny noise
            continue

        # Approximate the contour to identify the number of vertices
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        vertices = len(approx)

        # Classify shapes based on vertices and geometric properties
        if vertices == 3:
            shape_name = "Triangle"
        elif vertices == 4:
            # Check aspect ratio to differentiate between square and rectangle
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Square"
            else:
                shape_name = "Rectangle"
        else:
            # Check circularity to identify circles
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity > 0.90:
                    shape_name = "Circle"
                else:
                    shape_name = "Other"
            else:
                shape_name = "Other"

        shape_counts[shape_name] += 1
        shape_areas[shape_name].append(round(area, 2))

    # 5. Print out the summary of counts and areas
    for shape in shape_counts:
        count = shape_counts[shape]
        areas = shape_areas[shape]
        avg_area = sum(areas) / len(areas) if count > 0 else 0
        print(f"Shape: {shape}")
        print(f"  - Count: {count}")
        print(f"  - Areas: {areas}")
        print(f"  - Average Area: {avg_area:.2f} pixels\n")

    # 6. Display the original image
    cv2.imshow("Original Shapes Image", image)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Shape analysis completed successfully!")