import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the image
img = cv2.imread('3.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display original and grayscale images side by side
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Apply thresholding to keep values in the range of 200 to 255
_, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)

# Get image dimensions
height, width = thresh.shape[:2]

# Calculate the height where you want to start making it black (top 20%)
top_20_percent_height = int(height * 0.2)

# Make the top 20% black
thresh[0:top_20_percent_height, :] = 0  # Set the pixels to black

# Convert OpenCV image to RGB for displaying with pyplot
thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

# Display the modified image using pyplot
plt.imshow(thresh_rgb)
plt.axis('off')  # Hide axes
plt.title('Thresholded Image')
plt.show()

# Display the thresholded image
plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image (200-255)')
plt.axis('off')
plt.show()

# Define the kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Perform morphological opening
thresh_open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Perform morphological closing on the opened image
thresh_close = cv2.morphologyEx(thresh_open, cv2.MORPH_CLOSE, kernel)

# Display the result after morphological opening and closing
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image (200-255)')
plt.axis('off')

plt.subplot(132)
plt.imshow(thresh_open, cmap='gray')
plt.title('After Morphological Opening')
plt.axis('off')

plt.subplot(133)
plt.imshow(thresh_close, cmap='gray')
plt.title('After Morphological Closing')
plt.axis('off')

plt.tight_layout()
plt.show()

# Find contours on the resulting image after closing
contours, _ = cv2.findContours(thresh_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 5)

# Display the image with drawn contours
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Image with Contours')
plt.axis('off')
plt.show()

# Filter contours based on area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]

# Draw filtered contours on the original image
contour_img = img.copy()
cv2.drawContours(contour_img, filtered_contours, -1, (0, 0, 255), 5)

# Display the image with filtered contours
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title('Image with Filtered Contours')
plt.axis('off')
plt.show()

# Combine good contours
contours_combined = np.vstack(filtered_contours)

# Draw combined contours on a copy of the original image
combined_contour_img = img.copy()
cv2.drawContours(combined_contour_img, [contours_combined], -1, (0, 0, 255), 5)

# Plot the result with combined contours
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(combined_contour_img, cv2.COLOR_BGR2RGB))
plt.title('Combined Contours')
plt.axis('off')

plt.tight_layout()
plt.show()

# Find convex hull
hull = cv2.convexHull(contours_combined)

# Draw convex hull on a copy of the original image
convex_hull_img = img.copy()
cv2.polylines(convex_hull_img, [hull], True, (0, 0, 255), 5)

# Plot the result with convex hull
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(convex_hull_img, cv2.COLOR_BGR2RGB))
plt.title('Convex Hull')
plt.axis('off')

plt.tight_layout()
plt.show()