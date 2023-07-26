import cv2
import numpy as np

# Load the image
image = cv2.imread('pajama.jpg', cv2.IMREAD_GRAYSCALE)

# Create Sobel Filter X (Horizontal)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

# Create Sobel Filter Y (Vertical)
sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

# Apply Sobel Filters using cv2.filter2D() to perform convolution
filtered_image_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
filtered_image_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

# Take the absolute values to get the magnitude
filtered_image_x = np.abs(filtered_image_x)
filtered_image_y = np.abs(filtered_image_y)

# Normalize the filtered images to the range [0, 255]
filtered_image_x = cv2.normalize(filtered_image_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
filtered_image_y = cv2.normalize(filtered_image_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the filtered images
cv2.imshow('Sobel X (Horizontal)', filtered_image_x)
cv2.imshow('Sobel Y (Vertical)', filtered_image_y)
cv2.imwrite('sobelx_spa.png', filtered_image_x)
cv2.imwrite('sobely_spa.png', filtered_image_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
