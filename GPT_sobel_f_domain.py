import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('input.png', cv2.IMREAD_GRAYSCALE)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

sobel_x_freq = np.fft.fft2(sobel_x, s=image.shape)
sobel_x_freq_shifted = np.fft.fftshift(sobel_x_freq)

sobel_y_freq = np.fft.fft2(sobel_y, s=image.shape)
sobel_y_freq_shifted = np.fft.fftshift(sobel_y_freq)

image_freq = np.fft.fft2(image)
image_freq_shifted = np.fft.fftshift(image_freq)

sobel_x_magnitude = np.abs(sobel_x_freq_shifted)
sobel_y_magnitude = np.abs(sobel_y_freq_shifted)

filtered_image_x_freq_shifted = image_freq_shifted * sobel_x_magnitude
filtered_image_y_freq_shifted = image_freq_shifted * sobel_y_magnitude

filtered_image_x = np.fft.ifftshift(filtered_image_x_freq_shifted)
filtered_image_y = np.fft.ifftshift(filtered_image_y_freq_shifted)

filtered_image_x = np.fft.ifft2(filtered_image_x)
filtered_image_y = np.fft.ifft2(filtered_image_y)

filtered_image_x = np.abs(filtered_image_x)
filtered_image_y = np.abs(filtered_image_y)

filtered_image_x = np.uint8(filtered_image_x)
filtered_image_y = np.uint8(filtered_image_y)

plt.subplot(121), plt.imshow(filtered_image_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(122), plt.imshow(filtered_image_y, cmap='gray'), plt.title('Sobel Y')
plt.show()
