import cv2 as cv
import numpy as np

# Load the image
image = cv.imread('pajama.jpg', cv.IMREAD_GRAYSCALE)

# Create Filter Sobel in Spatial Domain
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0], 
                    [1, 2, 1]])

# Fourier Transform of Filter Sobel
# เเปลง filter เป็น frequency domain
sobel_x_freq = np.fft.fft2(sobel_x, s=image.shape)
sobel_x_freq_shifted = np.fft.fftshift(sobel_x_freq)


sobel_y_freq = np.fft.fft2(sobel_y, s=image.shape)
sobel_y_freq_shifted = np.fft.fftshift(sobel_y_freq)

# Fourier Transform of the input image
# เเปลง รุป เป็น frequency domain
image_freq = np.fft.fft2(image)
#shift frequency domain เป็น
image_freq_shifted = np.fft.fftshift(image_freq)

## ทำให้รูปพร้อม เเสดง #########################################
image_real = np.real(image_freq_shifted)
img_imagine = np.imag(image_freq_shifted)
image_magnitude = np.sqrt(image_real**2 + img_imagine**2)

#display magnitude
image_magnitude = np.log(1+image_magnitude)
image_magnitude = cv.normalize(image_magnitude,None,0,255,cv.NORM_MINMAX,cv.CV_8U)

image_freq_shifted_mag = np.abs(image_freq_shifted)




# Magnitude of the Sobel filters in the frequency domain
sobel_x_magnitude = np.abs(sobel_x_freq_shifted)
sobel_y_magnitude = np.abs(sobel_y_freq_shifted)

# Apply the filters in the frequency domain
filtered_image_x_freq_shifted = image_freq_shifted * sobel_x_magnitude
filtered_image_y_freq_shifted = image_freq_shifted * sobel_y_magnitude

# Inverse Fourier Transform and get the magnitude
filtered_image_x = np.fft.ifftshift(filtered_image_x_freq_shifted)
filtered_image_y = np.fft.ifftshift(filtered_image_y_freq_shifted)

filtered_image_x = np.fft.ifft2(filtered_image_x)
filtered_image_y = np.fft.ifft2(filtered_image_y)

filtered_image_x = np.abs(filtered_image_x)
filtered_image_y = np.abs(filtered_image_y)

# Convert back to uint8 and display the filtered images
filtered_image_x = np.uint8(filtered_image_x)
filtered_image_y = np.uint8(filtered_image_y)

cv.imshow("input image in gray scale",image)

cv.imshow("image_magnitude",image_magnitude)

cv.imshow("magnitude_sobelx",sobel_x_magnitude)

cv.imshow("magnitude_sobely",sobel_y_magnitude)

cv.imshow('Sobel X', filtered_image_x)

cv.imshow('Sobel Y', filtered_image_y)
cv.waitKey(0)