import numpy as np
import cv2 as cv

def getSobelX():
    return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
def getSobelY():
    return np.array([[-1, -2, -1],
                        [0, 0, 0], 
                        [1, 2, 1]])
    
def filterToFrequency(filter,s):
    # Fourier Transform of Filter Sobel
    # เเปลง filter เป็น frequency domain
    sobel_freq = np.fft.fft2(filter, s=s)
    sobel_freq_shifted = np.fft.fftshift(sobel_freq)
    return sobel_freq_shifted
    
def imgToFrequency(image):
    # เเปลง รุป เป็น frequency domain
    image_freq = np.fft.fft2(image)
    #shift frequency domain เป็น
    image_freq_shifted = np.fft.fftshift(image_freq)
    return image_freq_shifted

def imgfrequencyToMagnitude(image):
    image_real = np.real(image)
    img_imagine = np.imag(image)
    image_magnitude = np.sqrt(image_real**2 + img_imagine**2)
    image_magnitude = np.log(1+image_magnitude)
    image_magnitude = cv.normalize(image_magnitude,None,0,255,cv.NORM_MINMAX,cv.CV_8U)
    return image_magnitude

if __name__=="__main__":
    img=cv.imread("./ponspa.jpg",cv.IMREAD_GRAYSCALE);
    sobelx=getSobelX()
    sobely=getSobelY()
    sobelxfreq=filterToFrequency(sobelx,s=img.shape)
    sobelyfreq=filterToFrequency(sobely,s=img.shape)
    mag_sobelx=imgfrequencyToMagnitude(sobelx)
    mag_sobely=imgfrequencyToMagnitude(sobely)
    
    imgfreq=imgToFrequency(img)
    mag_img=frequencyToMagnitude(imgfreq)
    
    cv.imshow("magnitude_img",mag_img)
    cv.imshow("sobelx_freq",mag_sobelx)
    cv.imshow("sobely_freq",mag_sobely)
    cv.imshow("original",img)
    cv.waitKey(0)
    cv.destroyAllWindows()