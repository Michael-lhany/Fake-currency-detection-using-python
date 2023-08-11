# import the required library
import cv2
from skimage.feature import hog
import numpy as np

# Real currency image (Reference image)
real_image = cv2.imread("WhatsApp Image 2023-05-17 at 8.49.12 PM.jpeg")

# get the new dimensions to resize the image
width = 800
height = 400
dim = (width, height)
real_re_img = cv2.resize(real_image, dim, interpolation=cv2.INTER_AREA)

# Convert the real image to Grayscale & Blur
real_gray = cv2.cvtColor(real_re_img, cv2.COLOR_BGR2GRAY)
real_blur = cv2.GaussianBlur(real_gray, (3, 3), 0)
# Canny Edge detection the real image
real_edges = cv2.Canny(image=real_blur, threshold1=100, threshold2=200)
# Calculate HOG features for real image
real_hog_features, real_hog_image = hog(real_edges, orientations=8, pixels_per_cell=(8, 8),
                                        cells_per_block=(1, 1), visualize=True)
# -------------------------------------------------------------------------------------------#

# Read the test image
img = cv2.imread('WhatsApp Image 2023-05-17 at 8.50.54 PM.jpeg')

# get the new dimensions to resize the image
width = 800
height = 400
dim = (width, height)
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Display original & test image
cv2.imshow('Real', real_re_img)
cv2.imshow('Test', resized_img)
cv2.waitKey(0)

# Convert to grayscale
test_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
test_blur = cv2.GaussianBlur(test_gray, (3, 3), 0)

# Display grayscale & Blur image
cv2.imshow('Grayscale & Blur test', test_blur)
cv2.imshow('Grayscale & Blur real', real_blur)
cv2.waitKey(0)

# Canny Edge Detection for test image
test_edges = cv2.Canny(image=test_blur, threshold1=100, threshold2=200)

# Calculate HOG features for test image
test_hog_features, test_hog_image = hog(test_edges, orientations=8, pixels_per_cell=(8, 8),
                                        cells_per_block=(1, 1), visualize=True)

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection test', test_edges)
cv2.imshow('Canny Edge Detection real', real_edges)
cv2.waitKey(0)

# Display HOG features image
cv2.imshow('Test HOG image', test_hog_image)
cv2.imshow('Real HOG image', real_hog_image)
cv2.waitKey(0)

# Calculate the mean intensity of the pixels in the edge-detected images
real_mean = cv2.mean(real_edges)[0]
test_mean = cv2.mean(test_edges)[0]

# Set a threshold for the intensity difference
threshold = 1

# Determine if the test currency is real or fake based on intensity difference
if abs(real_mean - test_mean) < threshold:
    print("The test currency is real based on intensity difference.")
else:
    print("The test currency is fake based on intensity difference.")


# pad test_hog_features with zeros to make it the same size as real_hog_features
padding = np.zeros(real_hog_features.size - test_hog_features.size)
test_hog_features = np.concatenate((test_hog_features, padding))

# compute the distance between the two feature vectors
hog_distance = np.linalg.norm(real_hog_features - test_hog_features)

# Set a threshold for the HOG distance
hog_threshold = 45

# Determine if the test currency is real or fake based on HOG distance
if hog_distance < hog_threshold:
    print("The test currency is real based on HOG distance.")
else:
    print("The test currency is fake based on HOG distance.")

print(hog_distance)
print(real_mean)
print(test_mean)

cv2.waitKey(0)
cv2.destroyAllWindows()
