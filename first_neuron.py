import numpy as np
import cv2


def sigmoid(x):
    return 1/(1+np.exp(-x))


# 1. Input 
# 1.1 Image preprocessing 
img = cv2.imread("images/vertical.png", cv2.IMREAD_GRAYSCALE) 
img = cv2.resize(img, (3, 3))
img = img / 255 
img_flattened = img.flatten()

# 2. Weights
weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
print(img_flattened)
weighted_sum = sum(img_flattened * weights)
print(weighted_sum)

# 3. Activation function 
result = sigmoid(weighted_sum)

# 4. Output
# Vertical 0
if result < 0.5:
    print("Vertical")
# Horizontal 1
elif result > 0.5:
    print("Horizontal")