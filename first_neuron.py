import numpy as np
import cv2

# Sigmoid squishes any number into a value between 0 and 1
# This lets the neuron turn its weighted sum into an output
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid:
# Measures how much the sigmoid output changes
# Used when working out how to adjust the weights during training
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


# 1. Input 
# 1.1 Image preprocessing 
img = cv2.imread("images/vertical.png", cv2.IMREAD_GRAYSCALE) 
img = cv2.resize(img, (3, 3))
img = img / 255 
img_flattened = img.flatten()

# 2. Weights
weights = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

for i in range(200):
    print("Round: ", i +1)
    convolution = sum(img_flattened * weights)


    # 3. Activation function 
    result = sigmoid(convolution )

    # 4. Output
    # Vertical 0
    if result < 0.5:
        print("Vertical")
    # Horizontal 1
    elif result > 0.5:
        print("Horizontal")


    # 5. Error
    error = result - 0
    print("Error: ",error)

    # 6. Adjustment
    adjustment = error * sigmoid_der(result)
    print("Adjustment: ", adjustment)


    weights -=np.dot(img_flattened, adjustment)
    print("Weight: ",weights)
    print("\n")