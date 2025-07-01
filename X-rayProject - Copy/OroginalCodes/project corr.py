import numpy as np
from PIL import Image
import scipy.stats

def calculate_correlations(image_path):
    # Open the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Calculate the correlation of adjacent pixels in each direction
    correlations = {}

    # Convert the image to grayscale
    if len(img_array.shape) == 3:
        img_gray = img.convert('L')
        img_array_gray = np.array(img_gray)
    else:
        img_array_gray = img_array

    # Calculate horizontal correlation
    horizontal_corr = np.mean([scipy.stats.pearsonr(img_array_gray[i], img_array_gray[i+1])[0] for i in range(img_array_gray.shape[0] - 1)])
    correlations['Horizontal'] = horizontal_corr

    # Calculate vertical correlation
    vertical_corr = np.mean([scipy.stats.pearsonr(img_array_gray[:, j], img_array_gray[:, j+1])[0] for j in range(img_array_gray.shape[1] - 1)])
    correlations['Vertical'] = vertical_corr

    # Flatten the 2D array to 1D for diagonal correlation
    flattened_img = img_array_gray.flatten()

    # Calculate diagonal correlation (top-left to bottom-right and bottom-left to top-right)
    diagonal_corr = (np.mean([scipy.stats.pearsonr(flattened_img[:-i], flattened_img[i:])[0] for i in range(1, img_array_gray.shape[0])]) +
                     np.mean([scipy.stats.pearsonr(flattened_img[i:], flattened_img[:-i])[0] for i in range(1, img_array_gray.shape[0])])) / 2
    correlations['Diagonal'] = diagonal_corr

    return correlations

# Example usage
original_image_path = 'hand.jpg'
encrypted_image_path = 'encrypted_image.png'

original_correlations = calculate_correlations(original_image_path)
encrypted_correlations = calculate_correlations(encrypted_image_path)

print("Original Image Correlations:")
for key, value in original_correlations.items():
    print(f"{key} Correlation: {value}")

print("\nEncrypted Image Correlations:")
for key, value in encrypted_correlations.items():
    print(f"{key} Correlation: {value}")
