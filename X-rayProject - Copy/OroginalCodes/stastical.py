import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure

def calculate_entropy_and_plot_histogram(image, title, index):
    img_gray = image.convert('L')
    img_array = np.array(img_gray)
    
    # Calculate entropy
    hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
    prob_distribution = hist / np.sum(hist)
    entropy = -np.sum([p * np.log2(p) for p in prob_distribution if p > 0])
    print(f"{title} - Entropy: {entropy:.2f}")
    
    # Apply adaptive histogram equalization only to the encrypted image
    if title == 'Encrypted Image':
        img_eq = exposure.equalize_adapthist(img_array)
        img_eq = (img_eq * 255).astype(np.uint8)
        img_array = img_eq
    
    # Plot image and histogram
    plt.subplot(3, 3, index)
    plt.imshow(img_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    plt.subplot(3, 3, index + 3)
    plt.hist(img_array.ravel(), bins=256, range=(0, 256), color='blue')  # Change color here
    plt.title(f'{title} - Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

# Images
images = ['hand.jpg', 'encrypted_image.png', 'decrypted_image.png']
titles = ['Original Image', 'Encrypted Image', 'Decrypted Image']

plt.figure(figsize=(15, 10))

for index, (image_path, title) in enumerate(zip(images, titles), 1):
    img = Image.open(image_path)
    calculate_entropy_and_plot_histogram(img, title, index)

plt.tight_layout()
plt.show()
