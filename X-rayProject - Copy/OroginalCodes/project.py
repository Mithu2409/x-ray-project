import numpy as np
from PIL import Image
# from Cryptodome.Cipher import AES
from Crypto.Cipher import AES

def logistic_map(x, r):
    return r * x * (1 - x)

def henon_map(x, y, a, b):
    return y + 1 - a * x ** 2, b * x

def generate_key(seed, size):
    key = []
    x, y = seed
    for i in range(size):
        x, y = henon_map(x, y, 1.4, 0.3)
        key.append(int(logistic_map(x, 4.0) * 255) % 256)
    key_bytes = bytes(key)
    return key_bytes

def encrypt(image_path, key, aes_key):
    img = Image.open(image_path)
    img_array = np.array(img)
   # print(img_array)
    #len(img_array)
    height, width, _ = img_array.shape
    key_index = 0
    for i in range(height):
        for j in range(width):
            for k in range(3):
                img_array[i][j][k] ^= int(key[key_index])
                key_index = (key_index + 1) % len(key)
    img_bytes = img_array.tobytes()
   # print(img_bytes)
    cipher = AES.new(aes_key, AES.MODE_ECB)
    encrypted_bytes = cipher.encrypt(img_bytes)
    encrypted_img = Image.frombytes(img.mode, img.size, encrypted_bytes)
    encrypted_img.save('encrypted_image.png')

def decrypt(encrypted_image_path, key, aes_key):
    img = Image.open(encrypted_image_path)
    img_bytes = img.tobytes()
    cipher = AES.new(aes_key, AES.MODE_ECB)
    decrypted_bytes = cipher.decrypt(img_bytes)
    decrypted_img = Image.frombytes(img.mode, img.size, decrypted_bytes)

    img_array = np.array(decrypted_img)
    height, width, _ = img_array.shape
    key_index = 0
    for i in range(height):
        for j in range(width):
            for k in range(3):
                img_array[i][j][k] ^= int(key[key_index])
                key_index = (key_index + 1) % len(key)

    final_img = Image.fromarray(img_array)
    final_img.save('decrypted_image.png')

# Example usage
seed = (0.1, 0.2)
key = generate_key(seed, 1000)
aes_key = key[:32]  # Use the first 32 bytes of the key as AES key

encrypt('02.jpg', key, aes_key)
decrypt('encrypted_image.png', key, aes_key)