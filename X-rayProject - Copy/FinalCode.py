import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from Crypto.Cipher import AES
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats
import skimage.exposure as exposure
class ImageCryptographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Encryption/Decryption Tool")
        
        # Initialize encryption parameters
        seed = (0.1, 0.2)
        self.key = self.generate_key(seed, 1000)
        self.aes_key = self.key[:32]
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Create tabs
        self.encryption_tab = ttk.Frame(self.notebook)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.correlation_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.encryption_tab, text='Encryption/Decryption')
        # self.notebook.add(self.analysis_tab, text='Henon Map Analysis')
        self.notebook.add(self.correlation_tab, text='Correlation Analysis')
        
        self.setup_encryption_tab()
        self.setup_analysis_tab()
        self.setup_correlation_tab()

    def logistic_map(self, x, r):
        return r * x * (1 - x)

    def henon_map(self, x, y, a, b):
        return y + 1 - a * x ** 2, b * x

    def generate_key(self, seed, size):
        key = []
        x, y = seed
        for i in range(size):
            x, y = self.henon_map(x, y, 1.4, 0.3)
            key.append(int(self.logistic_map(x, 4.0) * 255) % 256)
        key_bytes = bytes(key)
        return key_bytes
        
    def setup_encryption_tab(self):
        # Input frame
        self.input_frame = tk.Frame(self.encryption_tab)
        self.input_frame.pack(pady=10)
        
        # Images frame
        self.images_frame = tk.Frame(self.encryption_tab)
        self.images_frame.pack(pady=10)
        
        # Create encryption tab widgets
        self.create_encryption_widgets()
        
    def setup_analysis_tab(self):
    # Create figure for Henon map analysis
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 15))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.analysis_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add button to update analysis
        self.update_analysis_btn = tk.Button(
            self.analysis_tab,
            text="Update Analysis",
            command=self.update_henon_analysis
        )
        self.update_analysis_btn.pack(pady=5)
    
    # Automatically generate plots when tab is created
        self.update_henon_analysis()
        
    def setup_correlation_tab(self):
        # Create frames for original and encrypted correlation values
        self.orig_corr_frame = tk.LabelFrame(self.correlation_tab, text="Original Image Correlations")
        self.orig_corr_frame.pack(pady=10, padx=10, fill="x")
        
        self.enc_corr_frame = tk.LabelFrame(self.correlation_tab, text="Encrypted Image Correlations")
        self.enc_corr_frame.pack(pady=10, padx=10, fill="x")
        
        # Labels for correlation values
        self.orig_corr_labels = {}
        self.enc_corr_labels = {}
        
        for direction in ['Horizontal', 'Vertical', 'Diagonal']:
            tk.Label(self.orig_corr_frame, text=f"{direction}:").pack(side=tk.LEFT, padx=5)
            self.orig_corr_labels[direction] = tk.Label(self.orig_corr_frame, text="N/A")
            self.orig_corr_labels[direction].pack(side=tk.LEFT, padx=5)
            
            tk.Label(self.enc_corr_frame, text=f"{direction}:").pack(side=tk.LEFT, padx=5)
            self.enc_corr_labels[direction] = tk.Label(self.enc_corr_frame, text="N/A")
            self.enc_corr_labels[direction].pack(side=tk.LEFT, padx=5)

    def create_encryption_widgets(self):
        # Upload button
        self.upload_btn = tk.Button(
            self.input_frame,
            text="Upload Image",
            command=self.upload_image,
            width=15
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Process button
        self.process_btn = tk.Button(
            self.input_frame,
            text="Encrypt/Decrypt",
            command=self.process_image,
            width=15,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Image labels and displays
        for i, label_text in enumerate(["Original Image", "Encrypted Image", "Decrypted Image"]):
            label = tk.Label(self.images_frame, text=label_text)
            label.grid(row=0, column=i, padx=10)
            
            display = tk.Label(self.images_frame)
            display.grid(row=1, column=i, padx=10)
            setattr(self, f"{label_text.lower().replace(' ', '_')}_display", display)

    def encrypt_image(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        height, width, _ = img_array.shape
        key_index = 0
        
        # XOR encryption
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    img_array[i][j][k] ^= int(self.key[key_index])
                    key_index = (key_index + 1) % len(self.key)
                    
        img_bytes = img_array.tobytes()
        cipher = AES.new(self.aes_key, AES.MODE_ECB)
        encrypted_bytes = cipher.encrypt(img_bytes)
        encrypted_img = Image.frombytes(img.mode, img.size, encrypted_bytes)
        return encrypted_img

    def decrypt_image(self, encrypted_image):
        img_bytes = encrypted_image.tobytes()
        cipher = AES.new(self.aes_key, AES.MODE_ECB)
        decrypted_bytes = cipher.decrypt(img_bytes)
        decrypted_img = Image.frombytes(encrypted_image.mode, encrypted_image.size, decrypted_bytes)

        img_array = np.array(decrypted_img)
        height, width, _ = img_array.shape
        key_index = 0
        
        # XOR decryption
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    img_array[i][j][k] ^= int(self.key[key_index])
                    key_index = (key_index + 1) % len(self.key)

        final_img = Image.fromarray(img_array)
        return final_img

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff")
            ]
        )
        if file_path:
            try:
                # Load and display original image
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, self.original_image_display)
                self.process_btn.config(state=tk.NORMAL)
                self.file_path = file_path
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def display_image(self, image, label, size=(200, 200)):
        # Resize image for display
        display_image = image.copy()
        display_image.thumbnail(size)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(display_image)
        
        # Update label
        label.config(image=photo)
        label.image = photo  # Keep a reference

    def henon_map_phase_plot(self, x, y, a, b, iterations):
        x_vals, y_vals = [], []
        for _ in range(iterations):
            x_vals.append(x)
            y_vals.append(y)
            x, y = self.henon_map(x, y, a, b)
        return x_vals, y_vals

    def henon_map_bifurcation(self, a_values, iterations, points_per_iteration):
        xy_values = []
        x, y = 0.1, 0.2
        for a in a_values:
            for _ in range(iterations):
                x, y = self.henon_map(x, y, a, 0.3)
            for _ in range(points_per_iteration):
                x, y = self.henon_map(x, y, a, 0.3)
                xy_values.append((a, x))
        return xy_values

    def henon_map_lyapunov_exponent(self, x, y, a, b, iterations, delta=1e-9):
        sum_log = 0.0
        lyapunov_vals = []
        for _ in range(iterations):
            x_prime, y_prime = self.henon_map(x, y, a, b)
            partial_derivative = (self.henon_map(x + delta, y, a, b)[0] - x_prime) / delta
            sum_log += np.log(np.abs(partial_derivative))
            x, y = x_prime, y_prime
            lyapunov_vals.append(sum_log / (_ + 1))
        return lyapunov_vals

    def update_henon_analysis(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        a_value = 1.4
        a_values = np.linspace(0.8, 1.4, 1000)
        iterations = 10000
        points_per_iteration = 100
        
        # Phase plot
        x_vals, y_vals = self.henon_map_phase_plot(0.1, 0.2, a_value, 0.3, iterations)
        self.ax1.plot(x_vals, y_vals, '.', markersize=1)
        self.ax1.set_title(f"Henon Map Phase Plot (a={a_value}, b=0.3)")
        self.ax1.set_xlabel("X")
        self.ax1.set_ylabel("Y")
        
        # Bifurcation diagram
        bifurcation_data = self.henon_map_bifurcation(a_values, iterations, points_per_iteration)
        self.ax2.scatter(*zip(*bifurcation_data), s=0.1, marker='.', color='red')
        self.ax2.set_title('Henon Map Bifurcation Diagram')
        self.ax2.set_xlabel('a')
        self.ax2.set_ylabel('x')
        
        # Lyapunov exponent
        lyapunov_vals = self.henon_map_lyapunov_exponent(0.1, 0.2, a_value, 0.3, 1000)
        self.ax3.plot(range(1, len(lyapunov_vals) + 1), lyapunov_vals)
        self.ax3.set_title(f"Henon Map Lyapunov Exponent (a={a_value}, b=0.3)")
        self.ax3.set_xlabel("Iterations")
        self.ax3.set_ylabel("Lyapunov Exponent")
        
        plt.tight_layout()
        self.canvas.draw()

    def calculate_correlations(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        correlations = {}

        if len(img_array.shape) == 3:
            img_gray = img.convert('L')
            img_array_gray = np.array(img_gray)
        else:
            img_array_gray = img_array

        def safe_pearsonr(arr1, arr2):
            if np.std(arr1) == 0 or np.std(arr2) == 0:
                return 0
            return scipy.stats.pearsonr(arr1, arr2)[0]

        horizontal_corr = np.mean([safe_pearsonr(img_array_gray[i], img_array_gray[i+1]) 
                             for i in range(img_array_gray.shape[0] - 1)])
        correlations['Horizontal'] = horizontal_corr

        vertical_corr = np.mean([safe_pearsonr(img_array_gray[:, j], img_array_gray[:, j+1]) 
                           for j in range(img_array_gray.shape[1] - 1)])
        correlations['Vertical'] = vertical_corr

        flattened_img = img_array_gray.flatten()
        diagonal_corr = (np.mean([safe_pearsonr(flattened_img[:-i], flattened_img[i:]) 
                            for i in range(1, img_array_gray.shape[0])]) +
                    np.mean([safe_pearsonr(flattened_img[i:], flattened_img[:-i]) 
                           for i in range(1, img_array_gray.shape[0])])) / 2
        correlations['Diagonal'] = diagonal_corr

        return correlations

    def update_correlation_analysis(self):
        if hasattr(self, 'file_path'):
            # Calculate correlations for original image
            orig_corr = self.calculate_correlations(self.file_path)
            for direction, value in orig_corr.items():
                self.orig_corr_labels[direction].config(text=f"{value:.4f}")

            # Calculate correlations for encrypted image
            enc_corr = self.calculate_correlations('encrypted_image.png')
            for direction, value in enc_corr.items():
                self.enc_corr_labels[direction].config(text=f"{value:.4f}")

    def process_image(self):
        try:
        # Encrypt the image
            self.encrypted_image = self.encrypt_image(self.file_path)
            self.encrypted_image.save('encrypted_image.png')
        
        # Display encrypted image
            self.display_image(self.encrypted_image, self.encrypted_image_display)
        
        # Decrypt the image
            self.decrypted_image = self.decrypt_image(self.encrypted_image)
            self.decrypted_image.save('decrypted_image.png')
        
        # Display decrypted image
            self.display_image(self.decrypted_image, self.decrypted_image_display)
        
        # Update correlation analysis
            self.update_correlation_analysis()
        
        # Perform statistical analysis
            self.statistical_analysis()
        
            messagebox.showinfo("Success", "Images processed and analysis updated!")
        
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

    def calculate_entropy_and_plot_histogram(self, image, title, index):
    # Convert image to grayscale
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
        self.ax[index-1].imshow(img_array, cmap='gray')
        self.ax[index-1].set_title(title)
        self.ax[index-1].axis('off')
    
        self.ax[index+2].hist(img_array.ravel(), bins=256, range=(0, 256), color='blue')
        self.ax[index+2].set_title(f'{title} - Histogram')
        self.ax[index+2].set_xlabel('Pixel Intensity')
        self.ax[index+2].set_ylabel('Frequency')

    def statistical_analysis(self):

    # Create a figure with 6 subplots (3 images, 3 histograms)
        self.fig_stat, self.ax = plt.subplots(2, 3, figsize=(15, 10))
        self.ax = self.ax.ravel()  # Flatten axes array
    
    # Images to analyze
        images = [
            self.original_image, 
            self.encrypted_image, 
            self.decrypted_image
        ]
        titles = ['Original Image', 'Encrypted Image', 'Decrypted Image']
    
    # Perform analysis for each image
        for index, (image, title) in enumerate(zip(images, titles), 1):
            self.calculate_entropy_and_plot_histogram(image, title, index)
    
        plt.tight_layout()
        plt.show()

# In the __init__ or setup method of your ImageCryptographyApp class, add:


def main():
    root = tk.Tk()
    app = ImageCryptographyApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()