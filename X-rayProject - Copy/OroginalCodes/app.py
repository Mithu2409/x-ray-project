from flask import Flask, render_template, request, send_file
import os
from project import encrypt, decrypt, generate_key
from plot import henon_map_phase_plot, henon_map_bifurcation, henon_map_lyapunov_exponent
from stastical import calculate_entropy_and_plot_histogram
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Seed and AES key for encryption/decryption
seed = (0.1, 0.2)
key = generate_key(seed, 1000)
aes_key = key[:32]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded!", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file!", 400
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform encryption
    encrypted_path = os.path.join(app.config['OUTPUT_FOLDER'], 'encrypted_image.png')
    encrypt(file_path, key, aes_key)

    # Perform decryption
    decrypted_path = os.path.join(app.config['OUTPUT_FOLDER'], 'decrypted_image.png')
    decrypt(encrypted_path, key, aes_key)

    return render_template('process.html', 
                           original=file.filename, 
                           encrypted='encrypted_image.png', 
                           decrypted='decrypted_image.png')

@app.route('/graph')
def plot_graphs():
    # Generate Henon Map phase plot
    x_vals, y_vals = henon_map_phase_plot(0.1, 0.2, 1.4, 0.3, 1000)
    plt.figure()
    plt.plot(x_vals, y_vals, '.', markersize=1)
    phase_plot_path = os.path.join(app.config['OUTPUT_FOLDER'], 'phase_plot.png')
    plt.savefig(phase_plot_path)
    plt.close()

    return send_file(phase_plot_path, mimetype='image/png')

@app.route('/stats')
def stats():
    # Perform statistical analysis
    images = [
        os.path.join(app.config['UPLOAD_FOLDER'], 'skull.jpg'),
        os.path.join(app.config['OUTPUT_FOLDER'], 'encrypted_image.png'),
        os.path.join(app.config['OUTPUT_FOLDER'], 'decrypted_image.png')
    ]
    plt.figure(figsize=(15, 10))
    for index, image_path in enumerate(images, 1):
        img = Image.open(image_path)
        calculate_entropy_and_plot_histogram(img, f"Image {index}", index)
    
    stats_path = os.path.join(app.config['OUTPUT_FOLDER'], 'stats.png')
    plt.savefig(stats_path)
    plt.close()

    return send_file(stats_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
