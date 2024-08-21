from flask import Flask, request, jsonify
from flask_cors import CORS  # Handle CORS for cross-origin requests
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import random
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load the pre-trained U2NET model
U2NET_MODEL = load_model('C:/Users/smart/Downloads/GP_Materials/ImageRecoloring.h5', compile=False)

def preprocess_image(image, size):
    """Preprocess the image for the model."""
    img = image.convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img)
    return np.expand_dims(img_array, axis=0)

def apply_mask(image, mask):
    """Apply a binary mask to the image."""
    mask = (mask > 0.5).astype(np.uint8) * 255
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def mcts_generate_colors(base_colors, n_new_colors, n_iterations=1000, variation_range=60):
    """Generate new colors using Monte Carlo Tree Search."""
    best_new_colors = []
    for _ in range(n_new_colors):
        best_color = None
        best_score = float('inf')
        for _ in range(n_iterations):
            base_color = random.choice(base_colors)
            new_color = base_color + np.random.randint(-variation_range, variation_range, size=3)
            new_color = np.clip(new_color, 0, 255)
            score = np.min(np.linalg.norm(base_colors - new_color, axis=1))
            if score < best_score:
                best_score = score
                best_color = new_color
        best_new_colors.append(best_color)
    return np.array(best_new_colors)

def rgb_to_hex(rgb):
    """Convert RGB to HEX color."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def plot_extended_palette(image, n_colors=5, n_new_colors=3):
    """Generate and return an extended color palette from the image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image_rgb.reshape(-1, 3)
    mask = (pixels.sum(axis=1) != 0)
    pixels = pixels[mask]

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    new_colors = mcts_generate_colors(colors, n_new_colors=n_new_colors)
    all_colors = np.vstack((colors, new_colors))

    return [rgb_to_hex(color) for color in all_colors]

@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file found in the request'}), 400

        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read()))

        image_size = 256
        input_array = preprocess_image(image, image_size)
        y_pred = U2NET_MODEL.predict(input_array)
        predicted_mask = y_pred[0]
        predicted_mask = cv2.resize(predicted_mask, (image_size, image_size))
        original_image = np.array(image.resize((image_size, image_size)))

        focal_object = apply_mask(original_image, predicted_mask)
        extended_palette = plot_extended_palette(focal_object, n_colors=5, n_new_colors=3)

        return jsonify({
            'extended_color_palette': extended_palette,
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5150, debug=True)
