#import base64
import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

# Crear carpetas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def apply_morphological_operations(image, filename):
    results = {}
    kernel_sizes = [15, 37, 60]  # Tamaños de máscaras aproximados

    for size in kernel_sizes:
        kernel = np.ones((size, size), np.uint8)

        # a) Erosión
        erosion = cv2.erode(image, kernel, iterations=1)
        erosion_path = os.path.join(app.config['PROCESSED_FOLDER'], f'erosion_{size}_{filename}')
        cv2.imwrite(erosion_path, erosion)
        results[f'erosion_{size}'] = erosion_path

        # b) Dilatación
        dilation = cv2.dilate(image, kernel, iterations=1)
        dilation_path = os.path.join(app.config['PROCESSED_FOLDER'], f'dilation_{size}_{filename}')
        cv2.imwrite(dilation_path, dilation)
        results[f'dilation_{size}'] = dilation_path

        # c) Top Hat
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        top_hat_path = os.path.join(app.config['PROCESSED_FOLDER'], f'top_hat_{size}_{filename}')
        cv2.imwrite(top_hat_path, top_hat)
        results[f'top_hat_{size}'] = top_hat_path

        # d) Black Hat
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        black_hat_path = os.path.join(app.config['PROCESSED_FOLDER'], f'black_hat_{size}_{filename}')
        cv2.imwrite(black_hat_path, black_hat)
        results[f'black_hat_{size}'] = black_hat_path

        # e) Imagen Original + (Top Hat - Black Hat)
        enhanced_image = cv2.add(image, cv2.subtract(top_hat, black_hat))
        enhanced_path = os.path.join(app.config['PROCESSED_FOLDER'], f'enhanced_{size}_{filename}')
        cv2.imwrite(enhanced_path, enhanced_image)
        results[f'enhanced_{size}'] = enhanced_path

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')
    if len(files) > 3:
        return "Error: Solo puedes subir hasta 3 imágenes"

    uploaded_images = []
    for file in files:
        if file and allowed_file(file.filename):
            # Guardar la imagen subida en la carpeta de uploads
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Leer la imagen desde la ruta guardada
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            uploaded_images.append({'image': image, 'filename': filename})

    # Procesar cada imagen subida y almacenar las versiones procesadas
    processed_images = []
    for image in uploaded_images:
        processed_paths = apply_morphological_operations(image['image'], image['filename'])
        processed_paths['person1_virus'] = image['filename']
        processed_images.append(processed_paths)

    # Guardar las imágenes procesadas en la sesión (en este caso, usando una variable global temporal)
    global processed_images_data
    processed_images_data = processed_images

    return redirect(url_for('show_processed_images'))

@app.route('/processed_images', methods=['GET'])
def show_processed_images():
    return render_template('process_images.html', processed_images=processed_images_data)

if __name__ == '__main__':
    app.run(debug=True)
