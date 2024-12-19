from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import numpy as np
import pymupdf  # PyMuPDF
from PIL import Image
import io

app = Flask(__name__)
base_dir = os.path.abspath(os.path.dirname(__file__))
upload_folder = os.path.join(base_dir, 'uploads')
processed_folder = os.path.join(base_dir, 'processed')

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['PROCESSED_FOLDER'] = processed_folder

os.makedirs(upload_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

@app.context_processor
def utility_processor():
    return dict(zip=zip)

# Functions for image analysis and processing
def calculate_brightness(image):
    return np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2])

def calculate_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return l.std()

def calculate_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def process_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Routes for the Flask application
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith('.pdf'):
            original_images, processed_images, original_params, processed_params, pdf_path = process_pdf(file_path)
            original_images = [os.path.basename(img_path) for img_path in original_images]
            processed_images = [os.path.basename(img_path) for img_path in processed_images]
            return render_template('review.html', original_images=original_images, processed_images=processed_images, original_params=original_params, processed_params=processed_params, pdf_path=file_path)
        else:
            processed_file_path, original_params, processed_params, original_contrast_ratio, processed_contrast_ratio = process_image_file(file.filename)
            return render_template('index.html',
                                   original_image_url=url_for('uploaded_file', filename=file.filename),
                                   image_url=url_for('processed_file', filename=processed_file_path),
                                   original_params=original_params,
                                   processed_params=processed_params,
                                   original_score=accessibility_score(original_contrast_ratio),
                                   processed_score=accessibility_score(processed_contrast_ratio))
    return redirect(url_for('home'))

def process_image_file(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)

    original_params = {
        'brightness': calculate_brightness(image),
        'contrast': calculate_contrast(image),
        'sharpness': calculate_sharpness(image)
    }

    enhanced_image = process_image(image)

    processed_params = {
        'brightness': calculate_brightness(enhanced_image),
        'contrast': calculate_contrast(enhanced_image),
        'sharpness': calculate_sharpness(enhanced_image)
    }

    processed_file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(processed_file_path, enhanced_image)

    return filename, original_params, processed_params, original_params['contrast'], processed_params['contrast']

def process_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    original_images = []
    processed_images = []
    original_params = []
    processed_params = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))
            original_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f"original_{page_num}_{img_index}.png")
            image.save(original_image_path)
            original_images.append(original_image_path)

            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            enhanced_image_cv = process_image(image_cv)
            processed_image = Image.fromarray(cv2.cvtColor(enhanced_image_cv, cv2.COLOR_BGR2RGB))
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{page_num}_{img_index}.png")
            processed_image.save(processed_image_path)
            processed_images.append(processed_image_path)

            original_params.append({
                'brightness': calculate_brightness(image_cv),
                'contrast': calculate_contrast(image_cv),
                'sharpness': calculate_sharpness(image_cv)
            })

            processed_params.append({
                'brightness': calculate_brightness(enhanced_image_cv),
                'contrast': calculate_contrast(enhanced_image_cv),
                'sharpness': calculate_sharpness(enhanced_image_cv)
            })

    return original_images, processed_images, original_params, processed_params, pdf_path

@app.route('/confirm', methods=['POST'])
def confirm():
    processed_images = request.form.getlist('processed_images')
    pdf_path = request.form['pdf_path']

    doc = pymupdf.open(pdf_path)
    for page_num, img_path in enumerate(processed_images):
        page = doc.load_page(page_num)
        xref = page.get_images(full=True)[0][0]
        img_path_full = os.path.join(app.config['PROCESSED_FOLDER'], img_path)
        with open(img_path_full, 'rb') as img_file:
            image_bytes = img_file.read()

        img_buffer = io.BytesIO(image_bytes)
        image = Image.open(img_buffer)
        img_buffer.seek(0)
        page.replace_image(xref, stream=img_buffer)

    processed_pdf_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(pdf_path))
    doc.save(processed_pdf_path)

    return send_from_directory(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(pdf_path))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
