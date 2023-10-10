import os
from flask import Flask, request, render_template, url_for
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained models
benign_model = tf.keras.models.load_model(os.path.join('models', 'benign_model.h5'))
malignant_model = tf.keras.models.load_model(os.path.join('models', 'malignant_model.h5'))
img_height = 180
img_width = 180
class_names = ['benign', 'malignant']
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def predict_image(file, model):
    image = Image.open(file)
    image_array = tf.keras.utils.img_to_array(image.resize((img_height, img_width)))
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    return class_names[np.argmax(score)], round(100 * np.max(score), 2)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    image_path = None
    image_filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('predict.html', error='No selected file')

        if file:
            image_filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_image" + image_filename)
            file.save(image_path)

            if np.argmax(malignant_model.predict(tf.keras.utils.img_to_array(Image.open(image_path).resize((img_height, img_width))).reshape(1, img_height, img_width, 3))) == 1:
                prediction = predict_image(image_path, malignant_model)
            else:
                prediction = predict_image(image_path, benign_model)

    
    return render_template('predict.html', prediction=prediction, image_path=image_path)  # Use image_path here

if __name__ == '__main__':
    app.run(debug=True)
