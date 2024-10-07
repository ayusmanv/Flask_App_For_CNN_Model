import os
from flask import Flask, request, render_template, redirect
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
model_path = 'models/cnn.h5'

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']

    if file.filename == '':
        return redirect('/')

    # Save the file locally
    filepath = os.path.join(file.filename)
    file.save(filepath)

    # Load image for prediction
    predictor = load_model(model_path)
    test_image = utils.load_img(filepath, target_size=(64, 64))
    test_image = utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    predictions = predictor.predict(test_image)
    #print(train_generator.class_indices)
    #train_generator.class_indices
    #print(predictions)
    os.remove(filepath)
    if predictions[0][0] == 1:
        return render_template('result.html', results = f"It's a dog")
    else:
        return render_template('result.html', results = f"It's a cat")

    # Remove the saved file (optional)



if __name__ == '__main__':
    # Create the uploads folder if not exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(host='0.0.0.0', port=5000)

