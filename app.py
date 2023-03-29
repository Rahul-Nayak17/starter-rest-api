from keras.utils import load_img, img_to_array
from PIL import Image
import json
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow import keras
import numpy as np

# from keras.preprocessing.image import
app = Flask(__name__)

# Load the pre-trained model

model = keras.models.load_model('model_yt_vgg16.h5')


@app.route('/predict', methods=['POST'])
def predict():
    class_names = ['amruthballi',
                   'brahmi',
                   'hipli',
                   'parijatha']

    # get the file from POST
    file = request.files['file']

    # print(file)
    image = Image.open(file.stream)

    #image preprocessing
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Pass the image through the model to get predictions
    predictions = model.predict(image)
    pred_label = np.argmax(predictions, axis=1)  # We take the highest probability
    result = class_names[pred_label[0]]
    return jsonify({'result': json.dumps(str(result))})


if __name__ == "__main__":
    app.run(debug=True)



