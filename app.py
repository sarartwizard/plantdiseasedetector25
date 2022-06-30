from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image


model = tf.keras.models.load_model("C:/Users/nadou/OneDrive/Documents/disease/disease_plant_transfer.h5")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']

    image_path = "C:/Users/nadou/PycharmProjects/plant_prediction/images/" + imagefile.filename
    imagefile.save(image_path)



    class_names = ['Cryptogamique',
                   'Fumagine',
                   'Healthy',
                   'Maladie bactériennes',
                   'Mildiou',
                   'Septoriose']
    image = np.array(
        Image.open(imagefile).convert("RGB").resize((224, 224))  # image resizing
    )

    image = image / 255.0  # normalize the image in 0 to 1 range
    img_array = tf.expand_dims(image, 0)

    img_array.numpy().astype('uint8')
    predictions = model.predict(img_array)

    classification = class_names[np.argmax(predictions)]

    print(image_path)

    return render_template('index.html', prediction=classification)





if __name__ == "__main__":
    app.run(port=3000, debug=True)