import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Letter_model:
    # Better Use h5
    def __init__(self, path):
        self.path = path
        self.encoder = LabelEncoder()
        self.classes = dict({0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
                            18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'})

    def init_model(self):
        try:
            print('Starting Model...')
            self.model = models.load_model(self.path, compile=False)
        except Exception as e:
            print('An error ocurred when tried to load the model:', e)

    def summary(self):
        return self.model.summary()

    def _img_transform(self, image_data):
        img = Image.open(image_data)
        img = img.resize((28, 28))  # Resize To img_size
        img = img.convert('L')
        img = np.array(img, dtype=np.float32)
        img = img / 255.00  # Normalize
        # Ensure the image has the correct shape (28, 28, 1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 28, 28, 1)

        return img

    def predict(self, img_path):

        img_converted = self._img_transform(image_data=img_path)
        prediction = self.model.predict([img_converted], verbose=0)
        top_3_in = tf.argsort(prediction[0], direction='DESCENDING')
        top_3_in = top_3_in.numpy()[:3].tolist()
        top_3_proba = tf.gather(prediction[0], top_3_in)
        prediction = list(map(lambda x: self.classes[x], top_3_in))
        return prediction, top_3_proba
