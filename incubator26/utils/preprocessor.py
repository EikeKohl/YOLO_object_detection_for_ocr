import numpy as np
import cv2

class PreProcessor():

    def preprocess_image(self, image, target_size):
        image = self.filter_noise_from_image(image)
        resized_image = cv2.resize(image, target_size)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, -1)  # Add batch dimension.
        return image_data

    def filter_noise_from_image(self, image):
        #     blur = cv2.GaussianBlur(image,(5,5),0)
        #     image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #     image = cv2.fastNlMeansDenoising(image, None, 10, 7, 15)
        return image

