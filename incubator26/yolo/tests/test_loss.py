from unittest import TestCase
import numpy as np
import keras.backend as K
from incubator26.yolo import loss


class Test(TestCase):
    def test_yolo_loss(self):
        np.random.seed(1)
        y_true = np.zeros((1, 3, 3, 2))
        y_true.fill(2)
        y_pred = np.zeros((1, 3, 3, 2))
        y_pred.fill(1)
        test = y_true - y_pred
        # print(K.sqrt(K.square(y_true)))
        # print(1-y_pred)
        print(K.sum(y_true))