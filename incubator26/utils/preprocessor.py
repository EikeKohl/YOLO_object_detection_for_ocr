import numpy as np
import os
import cv2
from incubator26.utils.datareader import DataReader
import incubator26.yolo.data_preparation as dp


class PreProcessor:
    def preprocess_image(self, image, target_size):
        image = self.filter_noise_from_image(image)
        resized_image = cv2.resize(image, target_size)
        image_data = np.array(resized_image, dtype="float32")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, -1)  # Add batch dimension.
        return image_data

    def filter_noise_from_image(self, image):
        #     blur = cv2.GaussianBlur(image,(5,5),0)
        #     image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #     image = cv2.fastNlMeansDenoising(image, None, 10, 7, 15)
        return image

    def generate_train_test_data(
        self,
        input_shape: tuple,
        grid_size: int,
        train_img_folder: str,
        train_annotation_folder: str,
        test_img_folder: str,
        test_annotation_folder: str,
        anchor_boxes: list
    ):

        reader = DataReader()
        train_img_files = reader.get_list_of_data(train_img_folder)
        m = len(train_img_files)
        n_C, n_W, n_H = input_shape
        number_of_grids = int(n_W / grid_size)

        X_train = np.ndarray(shape=(m, n_H, n_W, n_C))
        Y_train = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

        all_bounding_boxes = []

        for index, img in enumerate(train_img_files):
            img_filepath = os.path.join(train_img_folder, img)
            image, size = reader.read_image(img_filepath)
            original_W = size[1]
            original_H = size[0]

            image = self.preprocess_image(target_size=(n_W, n_H), image=image)
            X_train[index] = image

            json_filepath = os.path.join(
                train_annotation_folder, img.split(".")[0] + ".json"
            )
            bounding_boxes = reader.get_bounding_boxes_from_json(json_filepath)
            bounding_boxes_yolo_format = dp.preprocess_bounding_boxes(
                bounding_boxes=bounding_boxes,
                src_size=(original_H, original_W),
                target_size=(n_H, n_W),
                grid_size=grid_size,
            )
            dp.find_best_anchor_box_with_IoU(
                bounding_box_meta=bounding_boxes_yolo_format, anchor_boxes=anchor_boxes
            )

            y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(
                bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                output_shape=(number_of_grids, number_of_grids, 5),
            )

            Y_train[index] = y_tensor

        test_img_files = reader.get_list_of_data(test_img_folder)
        m = len(test_img_files)

        X_test = np.ndarray(shape=(m, n_H, n_W, n_C))
        Y_test = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

        all_bounding_boxes = []

        for index, img in enumerate(test_img_files):
            img_filepath = os.path.join(test_img_folder, img)
            image, size = reader.read_image(img_filepath)
            original_W = size[1]
            original_H = size[0]

            image = self.preprocess_image(target_size=(n_W, n_H), image=image)
            X_test[index] = image

            json_filepath = os.path.join(
                test_annotation_folder, img.split(".")[0] + ".json"
            )
            bounding_boxes = reader.get_bounding_boxes_from_json(json_filepath)
            bounding_boxes_yolo_format = dp.preprocess_bounding_boxes(
                bounding_boxes=bounding_boxes,
                src_size=(original_H, original_W),
                target_size=(n_H, n_W),
                grid_size=grid_size,
            )
            dp.find_best_anchor_box_with_IoU(
                bounding_box_meta=bounding_boxes_yolo_format, anchor_boxes=anchor_boxes
            )

            y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(
                bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                output_shape=(number_of_grids, number_of_grids, 5),
            )

            Y_test[index] = y_tensor

        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("Y_test shape: ", Y_test.shape)

        return X_train, X_test, Y_train, Y_test
