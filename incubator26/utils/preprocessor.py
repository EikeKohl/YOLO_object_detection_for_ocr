import numpy as np
import os
import cv2
from incubator26.utils.datareader import DataReader
import incubator26.yolo.data_preparation as dp


class PreProcessor:
    """
    A class to handle multiple preprocessing steps for images.

    ...

    Attributes
    ----------
    None

    Methods
    -------

    """

    def preprocess_image(self, image, target_size):
        """Reads, resizes and normalizes image data

        Additionally, another dimension is added to the input array.
        Please note, that no preprocessing steps will be needed in
        the tf.keras.preprocessing.image.ImageDataGenerator() class.

        Parameters
        ----------
        image : numpy array
            Image to be preprocessed.
        target_size : tuple
            Target shape of the preprocessed output (width, height).

        Returns
        -------
        numpy array
            The image after preprocessing.
        """

        image = self.filter_noise_from_image(image)
        resized_image = cv2.resize(image, target_size)
        image_data = np.array(resized_image, dtype="float32")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, -1)  # Add batch dimension.
        return image_data

    def filter_noise_from_image(self, image):
        """Applies additional preprocessing the image to improve data quality

        Please note, that this method is currently not used.

        Parameters
        ----------
        image : numpy array
           Image to be filtered.

        Returns
        -------
        numpy array
            Image after filtering / preprocessing.
        """

        #     blur = cv2.GaussianBlur(image,(5,5),0)
        #     image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        #     image = cv2.fastNlMeansDenoising(image, None, 10, 7, 15)

        return image

    def generate_train_test_data(
        self,
        input_shape,
        grid_size,
        train_img_folder,
        train_annotation_folder,
        test_img_folder,
        test_annotation_folder,
        anchor_boxes,
    ):
        """Generates training and testing data from two separate directories

        Parameters
        ----------
        input_shape : tuple
            Input shape of the model that uses the data: (number_of_channels, width, height).
        grid_size : int
            Size of each prediction square of the input image (number of pixel values per edge).
        train_img_folder : str
            Path to the folder containing the training images.
        train_annotation_folder : str
            Path to the folder containing the annotations to the training images.
        test_img_folder : str
            Path to the folder containing the testing images.
        test_annotation_folder : str
            Path to the folder containing the annotations to the testing images.
        anchor_boxes : list
            List of tuples containing the size of the anchor boxes: [(width, height)].

        Returns
        -------
        numpy ndarray
            X_train
        numpy ndarray
            X_test
        numpy ndarray
            Y_train
        numpy ndarray
            Y_test
        """

        # Setup reader and define parameters
        reader = DataReader()
        n_C, n_W, n_H = input_shape
        number_of_grids = int(n_W / grid_size)

        # Define train parameters and instantiate training ndarrays
        train_img_files = reader.get_list_of_data(train_img_folder)
        m = len(train_img_files)
        X_train = np.ndarray(shape=(m, n_H, n_W, n_C))
        Y_train = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

        for index, img in enumerate(train_img_files):
            # Read and preprocess image
            img_filepath = os.path.join(train_img_folder, img)
            image, size = reader.read_image(img_filepath)
            original_W = size[1]
            original_H = size[0]
            image = self.preprocess_image(target_size=(n_W, n_H), image=image)
            X_train[index] = image

            # Read and preprocess bounding boxes
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

            # Find best anchor box for each bounding box
            dp.find_best_anchor_box_with_IoU(
                bounding_box_meta=bounding_boxes_yolo_format, anchor_boxes=anchor_boxes
            )

            # Create true label tensor for image
            y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(
                bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                output_shape=(number_of_grids, number_of_grids, 5),
            )

            # Append Y_train with tensor
            Y_train[index] = y_tensor

        # Define train parameters and instantiate training ndarrays
        test_img_files = reader.get_list_of_data(test_img_folder)
        m = len(test_img_files)
        X_test = np.ndarray(shape=(m, n_H, n_W, n_C))
        Y_test = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

        for index, img in enumerate(test_img_files):
            # Read and preprocess image
            img_filepath = os.path.join(test_img_folder, img)
            image, size = reader.read_image(img_filepath)
            original_W = size[1]
            original_H = size[0]
            image = self.preprocess_image(target_size=(n_W, n_H), image=image)
            X_test[index] = image

            # Read and preprocess bounding boxes
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

            # Find best anchor box for each bounding box
            dp.find_best_anchor_box_with_IoU(
                bounding_box_meta=bounding_boxes_yolo_format, anchor_boxes=anchor_boxes
            )

            # Create true label tensor for image
            y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(
                bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                output_shape=(number_of_grids, number_of_grids, 5),
            )

            # Append Y_test with tensor
            Y_test[index] = y_tensor

        print("X_train shape: ", X_train.shape)
        print("Y_train shape: ", Y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("Y_test shape: ", Y_test.shape)

        return X_train, X_test, Y_train, Y_test
