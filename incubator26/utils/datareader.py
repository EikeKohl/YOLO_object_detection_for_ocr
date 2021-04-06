import os
import json
import numpy as np
import cv2


class DataReader:
    """
    A class to read images and bounding boxes.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    get_list_of_data(filepath)
        Creates a list of all files in a directory.

    read_image(image_path)
        Reads an image as a cv2 object.

    get_bounding_boxes_from_json(json_filepath):
        Reads json metafile for images and gets bounding boxes.
    """

    def get_list_of_data(self, filepath):
        """Creates a list of all files in a directory.

        Parameters
        ----------
        filepath : str
            The directory path of the files to be put in the list.

        Returns
        -------
        list
            A list of all files in the specified directory
        """

        folder = os.listdir(filepath)
        data = [file for file in folder]

        return data

    def read_image(self, image_path):
        """Reads an image as a cv2 object.

        Parameters
        ----------
        image_path : str
            The image's filepath.

        Returns
        -------
        cv2 object
            The image
        list
            Size of the image: [height, width]
        """

        image = cv2.imread(image_path, 0)
        size = image.shape
        return image, size

    def get_bounding_boxes_from_json(self, json_filepath):
        """Reads json metafile for images and gets bounding boxes

        This function was specifically designed for a kaggle dataset,
        it may require some adjustment to be applied to new datasets.
        Parameters
        ----------
        json_filepath : str
            Filepath to the json file containing the metadata

        Returns
        -------
        list
            List of dictionaries containing bounding boxes
        """

        with open(json_filepath) as file:
            meta = json.loads(file.read())
            bounding_boxes = np.ndarray(shape=(len(meta["form"]), 4))
            counter = 0
            for dictionary in meta["form"]:
                bounding_boxes[counter] = dictionary["box"]
                counter += 1
        return bounding_boxes
