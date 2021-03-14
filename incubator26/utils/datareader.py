import os
import json
import numpy as np
import cv2

class DataReader():

    def get_list_of_data(self, filepath):
        folder = os.listdir(filepath)
        data = [file for file in folder]

        return data

    def read_image(self, image_path):
        image = cv2.imread(image_path, 0)
        size = image.shape
        return image, size

    # def extract_nested_values(self, meta, list_of_dicts):
    #     if isinstance(meta, list):
    #         for sub_meta in meta:
    #             yield from self.extract_nested_values(sub_meta, list_of_dicts)
    #     elif isinstance(meta, dict):
    #         for key, value in meta.items():
    #             if key == "words":
    #                 list_of_dicts.append(value)
    #         for value in meta.values():
    #             yield from self.extract_nested_values(value, list_of_dicts)

    def get_bounding_boxes_from_json(self, json_filepath):
        with open(json_filepath) as file:
            meta = json.loads(file.read())
            bounding_boxes = np.ndarray(shape=(len(meta["form"]), 4))
            counter = 0
            for dictionary in meta["form"]:
                bounding_boxes[counter] = dictionary["box"]
                counter += 1
        return bounding_boxes