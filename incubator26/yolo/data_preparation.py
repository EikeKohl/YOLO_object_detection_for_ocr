import numpy as np

def preprocess_bounding_boxes(bounding_boxes, src_size, target_size, grid_size):

    image_H, image_W = src_size
    target_H, target_W = target_size
    bounding_boxes_yolo_format = []

    # what's the relative size of one grid compared to the target image size in the YOLO output?
    grid_W_norm, grid_H_norm = grid_size / target_W, grid_size / target_H

    current_bounding_box = 0

    for box in bounding_boxes:
        # rescale box size to target size
        x1, y1, x2, y2 = box
        rescale_factor_H = target_H / image_H
        rescale_factor_W = target_W / image_W
        x1_rescaled = x1 * rescale_factor_W
        x2_rescaled = x2 * rescale_factor_W
        y1_rescaled = y1 * rescale_factor_H
        y2_rescaled = y2 * rescale_factor_H

        # what's the relative size of each box compared to the grid size?
        box_W = x2_rescaled - x1_rescaled
        box_H = y2_rescaled - y1_rescaled
        box_W_relative_to_grid_size = box_W / grid_size
        box_H_relative_to_grid_size = box_H / grid_size

        # norm the box size with image size to between 0-1
        x1_norm = x1_rescaled / target_W
        x2_norm = x2_rescaled / target_W
        y1_norm = y1_rescaled / target_H
        y2_norm = y2_rescaled / target_H

        # where is the center of the bouding box located?
        box_W_norm = x2_norm - x1_norm
        box_H_norm = y2_norm - y1_norm
        x_center_bounding_box = x1_norm + box_W_norm / 2
        y_center_bounding_box = y1_norm + box_H_norm / 2

        # In which grid is the center of the box located?
        x_grid_with_box_center = int(x_center_bounding_box / grid_W_norm)
        y_grid_with_box_center = int(y_center_bounding_box / grid_H_norm)

        # what is the position of the center of the box within the grid between (0,0) and (1,1)?

        x_box_in_grid = (x_center_bounding_box / grid_W_norm) - int(x_center_bounding_box / grid_W_norm)
        y_box_in_grid = (y_center_bounding_box / grid_H_norm) - int(y_center_bounding_box / grid_H_norm)

        # fill dictionary with relevant information

        bounding_boxes_yolo_format.append({"grid": [x_grid_with_box_center, y_grid_with_box_center],
                          "x": x_box_in_grid,
                          "y": y_box_in_grid,
                          "w": box_W_relative_to_grid_size,
                          "h": box_H_relative_to_grid_size
                          })

        current_bounding_box += 1

    return bounding_boxes_yolo_format

def find_best_anchor_box_with_IoU(bounding_box_meta: list, anchor_boxes: list):
    for box in bounding_box_meta:

        area_bounding_box = box["w"] * box["h"]
        IoU_list = []

        for anchor_box in anchor_boxes:
            IoU = min(box["w"], anchor_box[0]) * min(box["h"], anchor_box[1])
            IoU_list.append(IoU)

        box["anchor box"] = np.argmax(IoU_list)

    return bounding_box_meta

def convert_bounding_boxes_to_numpy_ndarray(bounding_boxes_list_of_dicts, output_shape):

    # output_shape = (30,30,5)
    y_true = np.zeros(output_shape)

    for bounding_box in bounding_boxes_list_of_dicts:
        x = bounding_box["x"]
        y = bounding_box["y"]
        w = bounding_box["w"]
        h = bounding_box["h"]
        grid_x = bounding_box["grid"][0]
        grid_y = bounding_box["grid"][1]
        anchor_box = bounding_box["anchor box"]

        offset = 5 * anchor_box

        y_true[grid_x, grid_y, 0 + offset] = 1  # response
        y_true[grid_x, grid_y, 1 + offset:5 + offset] = [x, y, w, h]

    return y_true