import numpy as np
import keras.backend as K

"""
Intersection_over_Union() muss noch auf Tensoren angepasst werden.
"""


def Intersection_over_Union(y_true, y_pred):

    # define variables from input tensors
    y_pred_x = y_pred[4, 3, 1]
    y_pred_y = y_pred[4, 3, 2]
    y_pred_w = y_pred[4, 3, 3]
    y_pred_h = y_pred[4, 3, 4]

    y_true_x = y_true[3, 4, 1]
    y_true_y = y_true[3, 4, 2]
    y_true_w = y_true[3, 4, 3]
    y_true_h = y_true[3, 4, 4]

    y_pred_left_edge = y_pred_x - y_pred_w / 2
    y_pred_right_edge = y_pred_x + y_pred_w / 2
    y_pred_upper_edge = y_pred_y - y_pred_h / 2
    y_pred_lower_edge = y_pred_y + y_pred_h / 2

    y_true_left_edge = y_true_x - y_true_w / 2
    y_true_right_edge = y_true_x + y_true_w / 2
    y_true_upper_edge = y_true_y - y_true_h / 2
    y_true_lower_edge = y_true_y + y_true_h / 2

    left_right_out = False
    upper_lower_out = False

    if y_pred_right_edge < y_true_left_edge or y_pred_left_edge > y_true_right_edge:
        left_right_out = True

    if y_pred_upper_edge > y_true_lower_edge or y_pred_lower_edge < y_true_upper_edge:
        upper_lower_out = True

    if left_right_out or upper_lower_out:
        IoU = 0
        return IoU

    intersect_left_edge = max(y_pred_left_edge, y_true_left_edge)
    intersect_right_edge = min(y_pred_right_edge, y_true_right_edge)
    intersect_upper_edge = min(y_pred_upper_edge, y_true_upper_edge)
    intersect_lower_edge = max(y_pred_lower_edge, y_true_lower_edge)

    intersect_h = intersect_lower_edge - intersect_upper_edge
    intersect_w = intersect_right_edge - intersect_left_edge

    intersect_area = intersect_h * intersect_w
    pred_area = y_pred_w * y_pred_h
    true_area = y_true_w * y_true_h
    union_area = pred_area + true_area - intersect_area

    IoU = intersect_area / union_area

    return IoU


"""
Diese Funktion ist aktuell so gebaut, dass anchorboxes NICHT ber체cksichtigt werden, 
also es gibt nur eine Anchorbox. Es wird also keine Unterdr체ckung geringerer IoU durchgef체hrt.
Das sollte bei Gelegenheit noch erg채nzt werden.
"""


def yolo_loss(y_true, y_pred):
    # define variables from input tensors y_true and y_pred with shape (None, 32, 32, 5) respectively.
    y_pred_box_prob = y_pred[:, :, :, 0]  # shape = (None, 32, 32, 1)
    y_pred_x = y_pred[:, :, :, 1]  # shape = (None, 32, 32, 1)
    y_pred_y = y_pred[:, :, :, 2]  # shape = (None, 32, 32, 1)
    y_pred_w = y_pred[:, :, :, 3]  # shape = (None, 32, 32, 1)
    y_pred_h = y_pred[:, :, :, 4]  # shape = (None, 32, 32, 1)

    y_true_box_prob = y_true[:, :, :, 0]  # shape = (None, 32, 32, 1)
    y_true_x = y_true[:, :, :, 1]  # shape = (None, 32, 32, 1)
    y_true_y = y_true[:, :, :, 2]  # shape = (None, 32, 32, 1)
    y_true_w = y_true[:, :, :, 3]  # shape = (None, 32, 32, 1)
    y_true_h = y_true[:, :, :, 4]  # shape = (None, 32, 32, 1)

    # define response mask: object in grid, yes or no?
    response_mask = y_true[:, :, :, 0]  # shape = # shape = (None, 32, 32, 1)

    # set value for lamba_coord (default from paper = 5) and lambda_noobj (default from paper = 0.5)
    lambda_coord = 5
    lambda_noobj = 0.5

    # calculate sum of squared x and y differences
    x_diff = y_true_x - y_pred_x
    y_diff = y_true_y - y_pred_y

    x_and_y_loss = lambda_coord * response_mask * (K.square(x_diff) + K.square(y_diff))

    # calculate sum of squared w and h differences
    w_diff = K.sqrt(y_true_w) - K.sqrt(y_pred_w)
    h_diff = K.sqrt(y_true_h) - K.sqrt(y_pred_h)

    w_and_h_loss = lambda_coord * response_mask * (K.square(w_diff) + K.square(h_diff))

    # calculate sum of squared box prob differences
    # box_prob_diff = response_mask * K.square(y_true_box_prob - y_pred_box_prob)
    no_object_loss = lambda_noobj * (1 - response_mask) * K.square(0 - y_pred_box_prob)
    object_loss = response_mask * K.square(1 - y_pred_box_prob)

    box_prob_loss = no_object_loss + object_loss
                    # + box_prob_diff

    # calculate yolo loss
    loss = K.sum(x_and_y_loss + w_and_h_loss + box_prob_loss)

    return loss


