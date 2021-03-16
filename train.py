import incubator26.yolo.data_preparation as dp
from incubator26.yolo.model import make_yolov3_model
from incubator26.utils import datareader, preprocessor
from incubator26.yolo import loss
import numpy as np
import os

reader = datareader.DataReader()
preprocess = preprocessor.PreProcessor()

# Create Training Data
train_img_folder = "./yolo_data/training_data/images"
train_annotation_folder = "./yolo_data/training_data/annotations"

train_img_files = reader.get_list_of_data(train_img_folder)
m, n_C, n_W, n_H = len(train_img_files), 1, 768, 768
grid_size = 24
number_of_grids = int(n_W / grid_size)

X_train = np.ndarray(shape=(m, n_H, n_W, n_C))
Y_train = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

all_bounding_boxes = []

anchor_boxes = [(5, 0.25)]

for index, img in enumerate(train_img_files):
    img_filepath = os.path.join(train_img_folder, img)
    image, size = reader.read_image(img_filepath)
    original_W = size[1]
    original_H = size[0]

    image = preprocess.preprocess_image(target_size=(n_W, n_H),
                                        image=image
                                        )
    X_train[index] = image

    json_filepath = os.path.join(train_annotation_folder, img.split(".")[0] + ".json")
    bounding_boxes = reader.get_bounding_boxes_from_json(json_filepath)
    bounding_boxes_yolo_format = dp.preprocess_bounding_boxes(bounding_boxes=bounding_boxes,
                                                              src_size=(original_H, original_W),
                                                              target_size=(n_H, n_W),
                                                              grid_size=grid_size
                                                              )
    dp.find_best_anchor_box_with_IoU(bounding_box_meta=bounding_boxes_yolo_format,
                                     anchor_boxes=anchor_boxes
                                     )

    y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                                                          output_shape=(number_of_grids, number_of_grids, 5)
                                                          )

    Y_train[index] = y_tensor


# Create Testing Data
test_img_folder = "./yolo_data/testing_data/images"
test_annotation_folder = "./yolo_data/testing_data/annotations"

test_img_files = reader.get_list_of_data(test_img_folder)
m, n_C, n_W, n_H = len(test_img_files), 1, 768, 768
grid_size = 24
number_of_grids = int(n_W / grid_size)

X_test = np.ndarray(shape=(m, n_H, n_W, n_C))
Y_test = np.ndarray(shape=(m, number_of_grids, number_of_grids, 5))

all_bounding_boxes = []

anchor_boxes = [(5, 0.25)]

for index, img in enumerate(test_img_files):
    img_filepath = os.path.join(test_img_folder, img)
    image, size = reader.read_image(img_filepath)
    original_W = size[1]
    original_H = size[0]

    image = preprocess.preprocess_image(target_size=(n_W, n_H),
                                        image=image
                                        )
    X_test[index] = image

    json_filepath = os.path.join(test_annotation_folder, img.split(".")[0] + ".json")
    bounding_boxes = reader.get_bounding_boxes_from_json(json_filepath)
    bounding_boxes_yolo_format = dp.preprocess_bounding_boxes(bounding_boxes=bounding_boxes,
                                                              src_size=(original_H, original_W),
                                                              target_size=(n_H, n_W),
                                                              grid_size=grid_size
                                                              )
    dp.find_best_anchor_box_with_IoU(bounding_box_meta=bounding_boxes_yolo_format,
                                     anchor_boxes=anchor_boxes
                                     )

    y_tensor = dp.convert_bounding_boxes_to_numpy_ndarray(bounding_boxes_list_of_dicts=bounding_boxes_yolo_format,
                                                          output_shape=(number_of_grids, number_of_grids, 5)
                                                          )

    Y_test[index] = y_tensor

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

model = make_yolov3_model()
model.summary()

# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.compile(loss=loss.yolo_loss,
              optimizer='adam')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
train_flow = train_datagen.flow(X_train, Y_train, batch_size=10)

test_datagen = ImageDataGenerator()
test_flow = test_datagen.flow(X_test, Y_test)

model.fit(x=train_flow,
          steps_per_epoch = 15,
          epochs = 50,
          verbose = 1,
#           workers= 4,
          validation_data = test_flow,
#           validation_steps = int(len(X_val) // batch_size),
           callbacks=[
              mcp_save
          ]
         )
