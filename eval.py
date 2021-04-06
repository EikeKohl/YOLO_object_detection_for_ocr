import tensorflow as tf
import numpy as np
from incubator26.utils import datareader, preprocessor, plots
import incubator26.yolo.data_preparation as dp

TARGET_SIZE = (512, 512)
NON_MAX_SUPRESSION = 0.5
NUMBER_OF_GRIDS = 32
GRID_SIZE = 16
MODEL = "weight.hdf5"
EVAL_IMG_FILEPATH = "./yolo_data/training_data/images/00040534.png"


# instantiate DataReader and PreProcessor
reader = datareader.DataReader()
preprocess = preprocessor.PreProcessor()


# Read and preprocess the image to be evaluated
image, size = reader.read_image(EVAL_IMG_FILEPATH)
png_array = preprocess.preprocess_image(target_size=TARGET_SIZE, image=image)
png_array = np.expand_dims(png_array, axis=0)


# Load the model to be evaluated
loaded_model = tf.keras.models.load_model(MODEL, compile=False)
loaded_model.summary()

# Make prediction on image
prediction = loaded_model.predict(png_array)


# Decode the prediction
list_of_bounding_boxes = dp.decode_yolo_model_output(
    prediction[-1], NON_MAX_SUPRESSION, NUMBER_OF_GRIDS, GRID_SIZE
)


# Plot the image + the bounding boxes predicted
plots.plot_image_from_array(
    img_array=png_array[0, :, :, 0],
    bounding_boxes=list_of_bounding_boxes,
    height=20,
    width=20,
    plot_grids=True,
    number_of_grids=GRID_SIZE,
)
