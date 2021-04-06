from incubator26.yolo.model import make_yolov3_model
from incubator26.utils import preprocessor
from incubator26.yolo import loss, callbacks
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# Define constants
TRAIN_IMG_FOLDER = "./yolo_data/training_data/images"
TRAIN_ANNOTATION_FOLDER = "./yolo_data/training_data/annotations"
TEST_IMG_FOLDER = "./yolo_data/testing_data/images"
TEST_ANNOTATION_FOLDER = "./yolo_data/testing_data/annotations"
n_C, n_W, n_H = 1, 512, 512
GRID_SIZE = 16
ANCHOR_BOXES = [(5, 0.25)]
TEST_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 8
EPOCHS = 100

# Generate train and test data
preprocess = preprocessor.PreProcessor()
X_train, X_test, Y_train, Y_test = preprocess.generate_train_test_data(
    input_shape=(n_C, n_W, n_H),
    grid_size=GRID_SIZE,
    train_img_folder=TRAIN_IMG_FOLDER,
    train_annotation_folder=TRAIN_ANNOTATION_FOLDER,
    test_img_folder=TEST_IMG_FOLDER,
    test_annotation_folder=TEST_ANNOTATION_FOLDER,
    anchor_boxes=ANCHOR_BOXES,
)

# Define data flow for training and testing data
train_datagen = ImageDataGenerator()
train_flow = train_datagen.flow(X_train, Y_train, batch_size=TRAIN_BATCH_SIZE)

test_datagen = ImageDataGenerator()
test_flow = test_datagen.flow(X_test, Y_test, batch_size=TEST_BATCH_SIZE)

# Compile Model
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    model = make_yolov3_model((n_H, n_W, n_C))
    model.compile(loss=loss.yolo_loss, optimizer="adam")
    model.summary()


# Set callbacks
mcp_save = callbacks.MyModelCheckpoint()
tensorboard_callback = callbacks.MyTensorBoard("logs")
lr_scheduler = LearningRateScheduler(
    callbacks.MyLearningRateScheduler.lr_exp_decay(), verbose=1
)

# Fit the model
history = model.fit(
    x=train_flow,
    epochs=EPOCHS,
    verbose=1,
    validation_data=test_flow,
    callbacks=[mcp_save, lr_scheduler, tensorboard_callback],
)
