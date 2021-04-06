from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import math
import os


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self):
        self.filepath = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.save_best_only = True
        self.monitor = "val_loss"
        self.mode = "min"
        super().__init__(
            filepath=self.filepath,
            save_best_only=self.save_best_only,
            monitor=self.monitor,
            mode=self.mode,
        )


class MyLearningRateScheduler:
    def __init__(
        self,
        initial_learning_rate: float = 0.001,
        epochs: int = 100,
        decay: float = None,
        k: float = 0.1,
        drop_rate: float = 0.5,
        epochs_drop: float = 10,
    ):
        self.initial_learning_rate = initial_learning_rate
        self.epochs = epochs
        if decay == None:
            self.decay = self.initial_learning_rate / self.epochs
        else:
            self.decay = decay
        self.k = k
        self.drop_rate = drop_rate
        self.epochs_drop = epochs_drop

    def lr_time_based_decay(self, epoch, lr):
        return lr * 1 / (1 + self.decay * epoch)

    def lr_exp_decay(self, epoch, lr):
        return self.initial_learning_rate * math.exp(-self.k * epoch)

    def lr_step_decay(self, epoch, lr):

        return self.initial_learning_rate * math.pow(
            self.drop_rate, math.floor(epoch / self.epochs_drop)
        )


class MyTensorBoard(TensorBoard):
    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.create_log_dir()
        self.set_log_file_name()
        super().__init__(self.log_dir)

    def create_log_dir(self):
        if not os.path.exists(self.log_directory):
            os.mkdir(self.log_directory)

    def set_log_file_name(self):
        self.log_dir = self.log_directory + datetime.now().strftime("%Y%m%d-%H%M%S")
