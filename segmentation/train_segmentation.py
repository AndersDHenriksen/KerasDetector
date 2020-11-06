import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add module to path
from segmentation.annotation_helper import get_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import tensorflow.keras.backend as K
from dl_tools.utils.read_config import process_config
from dl_tools.utils.tensorboardtools import tensorboard_launch
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint
import segmentation_models as sm
from keras_unet.models import custom_unet


def sigmoid_iou_loss(y_true, y_pred):
    return 1 - K.sum(K.minimum(y_true, y_pred)) / K.sum(K.maximum(y_true, y_pred))


# read config JSON file
config = process_config(Path(__file__).parent / 'config.json')

if config.disable_gpu:
    print('Disabling GPU! Computations will be done on CPU.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load data
preprocess_input = None  # sm.get_preprocessing('resnet18')
X_train_gen, Y_train_gen, X_test, Y_test = get_data(config.data_folder, config.batch_size, config.test_split_ratio, preprocess_input)

# load model or create new
if config.load_model:
    model = load_model(str(config.load_model), compile=False)
else:
    model = custom_unet(input_shape=config.input_shape, use_batch_norm=True, num_classes=1,
                        filters=4, dropout=0.2, output_activation='sigmoid')
    # model = sm.Unet('resnet18', input_shape=config.input_shape, classes=1, activation='sigmoid',
    #                 decoder_filters=(256, 128, 64, 32, 16), encoder_freeze=False)
opt = Adam(lr=config.learning_rate)
model.compile(opt, sm.losses.bce_jaccard_loss, metrics=['mean_squared_error', sm.metrics.iou_score,
                                                        sm.losses.bce_jaccard_loss, sigmoid_iou_loss])

# print network info
model.summary()

# evaluate
# from EvalTest import save_overlay_images
# save_overlay_images(model, X_test)

# define callbacks. Learning rate decrease, tensorboard etc.
model_checkpoint = EpochCheckpoint(config.checkpoint_dir)
tensorboard = TensorBoard(log_dir=config.log_dir, profile_batch=0)
callbacks = [model_checkpoint, tensorboard]
if config.use_learning_rate_decay:
    learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, verbose=1, cooldown=50)
    callbacks.append(learning_rate_decay)

# launch tensorboard
tensorboard_launch(config.experiment_folder)

# train the network
H = model.fit(
    x=zip(X_train_gen, Y_train_gen),
    steps_per_epoch=X_train_gen.n // config.batch_size,
    epochs=config.training_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=(X_test, Y_test),
    initial_epoch=config.model_epoch)

if config.save_final_model:
    model.save(config.checkpoint_dir + "final_model.hdf5")
