from dl_tools.nn.mediumnet import MediumNet
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint
from dl_tools.data_loader import data_generator
from dl_tools.utils.read_config import process_config
from dl_tools.utils.freeze_tools import save_frozen_protobuf, augment_for_ocv
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import keras.backend as K

# read config JSON file
config = process_config(r"./golf_config.json")

# load data
data_generator, validation_data, scale_factor_wb = data_generator.get_data(config)

# load model or create new
if config.load_model:
    model = load_model(config.load_model)
    K.set_value(model.optimizer.lr, config.learning_rate)
else:
    # initialize the optimizer and model
    opt = Adam(lr=config.learning_rate)
    model = MediumNet.build(config, 2, softmax=False, scale_adjust_wb=scale_factor_wb)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mae"])
# print network info
print(model.summary())

# define callbacks. Learning rate decrease, tensorboard etc.
model_checkpoint = EpochCheckpoint(config.checkpoint_dir, start_epoch=config.model_epoch, best_limit=4)
tensorboard = TensorBoard(log_dir=config.log_dir)
callbacks = [model_checkpoint, tensorboard]
if config.use_learning_rate_decay:
    learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, cooldown=30)
    callbacks.append(learning_rate_decay)

# train the network
H = model.fit_generator(
    generator=data_generator,
    steps_per_epoch=config.num_iter_per_epoch,  # defaults to len(Sequence)
    epochs=config.training_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_data,
    initial_epoch=config.model_epoch)

# evaluate the network
if config.do_evaluate:
    pass
    # from sklearn.metrics import classification_report
    # predictions = model.predict_on_batch(validation_data[0])
    # print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

if config.save_final_model:
    model.save(config.checkpoint_dir + "final_model.hdf5")
    model = augment_for_ocv(model)
    save_frozen_protobuf(config.checkpoint_dir + "final_model.pb", K.get_session(),
                         output_names=[out.op.name for out in model.outputs])

# -------------------------------------------------------------------------------------------------------------------- #
# usage: conda activate keras
# usage: python train_detector.py
# usage: tensorboard --logdir "C:\Users\ahe\Google Drive\TrackMan\01. FullSwing\DeepLearning\experiments"
# -------------------------------------------------------------------------------------------------------------------- #
# import cv2, pathlib, numpy
# net = cv2.dnn.readNetFromTensorflow(pb_path)
# image = numpy.load(next(pathlib.Path('../Data/GolfBall/images').glob('*.npy')))
# blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0)
# net.setInput(blob)
# predictions = net.forward()
# -------------------------------------------------------------------------------------------------------------------- #
