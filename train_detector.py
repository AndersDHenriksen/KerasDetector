from pathlib import Path
from dl_tools.nn.ConvTransform import ConvTransform, CenterOfMass
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint
from dl_tools.data_loader import data_generator
from dl_tools.utils.read_config import process_config
from dl_tools.utils.freeze_tools import finalize_for_ocv
from dl_tools.utils.tensorboardtools import tensorboard_launch
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import keras.backend as K


def train():
    # read config JSON file
    config = process_config(Path(__file__).parent / 'config.json')

    # load data
    data_gen, validation_data, scale_factor_wb = data_generator.get_data(config)

    # load model or create new
    if config.load_model:
        center_of_mass = CenterOfMass([s // 4 for s in config.input_shape[:2]], (4, 4))
        custom_objects = {'center_of_mass': center_of_mass}
        model = load_model(str(config.load_model), custom_objects=custom_objects)
        K.set_value(model.optimizer.lr, config.learning_rate)
    else:
        # initialize the optimizer and model
        opt = Adam(lr=config.learning_rate)
        model = ConvTransform.build(config)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mae"])
    # print network info
    model.summary()

    # define callbacks. Learning rate decrease, tensorboard etc.
    model_checkpoint = EpochCheckpoint(config.checkpoint_dir, start_epoch=config.model_epoch, best_limit=600)
    tensorboard = TensorBoard(log_dir=config.log_dir)
    callbacks = [model_checkpoint, tensorboard]
    if config.use_learning_rate_decay:
        learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=1, cooldown=100)
        callbacks.append(learning_rate_decay)

    # launch tensorboard
    tensorboard_launch(config.experiment_folder)

    # train the network
    H = model.fit_generator(
        generator=data_gen,
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
        save_model_path = config.checkpoint_dir + "final_model.hdf5"
        model.save(save_model_path)
        finalize_for_ocv(save_model_path)


if __name__ == '__main__':
    train()

# -------------------------------------------------------------------------------------------------------------------- #
# usage: conda activate keras
# usage: python train_detector.py
# usage: tensorboard --logdir "/home/ahe/TensorFlow/experiments/GolfHosel"
# -------------------------------------------------------------------------------------------------------------------- #
# net = cv2.dnn.readNetFromTensorflow(pb_path)
# blob = cv2.dnn.blobFromImage(npy_image, scalefactor=1/255.0)
# net.setInput(blob)
# predictions = net.forward()
# -------------------------------------------------------------------------------------------------------------------- #
# Install environment with:
# conda create -n keras
# conda install keras-gpu
# conda install munch
# -------------------------------------------------------------------------------------------------------------------- #
