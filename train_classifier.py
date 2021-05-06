from pathlib import Path
from dl_tools.nn.mediumnet import MediumNet
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint
from dl_tools.data_loader.data_generator import get_data_for_classification
from dl_tools.utils.read_config import process_config
from dl_tools.utils.eval_tools import confusion_matrix, show_errors
from dl_tools.utils.freeze_tools import finalize_for_ocv_tf2
from dl_tools.utils.tensorboardtools import tensorboard_launch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau


def train():
    # read config JSON file
    config = process_config(Path(__file__).parent / 'config.json')

    # load data
    train_gen, validation_gen = get_data_for_classification(config)

    # load model or create new
    if config.load_model:
        model = load_model(str(config.load_model))
        if config.learning_rate:
            model.optimizer.lr = config.learning_rate
    else:
        # initialize the optimizer and model
        opt = Adam(lr=config.learning_rate)
        model = MediumNet.build(config, train_gen.num_classes)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # print network info
    model.summary()

    # define callbacks. Learning rate decrease, tensorboard etc.
    model_checkpoint = EpochCheckpoint(config.checkpoint_dir, best_limit=0.3)
    tensorboard = TensorBoard(log_dir=config.log_dir, profile_batch=0)
    callbacks = [model_checkpoint, tensorboard]
    if config.use_learning_rate_decay:
        learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=1, cooldown=100)
        callbacks.append(learning_rate_decay)

    # launch tensorboard
    tensorboard_launch(config.experiment_folder)

    # train the network
    H = model.fit(train_gen,
        epochs=config.training_epochs,
        callbacks=callbacks,
        validation_data=validation_gen,
        initial_epoch=config.model_epoch)

    confusion_matrix(config, model, validation_gen)
    if input("Enter 1 to see errors: ") == "1":
        show_errors(model, validation_gen)
    if input("Enter 1 to convert best model for OpenCV inference") == "1":
        best_model_path = sorted(Path(config.checkpoint_dir).glob("best*.hdf5"))[-1]
        finalize_for_ocv_tf2(str(best_model_path))


if __name__ == '__main__':
    train()
