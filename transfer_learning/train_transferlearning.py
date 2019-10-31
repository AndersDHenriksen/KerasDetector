import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # Add module to path
from dl_tools.utils.read_config import process_config
from dl_tools.nn.TL_MobileNet2 import MobileNet2
from dl_tools.utils.tensorboardtools import tensorboard_launch
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint
from dl_tools.data_loader.data_generator import get_data_for_classification
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau


def train(model=None):
    # Read config file
    config = process_config(Path(__file__).parent / 'config_transferlearning.json')
    training_epoch = config.training_epochs_warmup

    # Load or define transfer learning model
    if model is None and config.load_model:
        model = load_model(str(config.load_model))
        model.is_in_warmup = bool([layer for layer in model.layers if not layer.trainable])
        learning_rate = config.learning_rate_warmup if model.is_in_warmup else config.learning_rate
        if learning_rate:
            model.optimizer.lr = learning_rate
        if not model.is_in_warmup:
            training_epoch += config.training_epochs
    elif model is None:
        model = MobileNet2.build(config, classes=1)
        opt = Adam(lr=config.learning_rate_warmup)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    train_gen, validation_gen = get_data_for_classification(config)

    # Define callbacks. Learning rate decrease, tensorboard etc.
    model_checkpoint = EpochCheckpoint(config.checkpoint_dir, best_limit=0.3)
    tensorboard = TensorBoard(log_dir=config.log_dir)
    callbacks = [tensorboard, model_checkpoint]
    if config.use_learning_rate_decay:
        learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, cooldown=30)
        callbacks.append(learning_rate_decay)

    # launch tensorboard
    tensorboard_launch(config.experiment_folder)

    # train the network
    H = model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_gen.samples // config.batch_size,
        epochs=training_epoch,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_gen,
        validation_steps=validation_gen.samples // config.batch_size,
        initial_epoch=config.model_epoch)

    if not model.is_in_warmup:
        print('Fine tuning done. Now take best model and pass it through freeze_tools.finalize_for_ocv')
        return
    print("Model warmup done. Taking the last model and passing it through train again")

    # After warm-up prepare to do fine tuning
    for layer in model.layers:
        layer.trainable = True
    model.is_in_warmup = False
    model.optimizer.lr = config.learning_rate
    train(model)


if __name__ == '__main__':
    train()

