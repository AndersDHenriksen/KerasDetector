from pathlib import Path
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, DepthwiseConv2D, Dropout, Flatten, Dense, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from dl_tools.utils.read_config import process_config
from dl_tools.utils.tensorboardtools import tensorboard_launch
from dl_tools.callbacks.epochcheckpoint import EpochCheckpoint


# Read config file
config = process_config(Path(__file__).parent / 'config_transferlearning.json')


# Build transfer learning network
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(config.input_shape))
for layer in base_model.layers:
    layer.trainable = False

# Build new top
X = base_model.output
# X = DepthwiseConv2D((7, 7), activation='relu')(X)
X = MaxPooling2D((7, 7))(X)
X = Flatten()(X)
X = Dropout(0.5)(X)
X = Dense(64, activation='relu')(X)
X = Dropout(0.5)(X)
X = Dense(1, activation='sigmoid')(X)
model = Model(inputs=base_model.input, outputs=X)
model.summary()

# Optimizer and build model
opt = Adam(lr=config.learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Move to train / test directory
data_path = Path(config.data_folder)
if not (data_path / 'train').exists():
    image_paths = list((data_path / 'new').rglob('*.jpg'))
    import numpy as np
    rng = np.random.RandomState(0)
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)
    n_split = int(0.15 * indices.size)
    train_indices, test_indices = indices[n_split:], indices[:n_split]
    for type, indices in zip(['train', 'test'], [train_indices, test_indices]):
        Path.mkdir(data_path / type / 'NoBeads', parents=True, exist_ok=True)
        Path.mkdir(data_path / type / 'WithBeads', parents=True, exist_ok=True)
        for i in indices:
            image_paths[i].replace(str(image_paths[i]).replace('new', type))
config.data_folder_train, config.data_folder_test = str(data_path / 'train'), str(data_path / 'test')

# Load data generators
target_size = config.input_shape[:2]
aug_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=4, height_shift_range=4,
                             rotation_range=360, vertical_flip=True, horizontal_flip=True)
rescale_gen = ImageDataGenerator(rescale=1. / 255)
train_gen = aug_gen.flow_from_directory(config.data_folder_train, batch_size=config.batch_size,
                                        class_mode='binary', target_size=target_size)
validation_gen = rescale_gen.flow_from_directory(config.data_folder_test, batch_size=config.batch_size,
                                                 class_mode='binary', target_size=target_size)

# # Load data generators
# data_gen = ImageDataGenerator(rescale=1. / 255, samplewise_center=False, width_shift_range=4, height_shift_range=4,
#                               rotation_range=360, vertical_flip=True, horizontal_flip=True, validation_split=0.15)
# train_gen = data_gen.flow_from_directory(config.data_folder, batch_size=config.batch_size,
#                                          class_mode='binary', subset='training', target_size=target_size)
# validation_gen = data_gen.flow_from_directory(config.data_folder, batch_size=config.batch_size,
#                                               class_mode='binary', subset='validation', target_size=target_size)

# define callbacks. Learning rate decrease, tensorboard etc.
model_checkpoint = EpochCheckpoint(config.checkpoint_dir, start_epoch=config.model_epoch, best_limit=0.3)
tensorboard = TensorBoard(log_dir=config.log_dir)
callbacks = [model_checkpoint, tensorboard]
if config.use_learning_rate_decay:
    learning_rate_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, cooldown=30)
    callbacks.append(learning_rate_decay)

# launch tensorboard
tensorboard_launch(config.experiment_folder)

# train the network
H = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_gen.samples // config.batch_size,
    epochs=config.training_epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // config.batch_size,
    initial_epoch=config.model_epoch)

# After warm-up do fine tuning
for layer in model.layers:
    layer.trainable = True
opt = Adam(lr=config.learning_rate / 10)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

H = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_gen.samples // config.batch_size,
    epochs=config.training_epochs * 10,
    verbose=1,
    callbacks=callbacks,
    validation_data=validation_gen,
    validation_steps=validation_gen.samples // config.batch_size,
    initial_epoch=config.model_epoch)

print('Fine tuning done. Now take best model and pass it through freeze_tools.finalize_for_ocv')
