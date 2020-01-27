import numpy as np
from pathlib import Path
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import h5py
from dl_tools.io import HDF5DatasetWriter
from dl_tools.utils.file_tools import partition_dataset
from operator import itemgetter
ocv_present = True
try:
    import cv2
except ImportError:
    ocv_present = False


def get_data_for_classification(config, split_data_files=True):
    target_size = config.input_shape[:2]
    if split_data_files:
        # Move to train / test directory
        if not (config.data_folder / 'Train').exists():
            partition_dataset(config.data_folder, 1 - config.test_split_ratio, 0, config.test_split_ratio, True)
        config.data_folder_train = str(config.data_folder / 'Train')
        config.data_folder_test = str(config.data_folder / 'Test')

        # Load data generators
        aug_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=4, height_shift_range=4,
                                     brightness_range=[0.95, 1.05])  # zoom_range=0.05
        rescale_gen = ImageDataGenerator(rescale=1. / 255)
        train_gen = aug_gen.flow_from_directory(config.data_folder_train, batch_size=config.batch_size,
                                                class_mode='categorical', target_size=target_size)
        validation_gen = rescale_gen.flow_from_directory(config.data_folder_test, batch_size=config.batch_size,
                                                         class_mode='categorical', target_size=target_size)
        oversample_image_generator(train_gen)
        oversample_image_generator(validation_gen)
    else:
        # Load data generators
        data_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=4, height_shift_range=4, rotation_range=360,
                                      vertical_flip=True, horizontal_flip=True, validation_split=0.15)
        train_gen = data_gen.flow_from_directory(config.data_folder, batch_size=config.batch_size,
                                                 class_mode='categorical', subset='training', target_size=target_size)
        validation_gen = data_gen.flow_from_directory(config.data_folder, batch_size=config.batch_size,
                                                      class_mode='categorical', subset='validation', target_size=target_size)
    return train_gen, validation_gen


def oversample_image_generator(image_generator, n_samples=None):
    current_classes = image_generator.classes
    current_filepaths = np.array(image_generator.filepaths)
    class_count = np.bincount(current_classes)
    n_samples = n_samples or class_count.max()
    new_classes, new_filepaths = [], []
    for class_idx in range(class_count.size):
        n_add = n_samples - class_count[class_idx]
        if n_add <= 0:
            continue
        new_classes += n_add * [class_idx]
        n_repeats = n_add // class_count[class_idx] + 1
        new_filepaths += np.tile(current_filepaths[current_classes == class_idx], n_repeats)[:n_add].tolist()
    image_generator.classes = np.concatenate((current_classes, new_classes))
    image_generator._filepaths += new_filepaths
    image_generator.n = image_generator.samples = image_generator.classes.size
    image_generator._set_index_array()


def train_test_split(X, y, test_split_ratio, random_state=0):
    # Could also be done by sklearn
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0] if isinstance(X, np.ndarray) else len(X))
    rng.shuffle(indices)
    n_split = int(test_split_ratio * indices.size)
    train_indices, test_indices = indices[n_split:], indices[:n_split]
    if isinstance(X, np.ndarray):
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    return tuple(itemgetter(*indices)(data) for data in [X, y] for indices in [train_indices, test_indices])


def vertical_flip_random(X, y):
    width = X.shape[2]
    do_flip = np.random.randint(2, size=y.shape[0]).astype(np.bool)
    X[do_flip, ...] = X[do_flip, :, ::-1, :]
    y[do_flip, 0] = width - 1 - y[do_flip, 0]


def height_width_shift_random(X, y, width_range=2, height_range=2):
    batch_size, h, w = X.shape[:3]
    width_shift = np.random.randint(2 * width_range + 1, size=batch_size)
    height_shift = np.random.randint(2 * height_range + 1, size=batch_size)
    if ocv_present:
        X_padded = np.array([cv2.copyMakeBorder(x, height_range, height_range, width_range, width_range,
                                                cv2.BORDER_REPLICATE) for x in X])
    else:
        X_padded = np.pad(X, ((0, 0), (height_range, height_range), (width_range, width_range), (0, 0)), 'edge')
    for i in range(batch_size):
        X[i, ...] = X_padded[i, height_shift[i]: height_shift[i] + h, width_shift[i]: width_shift[i] + w, :]
    y -= np.array([width_shift - width_range, height_shift - height_range]).T


class DataSequence(Sequence):

    def __init__(self, X_train, y_train, batch_size, num_iter_per_epoch=None):
        self.X_train, self.y_train = X_train, y_train
        self.batch_size = batch_size
        self.num_iter_per_epoch = num_iter_per_epoch or int(np.ceil(y_train.shape[0] / float(batch_size)))

    def __len__(self):
        return self.num_iter_per_epoch

    def __getitem__(self, idx):
        idx = np.random.choice(self.y_train.shape[0], self.batch_size)
        X, y = self.X_train[idx], self.y_train[idx]
        vertical_flip_random(X, y)
        height_width_shift_random(X, y)
        return X, y


def get_data_for_detection(config):

    print('Loading data ... ', end='', flush=True)
    split_test_data_from_train_data = False

    # Load data from files
    data_path = Path('/home/ahe/TensorFlow/data/GolfHosel/train/images')
    label_path = Path('/home/ahe/TensorFlow/data/GolfHosel/train/hosel_uv')
    if not split_test_data_from_train_data:
        data_path_test = Path('/home/ahe/TensorFlow/data/GolfHosel/test/images')
        label_path_test = Path('/home/ahe/TensorFlow/data/GolfHosel/test/hosel_uv')

    # Get image/label paths from data paths
    X_paths = [p for p in data_path.glob('*.npy')]
    y_paths = [p for p in label_path.glob('*.npy')]
    if split_test_data_from_train_data:
        X_paths_train, X_paths_test, y_paths_train, y_paths_test = train_test_split(X_paths, y_paths, config.test_split_ratio)
    else:
        X_paths_train, y_paths_train = X_paths, y_paths
        X_paths_test = [p for p in data_path_test.glob('*.npy')]
        y_paths_test = [p for p in label_path_test.glob('*.npy')]

    # Load images and labels
    if config.hdf5_path == "":  # Work in RAM
        X_train, X_test, y_train, y_test = [np.array([np.load(i) for i in path_list]) for path_list in
                                            [X_paths_train, X_paths_test, y_paths_train, y_paths_test]]
        # Rescale
        X_train, X_test = X_train.astype(np.float) / 255, X_test.astype(np.float) / 255

    else:
        hdf5_files = [Path(config.hdf5_path) / (data_type + ".hdf5") for data_type in ["train", "test"]]
        if not all([f.exists() for f in hdf5_files]):
            print('Writing data to HDF5 ... ', end='', flush=True)
            data_sets = [(hdf5_files[0], X_paths_train, y_paths_train), (hdf5_files[1], X_paths_test, y_paths_test)]

            for output_path, X_paths, y_paths in data_sets:
                writer = HDF5DatasetWriter((len(X_paths), *config.input_shape), output_path, label_dim=2, label_dtype="float")
                try:
                    from tqdm import tqdm
                    iterator = tqdm(zip(X_paths, y_paths), total=len(X_paths))
                except ImportError:
                    iterator = zip(X_paths, y_paths)
                for X_path, y_path in iterator:
                    X = np.load(X_path).astype(np.float) / 255.0
                    y = np.load(y_path)
                    writer.add([X], [y])
                writer.close()

        dbs = [h5py.File(f) for f in hdf5_files]
        X_train, y_train, X_test, y_test = dbs[0]["X"], dbs[0]["y"], dbs[1]["X"], dbs[1]["y"]
    print('Done', flush=True)

    # The following two lines of code would have been a better way to augment if it changed y also.
    # datagen = ImageDataGenerator(rescale=1./255, width_shift_range=4, height_shift_range=4, vertical_flip=True)
    # self.batch_generator = datagen.flow(self.input, self.y, batch_size=self.config.batch_size)

    def next_batch():
        while True:
            idx = np.random.choice(y_train.shape[0], config.batch_size, replace=False)
            idx = list(np.sort(idx))
            X, y = X_train[idx], y_train[idx]
            height_width_shift_random(X, y)
            yield X, y

    # return (next_batch() for _ in range(config.num_iter_per_epoch)), (X_test, y_test)
    return next_batch(), (X_test, y_test), (y_train[:].std(axis=0), y_train[:].mean(axis=0))


