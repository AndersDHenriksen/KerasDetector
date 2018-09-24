import numpy as np
from pathlib import Path
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator  # Currently not used
import h5py
from dl_tools.io import HDF5DatasetWriter
from operator import itemgetter


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
    X_padded = np.pad(X, ((0, 0), (height_range, height_range), (width_range, width_range), (0, 0)), 'edge')
    for i in range(batch_size):
        X[i, ...] = X_padded[i, height_shift[i]: height_shift[i] + h, width_shift[i]: width_shift[i] + w, :]
    y -= np.array([width_shift - width_range, height_shift - height_range]).T


class GolfSequence(Sequence):

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


def get_data(config):

    print('Loading data ... ', end='', flush=True)
    # Load data from files
    data_path = Path('../Data/GolfHosel/images')
    label_path = Path('../Data/GolfHosel/hosel_uv')
    if config.hdf5_path == "": # Work in RAM
        input_all = np.array([np.load(i) for i in data_path.glob('*.npy')])
        y_all = np.array([np.load(i) for i in label_path.glob('*.npy')])
        np.save(str(data_path) + '.npy', input_all)
        np.save(str(label_path) + '.npy', y_all)

        # Split and rescale
        X_train, X_test, y_train, y_test = train_test_split(input_all, y_all, config.test_split_ratio)
        X_train, X_test = X_train.astype(np.float) / 255, X_test.astype(np.float) / 255

    else:
        hdf5_files = [Path(config.hdf5_path) / (data_type + ".hdf5") for data_type in ["train", "test"]]
        if not all([f.exists() for f in hdf5_files]):
            print('Writing data to HDF5 ... ', end='', flush=True)
            X_paths = [p for p in data_path.glob('*.npy')]
            y_paths = [p for p in label_path.glob('*.npy')]

            X_paths_train, X_paths_test, y_paths_train, y_paths_test = train_test_split(X_paths, y_paths, config.test_split_ratio)
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
                    y = np.load(y_path).astype(np.float) / 255.0
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


