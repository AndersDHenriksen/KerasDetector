import numpy as np
import pathlib
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator  # Currently not used


def train_test_split(X, y, test_split_ratio, random_state=0):
    # Could also be done by sklearn
    rng = np.random.RandomState(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    n_split = int(test_split_ratio * X.shape[0])
    train_indices, test_indices = indices[n_split:], indices[:n_split]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


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

    print('Loading data ...', end='')
    # Load data from files
    data_path = pathlib.Path('../Data/GolfBall/images')
    label_path = pathlib.Path('../Data/GolfBall/ball_uv')
    input_all = np.array([np.load(i) for i in data_path.glob('*.npy')])
    y_all = np.array([np.load(i) for i in label_path.glob('*.npy')])
    print('Done')

    # Split and rescale
    X_train, X_test, y_train, y_test = train_test_split(input_all, y_all, config.test_split_ratio)
    X_train, X_test = X_train.astype(np.float) / 255, X_test.astype(np.float) / 255

    # The following two lines of code would have been a better way to augment if it changed y also.
    # datagen = ImageDataGenerator(rescale=1./255, width_shift_range=4, height_shift_range=4, vertical_flip=True)
    # self.batch_generator = datagen.flow(self.input, self.y, batch_size=self.config.batch_size)

    def next_batch():
        while True:
            idx = np.random.choice(y_train.shape[0], config.batch_size)
            X, y = X_train[idx], y_train[idx]
            vertical_flip_random(X, y)
            height_width_shift_random(X, y)
            yield X, y

    # return (next_batch() for _ in range(config.num_iter_per_epoch)), (X_test, y_test)
    return next_batch(), (X_test, y_test), (y_train.std(axis=0), y_train.mean(axis=0))


