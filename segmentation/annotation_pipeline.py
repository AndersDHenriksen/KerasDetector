import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
try:
    import imgaug.augmenters as iaa
except ImportError:
    pass


BAG_WIDTH = 140


def convert_annotation_to_dot_map(annotation_path):
    img_size = (2048, 2448)
    annotations = np.load(annotation_path).astype(np.int)
    dot_map = np.zeros(img_size, np.bool)
    dot_map[annotations[:, 1], annotations[:, 0]] = True
    return dot_map


def dot_map_to_prop_map(dot_map):
    radius = int(BAG_WIDTH // 2.5)
    d = 2 * radius + 1
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    R = np.sqrt(X**2 + Y**2)
    cone_pattern = np.clip(1 - R/radius, a_min=0, a_max=1)
    prop_map = np.zeros((dot_map.shape[0] + 2 * radius, dot_map.shape[1] + 2 * radius), np.float32)
    for dot_location in np.argwhere(dot_map):
        prop_map[dot_location[0]:dot_location[0] + d, dot_location[1]:dot_location[1] + d] += cone_pattern
    return prop_map[radius:-radius, radius:-radius].clip(max=1)


class DataGenerator:
    def __init__(self, data_path, glob_pattern="*.png"):
        self.image_paths = list(Path(data_path).glob(glob_pattern))
        random.Random(42).shuffle(self.image_paths)
        self.annotation_paths = [ip.parent / (ip.stem + "_annotation.npy") for ip in self.image_paths]
        self.n = len(self.image_paths)
        assert self.n, "DataGenerator could not find any samples"

    def __call__(self):
        for ip, ap in zip(self.image_paths, self.annotation_paths):
            X = np.array(load_img(ip, color_mode="grayscale"))
            Y = dot_map_to_prop_map(convert_annotation_to_dot_map(ap))
            X = X[::2, :2432:2, None]
            Y = Y[::2, :2432:2, None]
            yield X, Y


def rescale(img, heatmap):
    return tf.cast(img, tf.float32) / 255, heatmap


def get_datasets(data_path, batch_size=16, test_split_ratio=0.15, cache_data=True, use_imgaug=True):
    data_gen = DataGenerator(data_path, "*.png")
    ds = tf.data.Dataset.from_generator(data_gen, (tf.uint8, tf.float32), ([1024, 1216, 1], [1024, 1216, 1]))
    # ds = ds.shuffle(data_gen.n)  # not needed anymore DataGenerator is doing the shuffling
    n_test = round(test_split_ratio * data_gen.n)
    test_dataset = ds.take(n_test)
    train_dataset = ds.skip(n_test)
    if cache_data:
        train_dataset, test_dataset = train_dataset.cache(), test_dataset.cache()
    else:
        print("Warning: Not caching dataset in memory, risk of decoding the every images over and over again. Consider caching to file")
    train_dataset, test_dataset = train_dataset.batch(batch_size), test_dataset.batch(batch_size)
    if use_imgaug:
        train_dataset = train_dataset.map(np_augmentation_wrapper)
    else:
        train_dataset = train_dataset.map(tf_augmentation)
    return train_dataset.map(rescale), test_dataset.map(rescale)


def np_augmentation(images, heatmaps):
    seq = iaa.Sequential([
        iaa.GaussianBlur((0, 1.0)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.GammaContrast((0.8, 1.2)),
        iaa.Affine(scale=(0.95, 1.05), rotate=(-180, 180), translate_px={"x": (-50, 50), 'y': (-50, 50)})
    ])
    return seq(images=images.numpy(), heatmaps=heatmaps.numpy())


def np_augmentation_wrapper(image, mask):
    image_shape, mask_shape = image.shape, mask.shape
    [image, mask] = tf.py_function(np_augmentation, [image, mask], [tf.uint8, tf.float32])
    image.set_shape(image_shape)
    mask.set_shape(mask_shape)
    return image, mask


def tf_augmentation(images, heatmaps):
    # First non GT altering transform
    images = tf.image.random_brightness(images, 0.1)
    # Ground Truth altering transforms
    img_and_gt = tf.concat((tf.cast(images, tf.float32), heatmaps), axis=-1)
    img_and_gt = tf.image.random_flip_left_right(img_and_gt)
    img_and_gt = tf.image.random_flip_up_down(img_and_gt)
    images, heatmaps = img_and_gt[:, :, :, :-1], img_and_gt[:, :, :, -1:]
    return images, heatmaps


def benchmark(dataset, num_epochs=5):
    import time
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)


def test_pipeline(dataset):
    import matplotlib.pyplot as plt
    from EvalTest import add_overlay_to_image
    dataset_np = dataset.as_numpy_iterator()
    images, heatmaps = dataset_np.next()
    for image, heatmap in zip(images, heatmaps):
        fig = plt.figure()
        fig.set_size_inches(18, 10, forward=True)
        plt.imshow(add_overlay_to_image(image, heatmap, alpha=1))
        plt.waitforbuttonpress()
        plt.close(fig)


if __name__ == "__main__":
    data_path = r"D:\DataMeasured\BinPicking"
    train_dataset, test_dataset = get_datasets(data_path)
    # test_pipeline(train_dataset)
    benchmark(train_dataset, num_epochs=1)  # Pre run to cache DS
    benchmark(train_dataset)  # 39.2 sec | np aug 128 | tf aug 81
    benchmark(train_dataset.prefetch(tf.data.experimental.AUTOTUNE))  # 39.0 sec | np aug 124 | tf aug 78
    _ = 'bp'
