from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dl_tools.data_loader.data_generator import train_test_split


CONE_WIDTH = 140


def convert_annotation_to_dot_map(annotation_path):
    img_size = (2048, 2448)
    annotations = np.load(annotation_path).astype(np.int)
    dot_map = np.zeros(img_size, np.bool)
    dot_map[annotations[:, 1], annotations[:, 0]] = True
    return dot_map


def dot_map_to_prop_map(dot_map):
    radius = int(CONE_WIDTH // 2.5)
    d = 2 * radius + 1
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    R = np.sqrt(X**2 + Y**2)
    cone_pattern = 255 * np.clip(1 - R/radius, a_min=0, a_max=1)
    prop_map = np.zeros((dot_map.shape[0] + 2 * radius, dot_map.shape[1] + 2 * radius), np.float)
    for dot_location in np.argwhere(dot_map):
        prop_map[dot_location[0]:dot_location[0] + d, dot_location[1]:dot_location[1] + d] += cone_pattern
    return prop_map[radius:-radius, radius:-radius].clip(max=255)


def get_all_dot_maps(annotation_folder):
    return np.array([convert_annotation_to_dot_map(ap) for ap in sorted(Path(annotation_folder).glob("*.npy"))])


def get_all_prop_maps(annotation_folder):
    return np.array([dot_map_to_prop_map(dm) for dm in get_all_dot_maps(annotation_folder)])


def get_all_images(image_folder):
    return np.array([np.array(load_img(ip, color_mode="grayscale")) for ip in sorted(Path(image_folder).glob("*.png"))])


def get_data(data_path, batch_size=16, test_split_ratio=0.15, preprocessor=None):
    # TODO consider cutting small patches
    X_all, Y_all = get_all_images(data_path)[..., None], get_all_prop_maps(data_path)[..., None]
    if preprocessor is None:
        X_all = X_all.astype(np.float32) / 255
    else:
        X_all = preprocessor(X_all)
    Y_all = Y_all.astype(np.float32) / 255

    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_split_ratio)
    seed = 42
    data_gen_args = dict(width_shift_range=100, height_shift_range=100, rotation_range=360, vertical_flip=True, horizontal_flip=True)
    X_gen = ImageDataGenerator(**data_gen_args)
    Y_gen = ImageDataGenerator(**data_gen_args)
    X_gen.fit(X_train, augment=True, seed=seed)  # Don't think this is needed when no mean, std scaling
    Y_gen.fit(Y_train, augment=True, seed=seed)
    X_train_gen = X_gen.flow(X_train, batch_size=batch_size, seed=seed)
    Y_train_gen = Y_gen.flow(Y_train, batch_size=batch_size, seed=seed)
    return X_train_gen, Y_train_gen, X_test, Y_test


def test_get_data():
    X_train_gen, Y_train_gen, X_test, Y_test = get_data()
    id = 0  # Use whatever method you wish for a name with no collision.
    for images, masks in zip(X_train_gen, Y_train_gen):
        for image, mask in zip(images, masks):
            np.save('data/images/' + str(id), image)
            np.save('data/masks/' + str(id), mask)
            id += 1
        if id > 5:
            break
